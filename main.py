#!/usr/bin/env python3
"""
optiprompt - Автоматическая отладка и тестирование промптов для LLM.
Использует Ollama для генерации гипотез и тестирования.
Хранит все данные в текстовых файлах (Markdown).

Новые возможности (уточнение заказчика):
- Промпт может содержать подстановочные переменные в формате {{ variable }} (Liquid-подобный синтаксис).
- Данные для подстановки загружаются из второго файла (JSON или CSV) и должны содержать колонку "reference" (эталонный ответ).
- Система оптимизирует только шаблон промпта, не изменяя входные данные.

Запуск:
  python main.py --prompt "Переведи {{ text }} на французский" --data-file data.json
  python main.py --prompt-file template.txt --data-file data.csv --max-iterations 10

Формат файла данных (data.json):
[
  {"text": "Hello", "reference": "Bonjour"},
  {"text": "Goodbye", "reference": "Au revoir"}
]

Формат файла данных (data.csv):
text,reference
Hello,Bonjour
Goodbye,"Au revoir"
"""

import os
import re
import json
import csv
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

# ============================================================
# КОНФИГУРАЦИЯ
# ============================================================

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1"
STATE_DIR = Path.home() / ".optiprompt"
STATE_DIR.mkdir(exist_ok=True)

# ============================================================
# МОДЕЛИ ДАННЫХ
# ============================================================

@dataclass
class Hypothesis:
    """Гипотеза и результат её проверки."""
    text: str
    variant_prompt: str       # Это шаблон промпта
    score: float = 0.0

@dataclass
class IterationResult:
    """Результат одной итерации оптимизации."""
    number: int
    baseline_prompt: str      # Шаблон
    baseline_score: float
    hypotheses: List[Hypothesis] = field(default_factory=list)
    best_score: float = 0.0
    best_prompt: str = ""
    timestamp: str = ""

@dataclass
class TestCase:
    """Один тестовый случай: переменные для подстановки и эталонный ответ."""
    variables: Dict[str, str]
    reference: str

# ============================================================
# КЛИЕНТ OLLAMA
# ============================================================

class OllamaClient:
    """Клиент для взаимодействия с локальной Ollama."""

    def __init__(self, model: str = DEFAULT_MODEL, url: str = DEFAULT_OLLAMA_URL):
        self.model = model
        self.url = url.rstrip('/')

    def _call_ollama(self, prompt: str, system: str = "") -> str:
        import urllib.request
        import urllib.error

        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 2048
            }
        }).encode('utf-8')

        req = urllib.request.Request(
            f"{self.url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"}
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(req, timeout=120) as response:
                    data = json.loads(response.read().decode('utf-8'))
                    return data.get("response", "").strip()
            except urllib.error.URLError as e:
                if attempt == max_retries - 1:
                    raise ConnectionError(f"Не удалось подключиться к Ollama: {e}")
                time.sleep(2 ** attempt)

    def generate_hypotheses(self, prompt_template: str, score: float, count: int = 3) -> List[str]:
        system_msg = (
            "Ты — эксперт по промпт-инжинирингу. "
            "Проанализируй шаблон промпта (с переменными вида {{ variable }}) и предложи "
            "конкретные гипотезы по его улучшению. Отвечай СТРОГО в формате JSON-списка строк."
        )

        user_msg = f"""Проанализируй следующий шаблон промпта:

ПРОМПТ-ШАБЛОН:
---
{prompt_template}
---

Текущая оценка эффективности: {score:.3f} (0.0 - 1.0)

Предложи ровно {count} различных гипотез по улучшению этого шаблона.
Каждая гипотеза должна быть конкретной и действенной.
Не меняй имена переменных в фигурных скобках.

Формат ответа (ТОЛЬКО JSON):
["гипотеза 1", "гипотеза 2", "гипотеза 3"]"""

        response = self._call_ollama(user_msg, system_msg)
        return self._parse_hypotheses_json(response)

    def _parse_hypotheses_json(self, text: str) -> List[str]:
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, list):
                    return [str(h) for h in data]
            except json.JSONDecodeError:
                pass

        lines = []
        for line in text.split('\n'):
            line = line.strip()
            if line and (line.startswith(('- ', '* ', '1.', '2.', '3.'))):
                lines.append(re.sub(r'^[-*\d.]+\s*', '', line))
        if lines:
            return lines
        return [text.strip()]

    def apply_hypothesis(self, prompt_template: str, hypothesis: str) -> str:
        system_msg = (
            "Ты — инструмент для модификации шаблонов промптов. "
            "Примени предложенное улучшение к шаблону. "
            "Верни ТОЛЬКО изменённый шаблон (с теми же переменными {{ }}), без объяснений."
        )

        user_msg = f"""ИСХОДНЫЙ ШАБЛОН:
---
{prompt_template}
---

ГИПОТЕЗА ДЛЯ ПРИМЕНЕНИЯ:
{hypothesis}

НОВЫЙ ШАБЛОН (только сам шаблон):"""

        new_template = self._call_ollama(user_msg, system_msg)

        new_template = re.sub(r'^[#*\s]*Новый шаблон:?\s*', '', new_template, flags=re.IGNORECASE)
        new_template = re.sub(r'^```\w*\n', '', new_template)
        new_template = re.sub(r'\n```$', '', new_template)

        return new_template.strip()

    def test_prompt(self, prompt: str) -> str:
        return self._call_ollama(prompt)

    def judge_response(self, prompt: str, response: str, reference: str) -> float:
        system_msg = (
            "Ты — строгий оценщик. Оцени соответствие ответа эталону по шкале от 0.0 до 1.0. "
            "Отвечай ТОЛЬКО числом, например: 0.85"
        )
        user_msg = f"""ПРОМПТ: {prompt}
ОТВЕТ МОДЕЛИ: {response}
ЭТАЛОННЫЙ ОТВЕТ: {reference}

Насколько ответ соответствует эталону? Оценка (0.0-1.0):"""
        result = self._call_ollama(user_msg, system_msg)
        match = re.search(r'(\d+\.?\d*)', result)
        if match:
            score = float(match.group())
            return max(0.0, min(1.0, score))
        return 0.0

# ============================================================
# ШАБЛОНИЗАТОР
# ============================================================

def apply_template(template: str, variables: Dict[str, str]) -> str:
    """Заменяет {{ variable }} на значения из словаря."""
    def replacer(match):
        var_name = match.group(1).strip()
        return variables.get(var_name, match.group(0))  # если переменная не найдена, оставляем как есть
    return re.sub(r'\{\{\s*(.*?)\s*\}\}', replacer, template)

# ============================================================
# ОЦЕНЩИК
# ============================================================

class Evaluator:
    def __init__(self, test_cases: List[TestCase], criterion: str, ollama: OllamaClient):
        self.test_cases = test_cases
        self.criterion = criterion
        self.ollama = ollama

    def evaluate(self, template: str) -> Tuple[float, List[Dict]]:
        results = []
        total_score = 0.0

        for i, case in enumerate(self.test_cases):
            # Подставляем переменные в шаблон, получаем итоговый промпт
            final_prompt = apply_template(template, case.variables)
            response = self.ollama.test_prompt(final_prompt)
            score = self._score_response(response, case.reference)
            total_score += score

            results.append({
                "case_id": i + 1,
                "variables": case.variables,
                "prompt_used": final_prompt,
                "response": response,
                "reference": case.reference,
                "score": score
            })

        avg_score = total_score / len(self.test_cases) if self.test_cases else 0.0
        return avg_score, results

    def _score_response(self, response: str, reference: str) -> float:
        if self.criterion == "exact-match":
            return 1.0 if response.strip().lower() == reference.strip().lower() else 0.0
        elif self.criterion == "contains":
            return 1.0 if reference.strip().lower() in response.strip().lower() else 0.0
        elif self.criterion == "llm-judge":
            return self.ollama.judge_response("", response, reference)
        else:
            raise ValueError(f"Неизвестный критерий: {self.criterion}")

# ============================================================
# ОПТИМИЗАТОР
# ============================================================

class Optimizer:
    def __init__(self, ollama: OllamaClient, session_name: str):
        self.ollama = ollama
        self.session_dir = STATE_DIR / session_name
        self.session_dir.mkdir(exist_ok=True)
        self.history: List[IterationResult] = []

    def optimize(
        self,
        initial_template: str,
        test_cases: List[TestCase],
        criterion: str,
        max_iterations: int = 5,
        hypotheses_count: int = 3,
        patience: int = 2
    ) -> Tuple[str, float]:
        evaluator = Evaluator(test_cases, criterion, self.ollama)

        print("\n" + "=" * 60)
        print("ВЫЧИСЛЕНИЕ ЭТАЛОННОЙ ОЦЕНКИ...")
        print("=" * 60)

        best_template = initial_template
        best_score, baseline_results = evaluator.evaluate(best_template)
        self._save_evaluation("baseline", best_template, best_score, baseline_results)
        print(f"Эталонная оценка: {best_score:.3f}")

        iterations_without_improvement = 0

        for iteration in range(1, max_iterations + 1):
            print(f"\n{'=' * 60}")
            print(f"ИТЕРАЦИЯ {iteration}/{max_iterations}")
            print(f"{'=' * 60}")

            print("Генерация гипотез...")
            hypothesis_texts = self.ollama.generate_hypotheses(best_template, best_score, hypotheses_count)

            iter_result = IterationResult(
                number=iteration,
                baseline_prompt=best_template,
                baseline_score=best_score,
                timestamp=datetime.now().isoformat()
            )

            improved = False

            for i, hyp_text in enumerate(hypothesis_texts):
                print(f"\n  Гипотеза {i+1}/{len(hypothesis_texts)}: {hyp_text[:80]}...")
                try:
                    variant_template = self.ollama.apply_hypothesis(best_template, hyp_text)
                    score, results = evaluator.evaluate(variant_template)

                    hyp = Hypothesis(text=hyp_text, variant_prompt=variant_template, score=score)
                    iter_result.hypotheses.append(hyp)

                    self._save_iteration_hypothesis(iteration, i + 1, hyp, results)

                    status = "УЛУЧШЕНИЕ!" if score > best_score else "без улучшения"
                    sign = "+" if score > best_score else " "
                    print(f"    Оценка: {score:.3f} (эталон: {best_score:.3f}) [{sign}] {status}")

                    if score > best_score:
                        best_score = score
                        best_template = variant_template
                        improved = True

                except Exception as e:
                    print(f"    Ошибка: {e}")

            if iter_result.hypotheses:
                best_in_iter = max(iter_result.hypotheses, key=lambda h: h.score)
                iter_result.best_score = best_in_iter.score
                iter_result.best_prompt = best_in_iter.variant_prompt

            self.history.append(iter_result)

            if improved:
                iterations_without_improvement = 0
                print(f"\n  >>> Новый эталон! Оценка: {best_score:.3f}")
                self._save_best_prompt(best_template, best_score, iteration)
            else:
                iterations_without_improvement += 1
                print(f"\n  Без улучшений. Терпение: {iterations_without_improvement}/{patience}")

            if iterations_without_improvement >= patience:
                print(f"\nОстановка: нет улучшений {patience} итераций подряд.")
                break

        self._save_final_report(best_template, best_score)
        return best_template, best_score

    # ========== Сохранение результатов ==========
    def _save_evaluation(self, name: str, template: str, score: float, results: List[Dict]):
        filepath = self.session_dir / f"{name}_evaluation.md"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Оценка: {name}\n\n")
            f.write(f"**Дата:** {datetime.now().isoformat()}\n\n")
            f.write(f"**Средняя оценка:** {score:.3f}\n\n")
            f.write(f"## Шаблон промпта\n\n```\n{template}\n```\n\n")
            f.write(f"## Детальные результаты\n\n")
            f.write("| № | Переменные | Промпт (подставленный) | Ответ | Эталон | Оценка |\n")
            f.write("|---|------------|------------------------|-------|--------|--------|\n")
            for r in results:
                vars_str = json.dumps(r['variables'], ensure_ascii=False)
                prompt_used = r['prompt_used'].replace('\n', ' ')[:80]
                resp = r['response'].replace('\n', ' ')[:80]
                ref = r['reference'].replace('\n', ' ')[:80]
                f.write(f"| {r['case_id']} | {vars_str} | {prompt_used} | {resp} | {ref} | {r['score']:.2f} |\n")

    def _save_iteration_hypothesis(self, iteration: int, hyp_num: int, hyp: Hypothesis, results: List[Dict]):
        filepath = self.session_dir / f"iter{iteration:02d}_hyp{hyp_num}.md"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Итерация {iteration}, Гипотеза {hyp_num}\n\n")
            f.write(f"**Гипотеза:** {hyp.text}\n\n")
            f.write(f"**Оценка:** {hyp.score:.3f}\n\n")
            f.write(f"## Вариант шаблона\n\n```\n{hyp.variant_prompt}\n```\n\n")
            f.write(f"## Результаты тестов\n\n")
            f.write("| № | Ответ | Эталон | Оценка |\n")
            f.write("|---|-------|--------|--------|\n")
            for r in results:
                resp = r['response'].replace('\n', ' ')[:100]
                ref = r['reference'].replace('\n', ' ')[:100]
                f.write(f"| {r['case_id']} | {resp} | {ref} | {r['score']:.2f} |\n")

    def _save_best_prompt(self, template: str, score: float, iteration: int):
        filepath = self.session_dir / "best_prompt.txt"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Лучший шаблон (итерация {iteration}, оценка {score:.3f})\n\n")
            f.write(template)

    def _save_final_report(self, template: str, score: float):
        filepath = self.session_dir / "final_report.md"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Итоговый отчёт оптимизации\n\n")
            f.write(f"**Дата:** {datetime.now().isoformat()}\n\n")
            f.write(f"**Итоговая оценка:** {score:.3f}\n\n")
            f.write(f"**Количество итераций:** {len(self.history)}\n\n")
            f.write(f"## Лучший шаблон\n\n```\n{template}\n```\n\n")
            f.write(f"## История итераций\n\n")
            for it in self.history:
                best_in_iter = max(it.hypotheses, key=lambda h: h.score) if it.hypotheses else None
                f.write(f"### Итерация {it.number}\n")
                f.write(f"- Эталонная оценка: {it.baseline_score:.3f}\n")
                if best_in_iter:
                    f.write(f"- Лучшая оценка: {best_in_iter.score:.3f}\n")
                    f.write(f"- Лучшая гипотеза: {best_in_iter.text[:100]}...\n")
                f.write("\n")

# ============================================================
# ЗАГРУЗКА ДАННЫХ ДЛЯ ПОДСТАНОВКИ
# ============================================================

def load_data_file(filepath: str) -> List[TestCase]:
    """
    Загружает данные для подстановки из JSON или CSV файла.
    Каждая запись должна содержать поле "reference" (эталонный ответ).
    Остальные поля используются как переменные для шаблона.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Файл данных не найден: {filepath}")

    if filepath.suffix.lower() == '.json':
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        if not isinstance(raw_data, list):
            raise ValueError("JSON-файл должен содержать массив объектов")
        test_cases = []
        for item in raw_data:
            # ищем эталон
            reference = None
            for possible_key in ['reference', 'expected', '_reference']:
                if possible_key in item:
                    reference = str(item.pop(possible_key))
                    break
            if reference is None:
                raise ValueError(f"В объекте {item} отсутствует поле 'reference' или 'expected'")
            # оставшиеся поля – переменные
            variables = {k: str(v) for k, v in item.items()}
            test_cases.append(TestCase(variables=variables, reference=reference))
        return test_cases

    elif filepath.suffix.lower() == '.csv':
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            test_cases = []
            for row in reader:
                reference = None
                for possible_key in ['reference', 'expected', '_reference']:
                    if possible_key in row:
                        reference = row.pop(possible_key)
                        break
                if reference is None:
                    raise ValueError(f"CSV файл должен содержать колонку 'reference' или 'expected'")
                variables = {k: v for k, v in row.items()}
                test_cases.append(TestCase(variables=variables, reference=reference))
            return test_cases
    else:
        raise ValueError(f"Неподдерживаемый формат файла данных: {filepath.suffix}. Используйте .json или .csv")

# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="optiprompt — автоматическая оптимизация промптов (шаблонов) с подстановкой переменных",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python main.py --prompt "Переведи {{ text }} на французский" --data-file data.json
  python main.py --prompt-file template.txt --data-file data.csv --max-iterations 10
        """
    )

    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", "-p", help="Шаблон промпта с переменными вида {{ variable }}")
    prompt_group.add_argument("--prompt-file", "-f", help="Файл с шаблоном промпта")

    parser.add_argument("--data-file", "-d", required=True,
                        help="Файл с данными для подстановки (JSON или CSV). Должен содержать колонку 'reference'.")

    parser.add_argument("--max-iterations", "-m", type=int, default=5)
    parser.add_argument("--hypotheses-count", "-n", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2,
                        help="Остановка после N итераций без улучшений")
    parser.add_argument("--criterion", "-c", choices=["exact-match", "contains", "llm-judge"],
                        default="contains", help="Критерий оценки")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Модель Ollama (по умолчанию: {DEFAULT_MODEL})")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL,
                        help=f"URL Ollama API (по умолчанию: {DEFAULT_OLLAMA_URL})")
    parser.add_argument("--session", "-s", default=None,
                        help="Имя сессии (по умолчанию генерируется по дате)")

    args = parser.parse_args()

    # Загрузка шаблона
    if args.prompt_file:
        template = Path(args.prompt_file).read_text(encoding='utf-8').strip()
    else:
        template = args.prompt

    if not template:
        print("Ошибка: шаблон промпта не может быть пустым.")
        return 1

    # Загрузка данных для подстановки
    try:
        test_cases = load_data_file(args.data_file)
    except Exception as e:
        print(f"Ошибка загрузки файла данных: {e}")
        return 1

    if not test_cases:
        print("Ошибка: файл данных не содержит записей.")
        return 1

    # Проверка наличия переменных в шаблоне
    required_vars = set()
    for case in test_cases:
        required_vars.update(case.variables.keys())
    # не обязательно, но предупредим, если какие-то переменные отсутствуют в шаблоне
    template_vars = set(re.findall(r'\{\{\s*(.*?)\s*\}\}', template))
    missing_in_template = required_vars - template_vars
    if missing_in_template:
        print(f"Предупреждение: переменные {missing_in_template} из данных не найдены в шаблоне.")

    # Сессия
    session_name = args.session or datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ollama
    ollama = OllamaClient(model=args.model, url=args.ollama_url)

    print(f"\n{'=' * 60}")
    print(f"OPTIPROMPT — Оптимизация шаблона промпта")
    print(f"{'=' * 60}")
    print(f"Сессия:           {session_name}")
    print(f"Модель:           {args.model}")
    print(f"Тестовых записей: {len(test_cases)}")
    print(f"Критерий:         {args.criterion}")
    print(f"Макс. итераций:   {args.max_iterations}")
    print(f"Гипотез/итерацию: {args.hypotheses_count}")
    print(f"Шаблон промпта:\n{'-' * 40}\n{template}\n{'-' * 40}")

    print("\nПроверка подключения к Ollama...")
    try:
        ollama._call_ollama("ping", "")
        print("Соединение с Ollama установлено.")
    except Exception as e:
        print(f"\nОшибка подключения к Ollama: {e}")
        return 1

    optimizer = Optimizer(ollama, session_name)
    start_time = time.time()
    try:
        best_template, best_score = optimizer.optimize(
            initial_template=template,
            test_cases=test_cases,
            criterion=args.criterion,
            max_iterations=args.max_iterations,
            hypotheses_count=args.hypotheses_count,
            patience=args.patience
        )
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
        return 130
    except Exception as e:
        print(f"\nКритическая ошибка: {e}")
        return 1

    elapsed = time.time() - start_time

    print(f"\n{'=' * 60}")
    print(f"ОПТИМИЗАЦИЯ ЗАВЕРШЕНА")
    print(f"{'=' * 60}")
    print(f"Время выполнения:  {elapsed:.1f} сек")
    print(f"Количество итераций: {len(optimizer.history)}")
    print(f"Итоговая оценка:   {best_score:.3f}")
    print(f"\nЛучший шаблон:\n{'-' * 40}\n{best_template}\n{'-' * 40}")
    print(f"\nРезультаты сохранены в: {optimizer.session_dir}")

    return 0

if __name__ == "__main__":
    exit(main())