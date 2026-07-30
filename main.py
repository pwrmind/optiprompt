#!/usr/bin/env python3
"""
optiprompt - Автоматическая отладка и тестирование промптов для LLM.
Использует Ollama для генерации гипотез и тестирования.
Хранит все данные в текстовых файлах (Markdown).

Запуск:
  python main.py --prompt "Переведи текст на французский" --test-data tests.md
  python main.py --prompt-file my_prompt.txt --test-data tests.md --max-iterations 10
  python main.py --prompt "Напиши стих" --test-data tests.md --criterion contains

Форматы тестовых данных:
  1. Q: вопрос
     A: ответ
  2. ## Заголовок
     **Вход:** текст
     **Эталон:** текст
  3. input|reference (по одному на строку)
"""

import os
import re
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict


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
    variant_prompt: str
    score: float = 0.0


@dataclass
class IterationResult:
    """Результат одной итерации оптимизации."""
    number: int
    baseline_prompt: str
    baseline_score: float
    hypotheses: List[Hypothesis] = field(default_factory=list)
    best_score: float = 0.0
    best_prompt: str = ""
    timestamp: str = ""


@dataclass
class TestCase:
    """Один тестовый случай: входные данные и эталонный ответ."""
    input_text: str
    reference: str


# ============================================================
# КЛИЕНТ OLLAMA
# ============================================================

class OllamaClient:
    """
    Клиент для взаимодействия с локальной Ollama.
    Использует только стандартную библиотеку Python (urllib).
    """

    def __init__(self, model: str = DEFAULT_MODEL, url: str = DEFAULT_OLLAMA_URL):
        self.model = model
        self.url = url.rstrip('/')

    def _call_ollama(self, prompt: str, system: str = "") -> str:
        """
        Вызов Ollama через HTTP API.
        Endpoint: POST /api/generate
        """
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

    def generate_hypotheses(self, prompt: str, score: float, count: int = 3) -> List[str]:
        """
        Генерация гипотез по улучшению промпта.
        Возвращает список строк-гипотез.
        """
        system_msg = (
            "Ты — эксперт по промпт-инжинирингу. "
            "Проанализируй промпт и предложи конкретные гипотезы по его улучшению. "
            "Отвечай СТРОГО в формате JSON-списка строк."
        )

        user_msg = f"""Проанализируй следующий промпт:

ПРОМПТ:
---
{prompt}
---

Текущая оценка эффективности: {score:.3f} (0.0 - 1.0)

Предложи ровно {count} различных гипотез по улучшению этого промпта.
Каждая гипотеза должна быть конкретной и действенной.

Формат ответа (ТОЛЬКО JSON):
["гипотеза 1", "гипотеза 2", "гипотеза 3"]"""

        response = self._call_ollama(user_msg, system_msg)
        return self._parse_hypotheses_json(response)

    def _parse_hypotheses_json(self, text: str) -> List[str]:
        """
        Извлечение списка гипотез из ответа LLM.
        Пытается найти JSON-массив, иначе парсит строки с дефисами/цифрами.
        """
        # Ищем JSON массив в тексте
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, list):
                    return [str(h) for h in data]
            except json.JSONDecodeError:
                pass

        # Fallback: строки с дефисами или цифрами
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            if line and (line.startswith(('- ', '* ', '1.', '2.', '3.'))):
                lines.append(re.sub(r'^[-*\d.]+\s*', '', line))

        if lines:
            return lines

        # Последний fallback — весь текст как одна гипотеза
        return [text.strip()]

    def apply_hypothesis(self, prompt: str, hypothesis: str) -> str:
        """
        Применяет гипотезу к промпту, возвращает новый вариант промпта.
        """
        system_msg = (
            "Ты — инструмент для модификации промптов. "
            "Примени предложенное улучшение к промпту. "
            "Верни ТОЛЬКО измененный промпт, без объяснений."
        )

        user_msg = f"""ИСХОДНЫЙ ПРОМПТ:
---
{prompt}
---

ГИПОТЕЗА ДЛЯ ПРИМЕНЕНИЯ:
{hypothesis}

НОВЫЙ ПРОМПТ (только сам промпт, без комментариев):"""

        new_prompt = self._call_ollama(user_msg, system_msg)

        # Очистка от возможных маркеров и обёрток
        new_prompt = re.sub(r'^[#*\s]*Новый промпт:?\s*', '', new_prompt, flags=re.IGNORECASE)
        new_prompt = re.sub(r'^```\w*\n', '', new_prompt)
        new_prompt = re.sub(r'\n```$', '', new_prompt)

        return new_prompt.strip()

    def test_prompt(self, prompt: str) -> str:
        """
        Тестовый прогон промпта (без системного сообщения).
        Используется для получения ответа модели на тестовых данных.
        """
        return self._call_ollama(prompt)

    def judge_response(self, prompt: str, response: str, reference: str) -> float:
        """
        LLM-оценщик: насколько ответ соответствует эталону.
        Возвращает оценку от 0.0 до 1.0.
        """
        system_msg = (
            "Ты — строгий оценщик. Оцени соответствие ответа эталону по шкале от 0.0 до 1.0. "
            "Отвечай ТОЛЬКО числом, например: 0.85"
        )

        user_msg = f"""ПРОМПТ: {prompt}

ОТВЕТ МОДЕЛИ: {response}

ЭТАЛОННЫЙ ОТВЕТ: {reference}

Насколько ответ соответствует эталону? Оценка (0.0-1.0):"""

        result = self._call_ollama(user_msg, system_msg)

        # Извлекаем число из ответа
        match = re.search(r'(\d+\.?\d*)', result)
        if match:
            score = float(match.group())
            return max(0.0, min(1.0, score))
        return 0.0


# ============================================================
# ОЦЕНЩИК
# ============================================================

class Evaluator:
    """
    Оценивает качество промпта на наборе тестовых кейсов.
    Поддерживает критерии: exact-match, contains, llm-judge.
    """

    def __init__(self, test_cases: List[TestCase], criterion: str, ollama: OllamaClient):
        self.test_cases = test_cases
        self.criterion = criterion
        self.ollama = ollama

    def evaluate(self, prompt: str) -> Tuple[float, List[Dict]]:
        """
        Прогоняет промпт на всех тестовых случаях.
        Возвращает (средняя_оценка, список_результатов).
        """
        results = []
        total_score = 0.0

        for i, case in enumerate(self.test_cases):
            # Формируем финальный промпт с входными данными
            if case.input_text:
                final_prompt = f"{prompt}\n\n{case.input_text}"
            else:
                final_prompt = prompt

            # Получаем ответ модели
            response = self.ollama.test_prompt(final_prompt)

            # Оцениваем
            score = self._score_response(response, case.reference)
            total_score += score

            results.append({
                "case_id": i + 1,
                "input": case.input_text,
                "response": response,
                "reference": case.reference,
                "score": score
            })

        avg_score = total_score / len(self.test_cases) if self.test_cases else 0.0
        return avg_score, results

    def _score_response(self, response: str, reference: str) -> float:
        """Оценка одного ответа согласно выбранному критерию."""
        if self.criterion == "exact-match":
            # Точное совпадение (без учёта регистра и пробелов)
            return 1.0 if response.strip().lower() == reference.strip().lower() else 0.0

        elif self.criterion == "contains":
            # Эталон содержится в ответе
            return 1.0 if reference.strip().lower() in response.strip().lower() else 0.0

        elif self.criterion == "llm-judge":
            # LLM-оценка
            return self.ollama.judge_response("", response, reference)

        else:
            raise ValueError(f"Неизвестный критерий: {self.criterion}")


# ============================================================
# ОПТИМИЗАТОР
# ============================================================

class Optimizer:
    """
    Основной движок оптимизации промптов.
    Реализует итеративный цикл: гипотезы -> проверка -> выбор лучшего.
    Все результаты сохраняются в текстовые Markdown-файлы.
    """

    def __init__(self, ollama: OllamaClient, session_name: str):
        self.ollama = ollama
        self.session_dir = STATE_DIR / session_name
        self.session_dir.mkdir(exist_ok=True)
        self.history: List[IterationResult] = []

    def optimize(
        self,
        initial_prompt: str,
        test_cases: List[TestCase],
        criterion: str,
        max_iterations: int = 5,
        hypotheses_count: int = 3,
        patience: int = 2
    ) -> Tuple[str, float]:
        """
        Запускает цикл оптимизации.
        Возвращает (лучший_промпт, его_оценка).
        """
        evaluator = Evaluator(test_cases, criterion, self.ollama)

        # --- Эталонный прогон ---
        print("\n" + "=" * 60)
        print("ВЫЧИСЛЕНИЕ ЭТАЛОННОЙ ОЦЕНКИ...")
        print("=" * 60)

        best_prompt = initial_prompt
        best_score, baseline_results = evaluator.evaluate(best_prompt)
        self._save_evaluation("baseline", best_prompt, best_score, baseline_results)

        print(f"Эталонная оценка: {best_score:.3f}")

        iterations_without_improvement = 0

        # --- Цикл итераций ---
        for iteration in range(1, max_iterations + 1):
            print(f"\n{'=' * 60}")
            print(f"ИТЕРАЦИЯ {iteration}/{max_iterations}")
            print(f"{'=' * 60}")

            # 1. Генерация гипотез
            print("Генерация гипотез...")
            hypothesis_texts = self.ollama.generate_hypotheses(
                best_prompt, best_score, hypotheses_count
            )

            iter_result = IterationResult(
                number=iteration,
                baseline_prompt=best_prompt,
                baseline_score=best_score,
                timestamp=datetime.now().isoformat()
            )

            improved = False

            # 2. Тестирование каждой гипотезы
            for i, hyp_text in enumerate(hypothesis_texts):
                print(f"\n  Гипотеза {i+1}/{len(hypothesis_texts)}: {hyp_text[:80]}...")

                try:
                    # Применяем гипотезу -> получаем вариант промпта
                    variant_prompt = self.ollama.apply_hypothesis(best_prompt, hyp_text)

                    # Оцениваем вариант
                    score, results = evaluator.evaluate(variant_prompt)

                    hyp = Hypothesis(
                        text=hyp_text,
                        variant_prompt=variant_prompt,
                        score=score
                    )
                    iter_result.hypotheses.append(hyp)

                    # Сохраняем в файл
                    self._save_iteration_hypothesis(iteration, i + 1, hyp, results)

                    # Вывод
                    status = "УЛУЧШЕНИЕ!" if score > best_score else "без улучшения"
                    sign = "+" if score > best_score else " "
                    print(f"    Оценка: {score:.3f} (эталон: {best_score:.3f}) [{sign}] {status}")

                    # Обновление эталона
                    if score > best_score:
                        best_score = score
                        best_prompt = variant_prompt
                        improved = True

                except Exception as e:
                    print(f"    Ошибка: {e}")

            # Сохраняем лучшее в итерации
            if iter_result.hypotheses:
                best_in_iter = max(iter_result.hypotheses, key=lambda h: h.score)
                iter_result.best_score = best_in_iter.score
                iter_result.best_prompt = best_in_iter.variant_prompt

            self.history.append(iter_result)

            # 3. Проверка на раннюю остановку
            if improved:
                iterations_without_improvement = 0
                print(f"\n  >>> Новый эталон! Оценка: {best_score:.3f}")
                self._save_best_prompt(best_prompt, best_score, iteration)
            else:
                iterations_without_improvement += 1
                print(f"\n  Без улучшений. Терпение: {iterations_without_improvement}/{patience}")

            if iterations_without_improvement >= patience:
                print(f"\nОстановка: нет улучшений {patience} итераций подряд.")
                break

        # Сохраняем финальный отчёт
        self._save_final_report(best_prompt, best_score)

        return best_prompt, best_score

    # =================================================================
    # МЕТОДЫ СОХРАНЕНИЯ В ФАЙЛЫ
    # =================================================================

    def _save_evaluation(self, name: str, prompt: str, score: float, results: List[Dict]):
        """Сохраняет результаты оценки в Markdown-файл."""
        filepath = self.session_dir / f"{name}_evaluation.md"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Оценка: {name}\n\n")
            f.write(f"**Дата:** {datetime.now().isoformat()}\n\n")
            f.write(f"**Средняя оценка:** {score:.3f}\n\n")
            f.write(f"## Промпт\n\n```\n{prompt}\n```\n\n")
            f.write(f"## Детальные результаты\n\n")
            f.write("| № | Вход | Ответ | Эталон | Оценка |\n")
            f.write("|---|------|-------|--------|--------|\n")
            for r in results:
                resp = r['response'].replace('\n', ' ')[:80]
                ref = r['reference'].replace('\n', ' ')[:80]
                inp = r.get('input', '').replace('\n', ' ')[:60]
                f.write(f"| {r['case_id']} | {inp} | {resp} | {ref} | {r['score']:.2f} |\n")

    def _save_iteration_hypothesis(self, iteration: int, hyp_num: int, hyp: Hypothesis, results: List[Dict]):
        """Сохраняет результат проверки одной гипотезы."""
        filepath = self.session_dir / f"iter{iteration:02d}_hyp{hyp_num}.md"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Итерация {iteration}, Гипотеза {hyp_num}\n\n")
            f.write(f"**Гипотеза:** {hyp.text}\n\n")
            f.write(f"**Оценка:** {hyp.score:.3f}\n\n")
            f.write(f"## Вариант промпта\n\n```\n{hyp.variant_prompt}\n```\n\n")
            f.write(f"## Результаты тестов\n\n")
            f.write("| № | Ответ | Эталон | Оценка |\n")
            f.write("|---|-------|--------|--------|\n")
            for r in results:
                resp = r['response'].replace('\n', ' ')[:100]
                ref = r['reference'].replace('\n', ' ')[:100]
                f.write(f"| {r['case_id']} | {resp} | {ref} | {r['score']:.2f} |\n")

    def _save_best_prompt(self, prompt: str, score: float, iteration: int):
        """Сохраняет текущий лучший промпт."""
        filepath = self.session_dir / "best_prompt.txt"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Лучший промпт (итерация {iteration}, оценка {score:.3f})\n\n")
            f.write(prompt)

    def _save_final_report(self, prompt: str, score: float):
        """Сохраняет итоговый отчёт."""
        filepath = self.session_dir / "final_report.md"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Итоговый отчёт оптимизации\n\n")
            f.write(f"**Дата:** {datetime.now().isoformat()}\n\n")
            f.write(f"**Итоговая оценка:** {score:.3f}\n\n")
            f.write(f"**Количество итераций:** {len(self.history)}\n\n")
            f.write(f"## Лучший промпт\n\n```\n{prompt}\n```\n\n")
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
# ЗАГРУЗКА ТЕСТОВЫХ ДАННЫХ
# ============================================================

def load_test_cases_from_file(filepath: str) -> List[TestCase]:
    """
    Загружает тестовые кейсы из текстового/Markdown файла.

    Поддерживаемые форматы:

    1. Q/A формат:
       Q: вопрос
       A: ответ

    2. Markdown с заголовками:
       ## Название теста
       **Вход:**
       текст входа
       **Эталон:**
       эталонный ответ

    3. Простой формат с разделителем |:
       input|reference
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Файл тестовых данных не найден: {filepath}")

    content = filepath.read_text(encoding='utf-8')
    test_cases = []

    # --- Попытка 1: Формат Q/A ---
    qa_pattern = re.findall(r'Q:\s*(.*?)\nA:\s*(.*?)(?:\n\n|\n$|$)', content, re.DOTALL)
    if qa_pattern:
        for q, a in qa_pattern:
            test_cases.append(TestCase(input_text=q.strip(), reference=a.strip()))
        if test_cases:
            return test_cases

    # --- Попытка 2: Markdown с заголовками ---
    sections = re.split(r'\n##\s*', content)
    # Первый элемент может быть "шапкой" до первого заголовка, пропускаем если есть другие
    if len(sections) > 1:
        sections = sections[1:]  # отбрасываем текст до первого ##
    for section in sections:
        # Ищем **Вход:** или **Input:** или **Вопрос:**
        input_match = re.search(
            r'\*\*(?:Вход|Input|Вопрос)\*\s*:?\s*\n?(.*?)(?:\n\*\*|\n##|\Z)',
            section, re.DOTALL | re.IGNORECASE
        )
        # Ищем **Эталон:** или **Reference:** или **Ответ:**
        ref_match = re.search(
            r'\*\*(?:Эталон|Reference|Ответ)\*\s*:?\s*\n?(.*?)(?:\n\*\*|\n##|\Z)',
            section, re.DOTALL | re.IGNORECASE
        )
        if input_match and ref_match:
            test_cases.append(TestCase(
                input_text=input_match.group(1).strip(),
                reference=ref_match.group(1).strip()
            ))
    if test_cases:
        return test_cases

    # --- Попытка 3: Простой формат input|reference ---
    for line in content.strip().split('\n'):
        line = line.strip()
        if '|' in line and not line.startswith('|') and not line.startswith('#'):
            parts = line.split('|')
            if len(parts) >= 2:
                test_cases.append(TestCase(
                    input_text=parts[0].strip(),
                    reference=parts[1].strip()
                ))
    if test_cases:
        return test_cases

    # Если ничего не найдено
    return []


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="optiprompt — автоматическая оптимизация промптов через Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Базовый запуск
  python main.py --prompt "Переведи текст на французский" --test-data tests.md

  # Загрузка промпта из файла
  python main.py --prompt-file my_prompt.txt --test-data tests.md

  # С расширенными параметрами
  python main.py --prompt "Напиши стих" --test-data tests.md \\
      --max-iterations 10 --hypotheses-count 5 --criterion llm-judge

  # Выбор модели Ollama
  python main.py --prompt "Классифицируй текст" --test-data tests.md --model mistral

Форматы файла тестовых данных (tests.md):
  1. Q: вопрос
     A: ответ
  2. ## Заголовок
     **Вход:** текст
     **Эталон:** текст
  3. input|reference
        """
    )

    # --- Входные данные ---
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--prompt", "-p",
        help="Исходный промпт (строка)"
    )
    prompt_group.add_argument(
        "--prompt-file", "-f",
        help="Файл с исходным промптом"
    )

    parser.add_argument(
        "--test-data", "-t",
        required=True,
        help="Файл с тестовыми данными (Markdown или TXT)"
    )

    # --- Параметры оптимизации ---
    parser.add_argument(
        "--max-iterations", "-m",
        type=int, default=5,
        help="Максимальное количество итераций (по умолчанию: 5)"
    )
    parser.add_argument(
        "--hypotheses-count", "-n",
        type=int, default=3,
        help="Количество гипотез на итерацию (по умолчанию: 3)"
    )
    parser.add_argument(
        "--patience",
        type=int, default=2,
        help="Остановка после N итераций без улучшений (по умолчанию: 2)"
    )
    parser.add_argument(
        "--criterion", "-c",
        choices=["exact-match", "contains", "llm-judge"],
        default="contains",
        help="Критерий оценки (по умолчанию: contains)"
    )

    # --- Параметры Ollama ---
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Модель Ollama (по умолчанию: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help=f"URL Ollama API (по умолчанию: {DEFAULT_OLLAMA_URL})"
    )

    # --- Вывод ---
    parser.add_argument(
        "--session", "-s",
        default=None,
        help="Имя сессии для сохранения результатов (по умолчанию: авто-генерация по дате)"
    )

    args = parser.parse_args()

    # ================================================================
    # 1. Загрузка промпта
    # ================================================================
    if args.prompt_file:
        prompt_path = Path(args.prompt_file)
        if not prompt_path.exists():
            print(f"Ошибка: файл с промптом не найден: {args.prompt_file}")
            return 1
        prompt = prompt_path.read_text(encoding='utf-8').strip()
    else:
        prompt = args.prompt

    if not prompt:
        print("Ошибка: промпт не может быть пустым.")
        return 1

    # ================================================================
    # 2. Загрузка тестовых данных
    # ================================================================
    try:
        test_cases = load_test_cases_from_file(args.test_data)
    except Exception as e:
        print(f"Ошибка загрузки тестовых данных: {e}")
        return 1

    if not test_cases:
        print("Ошибка: не удалось найти ни одного тестового случая в файле.")
        print("Поддерживаемые форматы:")
        print("  1. Q: вопрос")
        print("     A: ответ")
        print("  2. ## Заголовок")
        print("     **Вход:** текст")
        print("     **Эталон:** текст")
        print("  3. input|reference (по одному на строку)")
        return 1

    # ================================================================
    # 3. Инициализация сессии
    # ================================================================
    if args.session:
        session_name = args.session
    else:
        session_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ================================================================
    # 4. Проверка подключения к Ollama
    # ================================================================
    ollama = OllamaClient(model=args.model, url=args.ollama_url)

    print(f"\n{'=' * 60}")
    print(f"OPTIPROMPT — Оптимизация промптов")
    print(f"{'=' * 60}")
    print(f"Сессия:           {session_name}")
    print(f"Модель:           {args.model}")
    print(f"Ollama URL:       {args.ollama_url}")
    print(f"Тестовых случаев: {len(test_cases)}")
    print(f"Критерий:         {args.criterion}")
    print(f"Макс. итераций:   {args.max_iterations}")
    print(f"Гипотез/итерацию: {args.hypotheses_count}")
    print(f"Терпение:         {args.patience}")
    print(f"\nИсходный промпт:\n{'-' * 40}\n{prompt}\n{'-' * 40}")

    print("\nПроверка подключения к Ollama...")
    try:
        ollama._call_ollama("ping", "")
        print("Соединение с Ollama установлено.")
    except Exception as e:
        print(f"\nОшибка подключения к Ollama: {e}")
        print("\nУбедитесь, что:")
        print("  1. Ollama запущена: ollama serve")
        print(f"  2. Модель {args.model} установлена: ollama pull {args.model}")
        print(f"  3. URL указан верно: {args.ollama_url}")
        return 1

    # ================================================================
    # 5. Запуск оптимизации
    # ================================================================
    optimizer = Optimizer(ollama, session_name)

    start_time = time.time()
    try:
        best_prompt, best_score = optimizer.optimize(
            initial_prompt=prompt,
            test_cases=test_cases,
            criterion=args.criterion,
            max_iterations=args.max_iterations,
            hypotheses_count=args.hypotheses_count,
            patience=args.patience
        )
    except KeyboardInterrupt:
        print("\n\nПрервано пользователем.")
        return 130
    except Exception as e:
        print(f"\nКритическая ошибка: {e}")
        return 1

    elapsed = time.time() - start_time

    # ================================================================
    # 6. Финальный вывод
    # ================================================================
    print(f"\n{'=' * 60}")
    print(f"ОПТИМИЗАЦИЯ ЗАВЕРШЕНА")
    print(f"{'=' * 60}")
    print(f"Время выполнения:  {elapsed:.1f} сек")
    print(f"Количество итераций: {len(optimizer.history)}")
    print(f"Итоговая оценка:   {best_score:.3f}")
    print(f"\nЛучший промпт:\n{'-' * 40}\n{best_prompt}\n{'-' * 40}")
    print(f"\nРезультаты сохранены в: {optimizer.session_dir}")
    print(f"  - baseline_evaluation.md  — эталонная оценка")
    print(f"  - iter01_hyp1.md ...      — результаты проверки гипотез")
    print(f"  - best_prompt.txt         — лучший промпт")
    print(f"  - final_report.md         — итоговый отчёт")

    return 0


if __name__ == "__main__":
    exit(main())