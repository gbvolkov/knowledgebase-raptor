import re
from typing import List

def filter_entries(input_path: str, output_path: str, authors_to_remove: List[str]):
    # Читаем входной файл
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Готовим регулярку для списка авторов: экранируем имена и объединяем через |
    escaped = [re.escape(a.strip()) for a in authors_to_remove]
    authors_pattern = r'|'.join(escaped)
    # Ищем строки вида "Author:Имя" в начале строки
    author_re = re.compile(
        rf'^Author:\s*(?:{authors_pattern})\b',
        re.IGNORECASE | re.MULTILINE
    )

    # Разделитель блоков — 23 знака "=" на отдельной строке
    delimiter = r'(?:\n?={23}\n?)'
    parts = re.split(f'({delimiter})', content)

    # Фильтруем блоки
    filtered = []
    for i in range(0, len(parts), 2):
        block = parts[i]
        sep = parts[i+1] if i+1 < len(parts) else ''
        # Если блок НЕ относится ни к одному из указанных авторов — оставляем
        if not author_re.search(block):
            filtered.append(block)
            filtered.append(sep)

    result = ''.join(filtered).rstrip('\n')

    # Записываем результат
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result)


if __name__ == '__main__':
    input_file = 'data_full/qa_full.txt'   # исходный файл
    output_file = 'data_full/qa_full_cleaned.txt' # куда записать отфильтрованный текст
    authors = ['Зерокодер 19', 'Университет', 'Университет "Zerocoder"', 'Zerocoder Новости', 'Помощник Зерокота']         # имя автора, чьи записи нужно удалить
    
    result = filter_entries(input_file, output_file, authors)


    print(f'Готово! Все записи от "{authors}" удалены и сохранены в "{output_file}".')
