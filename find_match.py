import difflib
import config
from thefuzz import process
from rapidfuzz import process as process_rapid, fuzz

#from langchain_openai import ChatOpenAI
#from langchain.prompts import ChatPromptTemplate
#from langchain_core.output_parsers import StrOutputParser
#from llm import llm
#model = ChatOpenAI(temperature=0, model="gpt-4.1-nano")
#model = llm
# Define a simple summarization prompt template.
_TEMPLATE = """Here is a term in Russian:

{context}

Please provide a mostly close translation to English from the list of terms:

#LIST OF TERMS:
{terms}
#END OF LIST OF TERMS:

Return only one term from the list of terms without any other phrases and modifications.
If there are no close translations, return context itself with no translations.
"""
#prompt = ChatPromptTemplate.from_template(_TEMPLATE)
#chain = prompt | model | StrOutputParser()


from functools import lru_cache

valid_terms = [
    "general info"
    "project",
    "it system",
    "task number",
    "task date",
    "initiator",
    "department of initiator",
    "curator",
    "executor",
    "terms and definitions", 
    #"term", 
    "abbreviation", 
    "description",
    "list of problems",
    "business goals",
    "current and target models",
    "'as is' process description",
    "'to be' process description",
    "interested parties",
    "number",
    "responsibility area",
    "regulatory changes",
    "business requirements",
    "description",
    "functional requirements",
    "use cases",
    "user action",
    "system action",
    "non-functional requirements",
    "risks",
    "risk probability",
    "risk mitigation",
    "limitations",
]

valid_terms_ru = [
    "Общая информация"
    "Проект",
    "Информационная система",
    "Номер задачи в Bitrix24",
    "Дата постановки задачи",
    "Инициатор",
    "Подразделение инициатора",
    "Куратор",
    "Исполнитель (БА, СА)",
    "Термины и определения", 
    "Термин/определение", 
    #"term", 
    "Сокращение", 
    "Описание",
    "Перечень решаемых проблем",
    "Бизнес-цели",
    "Текущая и целевая модели",
    "Описание процесса «as is»",
    "Описание процесса «to be»",
    "Заинтересованные стороны",
    "Номер",
    "Процесс/зона ответственности",
    "Изменения в ЛНА и пользовательских инструкциях",
    "Бизнес-требования",
    #"Описание",
    "Функциональные требования",
    "Сценарии использования",
    "Действие пользователя",
    "Действие системы",
    "Нефункциональные требования",
    "Риски",
    "Вероятность наступления риска",
    "Способы митигации риска",
    "Ограничения",
]

ru_to_en = {
    "Общая информация": "general info",
    "Сведения о задаче": "task info",
    "Проект": "project",
    "Информационная система": "it system",
    "Номер задачи в Bitrix24": "task number",
    "Дата постановки задачи": "task date",
    "Инициатор": "initiator",
    "Подразделение инициатора": "department of initiator",
    "Куратор": "curator",
    "Исполнитель (БА, СА)": "executor",
    "Термины и определения": "terms and definitions",
    "Термин/определение": "term",
    "Сокращение": "abbreviation",
    "Описание": "description",
    "Описание (не обязательно)": "description",
    "Перечень решаемых проблем": "list of problems",
    "Решаемые проблемы": "list of problems",
    "Бизнес-цели": "business goals",
    "Текущая и целевая модели": "current and target models",
    "Описание процесса «as is»": "'as is' process description",
    "Описание процесса «to be»": "'to be' process description",
    "Заинтересованные стороны": "interested parties",
    "Заинтересованная сторона": "interested parties",
    "Номер": "number",
    "Процесс/зона ответственности": "responsibility area",
    "Изменения в ЛНА и пользовательских инструкциях": "regulatory changes",
    "Бизнес-требования": "business requirements",
    "Функциональные требования": "functional requirements",
    "Сценарии использования": "use cases",
    "Действие пользователя": "user action",
    "Действие системы": "system action",
    "Нефункциональные требования": "non-functional requirements",
    "Риски": "risks",
    "Вероятность наступления риска": "risk probability",
    "Вероятность наступления риска (высок., ср., низк.)": "risk probability",
    "Способы митигации риска": "risk mitigation",
    "Ограничения": "limitations",
}

def find_best_tag(title: str, tag_list: list[str], cutoff=0.75):
    # Normalize case/punctuation for robust matching
    title_norm = title.strip().lower()
    tags_norm = [t.lower() for t in tag_list]
    if best_matches := difflib.get_close_matches(
        title_norm, tags_norm, n=1, cutoff=cutoff
    ):
        # Return the matching tag in original form (find index)
        best_index = tags_norm.index(best_matches[0])
        return tag_list[best_index]
    return None

def find_best_tag_fuzzy(title: str, tag_list: list[dict], score_cutoff=92):
    # We can optionally normalize case and punctuation similar to difflib example
    # Find best match and score
    ru_tags = list(ru_to_en.keys())
    match, score = process.extractOne(title, ru_tags)
    return ru_to_en.get(match, title) if score >= score_cutoff else title

def find_best_tag_rapid_fuzzy(title: str, tag_list: list[dict], score_cutoff=80):
    # We can optionally normalize case and punctuation similar to difflib example
    # Find best match and score
    ru_tags = list(ru_to_en.keys())
    match = process_rapid.extractOne(title, ru_tags, scorer=fuzz.WRatio, score_cutoff=0)
    return ru_to_en.get(match, title)# if score >= score_cutoff else title

@lru_cache(maxsize=5000, typed=True)
def get_synonim(term: str) -> str:
    term = term.replace("/", " или ")
    return find_best_tag_fuzzy(term, ru_to_en)

if __name__ == "__main__":
    #print(get_synonim("Описание процесса «as is»"))
    #print(get_synonim("процесс to-be"))
    print(get_synonim("Решаемые проблемы"))
    print(get_synonim("Перечень решаемых проблем"))
    print(get_synonim("определение или термин"))
    print(get_synonim("автоматически (при механизме «авторешение», когда сделка рассматривается без участия андеррайтеров и других сотрудников службы риск-менеджмента)"))
