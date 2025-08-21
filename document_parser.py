from typing import List, Any
from enum import Enum, EnumMeta

import config


import os, pprint, json
from pathlib import Path

from bs4 import BeautifulSoup


from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain.docstore.document import Document


from find_match import get_synonim


class StrEnumMeta(EnumMeta):
    def __getattribute__(cls, name):
        obj = super().__getattribute__(name)
        return get_synonim(obj.value) if isinstance(obj, cls) else obj

class TAGS(Enum, metaclass = StrEnumMeta):
    START_OF_CONTENT    = "Общая информация"
    METADATA            = "Сведения о задаче"
    TND                 = "Термины и определения"
    PROBLEMS            = "Перечень решаемых проблем"
    GOALS               = "Бизнес-цели"
    MODELS              = "Текущая и целевая модели"
    ASIS                = "Описание процесса «as is»"
    TOBE                = "Описание процесса «to be»"
    PARTIES             = "Заинтересованные стороны"
    REG_CHANGES         = "Изменения в ЛНА и пользовательских инструкциях"
    BUSINESS_REQS       = "Бизнес-требования"
    FUNC_REQS           = "Функциональные требования"
    USCASES             = "Сценарии использования"
    NONFUNC_REQS        = "Нефункциональные требования"
    RISKS               = "Риски"
    LIMITATIONS         = "Ограничения"
    ATTACHMENTS         = "Приложения"

    def __eq__(self, other):
        if isinstance (other, str):
            return (get_synonim(self.value.lower()) == get_synonim(other.strip().lower())) or (self.value.lower() == other.strip().lower())
        elif isinstance (other, TAGS):
            return (get_synonim(self.value.lower()) == get_synonim(other.value.lower())) or (self.value.lower() == other.value.lower())
        return (get_synonim(self.value.lower()) == get_synonim(str(other).strip().lower())) or (self.value.lower() == str(other).strip().lower())
    
    def __ne__(self, other):
        return not self.__eq__(self, other)


common_tags = [
    TAGS.METADATA, 
    TAGS.TND, 
    TAGS.PROBLEMS, 
    TAGS.GOALS, 
    TAGS.MODELS,
    TAGS.ASIS, 
    TAGS.TOBE, 
    TAGS.PARTIES, 
    TAGS.LIMITATIONS,
    TAGS.RISKS
]

def parse_bfr(bfr_path: str)-> dict:

    loader = UnstructuredWordDocumentLoader(bfr_path, mode="elements")
    docs = loader.load()

    def parse_table(html: str, orientation = 'h')-> dict:
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        if not table:
            return {}

        rows = []
        for info_block in table.find_all('tr'):
            cells = info_block.find_all(['th', 'td'])
            if cells:
                data = [cell.getText(strip=True) for cell in cells]
                rows.append(data)
        if orientation == 'v':
            rows = list(map(list, zip(*rows)))
        rows[0] = [get_synonim(col) for col in rows[0]]
        return [dict(zip(rows[0], row)) for row in rows[1:]] if rows and len(rows) > 1 else []

    topic = ""
    title_rows = []
    b_title = True
    topics = {}

    topics["ID"] = os.path.basename(bfr_path)
    for doc in docs:
        content = doc.page_content.strip()
        if len(content) == 0:
            continue
        if b_title and doc.metadata["page_number"] > 1: #content == TAGS.START_OF_CONTENT:
            b_title = False
            topics["Title"] = "\n".join(title_rows)
        if b_title:
            title_rows.append(content)
        if doc.metadata["category"] in ["UncategorizedText", "NarrativeText"]:
            if topic != "":
                topics[topic] = topics.get(topic, []) + [content]
            #print(f"TEXT: {content}")
        elif doc.metadata["category"] == "Title":
            #print(f"TITLE: {content}")
            if any (member == content for member in TAGS):
                topic = get_synonim(content)
                topics[topic] = topics.get(topic, [])
            elif topic != "":
                topics[topic] = topics.get(topic, []) + [content]
        elif doc.metadata["category"] == "ListItem":
            #print(f"LIST_ITEM: {content}")
            if any (member == content for member in TAGS):
                topic = get_synonim(content)
                topics[topic] = topics.get(topic, [])
            elif topic != "":
                topics[topic] = topics.get(topic, []) + [content]
        elif doc.metadata["category"] == "Table":
            #print(f"TABLE: {topic}")
            orientation = "h"
            if topic == TAGS.METADATA:
                orientation = "v"
            table_content = parse_table(doc.metadata["text_as_html"], orientation=orientation)
            if topic != "":
                topics[topic] = table_content
            #pprint.pprint(table_content, indent=4)
        #else:
            #print(f"OTHER: {doc.metadata["category"]}: {content}")
    return topics

def get_enumerated_content(doc: dict, tag: TAGS)-> list:
    rows = []
    sub_doc = doc.get(tag, [])
    for idx, item in enumerate(sub_doc):
        row = f"{idx+1}. {item}"
        rows.append(row)
    return rows

def get_tnd_content(doc: dict)-> str:
    rows = []
    terms = doc.get(TAGS.TND, [])
    for idx, term in enumerate(terms):
        row = f"{idx+1}. {term["abbreviation"]}: {term["term"]}; {term["description"]}"
        rows.append(row)
    return "##Термины и определения\n" + "\n".join(rows)

def get_problems_content(doc: dict)-> str:
    rows = get_enumerated_content(doc, TAGS.PROBLEMS)
    return "##Перечень решаемых проблем\n" + "\n".join(rows)

def get_business_goals_content(doc: dict)-> str:
    rows = get_enumerated_content(doc, TAGS.GOALS)
    return "##Бизнес-цели\n" + "\n".join(rows)

def get_models_content(doc: dict) -> str:
    models_rows = get_enumerated_content(doc, TAGS.MODELS)
    return (f"##Текущая и целевая модели\n{"\n".join(models_rows)}")

def get_parties_content(doc: dict)-> str:
    rows = []
    parties = doc.get(TAGS.PARTIES, [])
    for idx, term in enumerate(parties):
        row = f"{idx+1}. {term["interested parties"]} отвечает за: {term["responsibility area"]}"
        rows.append(row)

    return "##Заинтересованные стороны\n" + "\n".join(rows)

def get_changes_content(doc: dict)-> str:
    rows = get_enumerated_content(doc, TAGS.REG_CHANGES)
    return "##Изменения в ЛНА и пользовательских инструкциях\n" + "\n".join(rows)

def get_non_func_reqs_content(doc: dict)-> str:
    rows = get_enumerated_content(doc, TAGS.NONFUNC_REQS)
    return "##Нефункциональные требования\n" + "\n".join(rows)

def get_risks_content(doc: dict)-> str:
    rows = []
    risks = doc.get(TAGS.RISKS, [])
    for idx, term in enumerate(risks):
        row = f"{idx+1}. {term["Риски"]}\nВероятность наступления: {term["Вероятность наступления риска (высок., ср., низк.)"]}\nСпособ митигации: {term["Способы митигации риска"]}"
        rows.append(row)

    return "##Риски\n" + "\n\n".join(rows)

def get_limitations_content(doc: dict)-> str:
    rows = get_enumerated_content(doc, TAGS.LIMITATIONS)
    return "##Ограничения\n" + "\n".join(rows)

def build_br_documents(document: dict, metadata: dict)-> List[Document]:
    br_docs = document.get(TAGS.BUSINESS_REQS, [])
    metadata["type"] = "br" #business requirement
    documents: List[Document] = []
    for doc in br_docs:
        if isinstance(doc, dict):
            doc_id = doc["ID"]
            requirement = doc["description"].strip()
        if len(requirement) > 0:
            content = f"Требование {doc_id}: {requirement}"
            documents.append(Document(content, metadata=metadata))

    return documents

def build_fr_documents(document: dict, metadata: dict)-> List[Document]:
    br_docs = document.get(TAGS.FUNC_REQS, [])
    metadata["type"] = "fr" #functional requiremnet
    documents: List[Document] = []
    idx = 1
    for doc in br_docs:
        if isinstance(doc, dict):
            doc_id = doc["ID"]
            requirement = doc["description"].strip()
            #br_no = doc["ID BR"].strip()
            if len(requirement) > 0:
                content = f"Требование {doc_id}: {requirement}." # Относится к бизнес-требованию: {br_no}"
                documents.append(Document(content, metadata=metadata))

    return documents


def build_uc_documents(document: dict, metadata: dict)-> List[Document]:
    br_docs = document.get(TAGS.USCASES, [])
    metadata["type"] = "uc" # use case
    documents: List[Document] = []
    for doc in br_docs:
        doc_id = doc["№"]
        user_act = doc["user action"].strip()
        system_act = doc["system action"].strip()
        if len(user_act) > 0:
            content = f"#{doc_id}. Действие пользователя: {user_act}.\nРеакция системы: {system_act}"
            documents.append(Document(content, metadata=metadata))
    return documents

def build_risks_documents(document: dict, metadata: dict) -> List[Document]:
    br_docs = document.get(TAGS.RISKS, [])
    metadata["type"] = "rm" #risc matrix
    documents: List[Document] = []
    for doc in br_docs:
        risk = doc["risks"].strip()
        if len(risk) > 0:
            probability = doc["risk probability"]
            mitigation = doc["risk mitigation"]
            content = f"Описание риска: {risk}.\nВероятность наступления: {probability}\nСпособы митигации: {mitigation}"
            documents.append(Document(content, metadata=metadata))
    return documents

def build_asid_document(doc: dict, metadata: dict) -> str:
    asis_rows = get_enumerated_content(doc, TAGS.ASIS)
    content = f"##Описание процесса «as is»\n{"\n".join(asis_rows)}"
    return [Document(content, metadata=metadata)]

def build_tobe_document(doc: dict, metadata: dict) -> str:
    tobe_rows = get_enumerated_content(doc, TAGS.TOBE)
    content = f"##Описание процесса to be»\n{"\n".join(tobe_rows)}"
    return [Document(content, metadata=metadata)]


def build_documents(document: dict) -> Document:

    title = document["Title"]
    meta = document[TAGS.METADATA][0]

    metadata = {
        "type": "head",
        "id": document["ID"],
        "title": title,
        "project": meta.get("project", ""),
    }
    metadata["systems"] = meta.get("it system", "")
    metadata["task_no"] = meta.get("task number", "")
    metadata["task_date"] = meta.get("task date", "")
    metadata["initiator"] = meta.get("initiator", "")
    metadata["department"] = meta.get("department of initiator", "")
    metadata["curator"] = meta.get("curator", "")
    metadata["executor"] = meta.get("executor", "")

    tnd_content = get_tnd_content(document)
    problems_content = get_problems_content(document)
    business_goals = get_business_goals_content(document)
    models_content = get_models_content(document)
    parties_content = get_parties_content(document)
    changes_content = get_changes_content(document)
    non_funq_reqs_content = get_non_func_reqs_content(document)
    limitations_content = get_limitations_content(document)

    content = f"{title}\n\n{tnd_content}\n{problems_content}\n{business_goals}\n{models_content}\n{parties_content}\n{changes_content}\n{non_funq_reqs_content}\n{limitations_content}"

    head_document = Document(content, metadata=metadata)
    risk_documents = build_risks_documents(document, metadata)
    br_documents = build_br_documents(document, metadata)
    fr_documents = build_fr_documents(document, metadata)
    uc_documents = build_uc_documents(document, metadata)
    asis_documents = build_asid_document(document, metadata)
    tobe_documents = build_tobe_document(document, metadata)

    return [head_document] + risk_documents + br_documents + fr_documents + uc_documents + asis_documents + tobe_documents


VALID_EXT = [".doc", ".docx"]
def load_documents(doc_path: Any) -> List[Document]:

    if not isinstance(doc_path, Path):
        doc_path = Path(doc_path)

    documents: List[Document] = []
    
    #doc_path = Path(path)
    for file_path in doc_path.iterdir():
        #print(f"===========>{file_path}")
        if not file_path.is_file():
            continue
        if file_path.name.startswith("~"):
            continue
        if file_path.suffix in VALID_EXT:
            topics = parse_bfr(file_path)
            documents.extend(build_documents(topics))
        #print(f"{file_path}<===========")
    return documents


if __name__ == "__main__":
    #print(get_synonim("Описание процесса «as is»"))
    #print(get_synonim("Описание процесса «to be»"))
    #print(get_synonim("Термин/ определение"))

    documents = load_documents(Path("./data/bfts"))
    pprint.pprint(documents, indent=2)