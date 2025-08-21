import os
#os.environ["TRANSFORMERS_VERBOSITY"] = "debug"

os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

import config

from pprint import pprint

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace



# 2) Model + tokenizer + safe generation config
model_path = "google/gemma-3-270m-it"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="cpu",
)

# Make sure generation stops at EOS and padding is defined
end_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
eos_id = end_id if isinstance(end_id, int) and end_id >= 0 else tokenizer.eos_token_id
if model.generation_config is not None:
    model.generation_config.eos_token_id = eos_id
    model.generation_config.pad_token_id = eos_id

# Two sensible decoding profiles:
GEN_KWARGS_QA_STRICT = dict(
    max_new_tokens=512,
    do_sample=False,          # greedy for factual QA
    #temperature=0.0,
)
GEN_KWARGS_QA_BALANCED = dict(
    max_new_tokens=512,
    do_sample=True,           # a touch of sampling if you want fuller answers
    temperature=0.2,
    top_p=0.9,
    repetition_penalty=1.05,
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    clean_up_tokenization_spaces=True,
    **GEN_KWARGS_QA_STRICT,
)

hf_llm = HuggingFacePipeline(pipeline=pipe, verbose=True)
llm = ChatHuggingFace(llm=hf_llm)