

import os
from fastapi import FastAPI
from pydantic import BaseModel, Json
from typing import Union, Dict, List

import base64 
import json

import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForQuestionAnswering

class Text(BaseModel):
    contents: Union[str, None] = None
    search: str

class Response(BaseModel):
    contents: Union[str, None] = None
    elements: Json

TORCH_MODEL_LOCATION = "PATH TO YOUR MODEL. E.G.- /home/johndoe/roberta-base"

def do_your_job(text: Text):
    question = "Highlight the parts (if any) of this contract related to \"" + text.search +"\" that should be reviewed by a lawyer. Details: The date of the contract"

    inputs = tokenizer.encode_plus(question, text.contents, add_special_tokens=True, max_length=512, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    answer_start_scores, answer_end_scores = model(**inputs)[0], model(**inputs)[1]

    answer_start = torch.argmax(answer_start_scores)  # Get the most likely beginning of answer with the argmax of the score
    answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

    print(f"Question: {question}")
    print(f"Answer: {answer}\n")

    result = Response(contents=text.contents, elements = json.dumps({"elem": ""}))

    print(type(answer))
    if text.contents is not None:
        result.elements = json.dumps({"answer": answer, "start_pos": text.contents.find(answer), "end_pos":  text.contents.find(answer) + len(answer)})
       
    return result



tokenizer = AutoTokenizer.from_pretrained(TORCH_MODEL_LOCATION)
model = AutoModelForQuestionAnswering.from_pretrained(TORCH_MODEL_LOCATION)

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/call-saul")
async def call_saul(text: Text):
    return {"message": do_your_job(text)}

@app.post("/call-saul")
async def call_saul(text: Text):
    return {"message": do_your_job(text)}
