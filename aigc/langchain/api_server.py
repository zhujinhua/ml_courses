"""
Author: jhzhu
Date: 2024/8/17
Description: 
"""
import os

#!/usr/bin/env python
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langchain_community.chat_models import ChatTongyi
from langserve import add_routes
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
api_key = os.environ['DASHSCOPE_API_KEY']

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)
model = ChatTongyi(model='qwen-long', top_p=0.8, temperature=0, api_key=api_key)

prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
add_routes(
    app,
    prompt | model,
    path="/joke",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
