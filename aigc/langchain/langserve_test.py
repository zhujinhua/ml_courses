"""
Author: jhzhu
Date: 2024/8/27
Description: 
"""
"""
Author: jhzhu
Date: 2024/8/27
Description: 
"""


# Import statements

# Functions and Classes

#!/usr/bin/env python
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langserve import add_routes

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

add_routes(
    app,
    ChatOpenAI(api_key='EMPTY',
        base_url='http://direct.virtaicloud.com:28408/v1'),
    path="/openai",
)

model = ChatAnthropic(api_key='EMPTY',
        base_url='http://direct.virtaicloud.com:28408/v1',)
prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
add_routes(
    app,
    prompt | model,
    path="/joke",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="direct.virtaicloud.com", port=28408)

