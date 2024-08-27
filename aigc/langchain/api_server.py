from fastapi import FastAPI
from langserve import add_routes

from utils import get_qwen_models
llm, _, _ = get_qwen_models()

from langchain_core.prompts import PromptTemplate

# 定义一个FastAPI应用
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

# 构建我的应用
prompt = PromptTemplate.from_template(template="请列出{num}本值得一读的{type}书！\n你的返回应当是用逗号分开的一系列的值，比如： `苹果, 桃, 梨` 或者 `苹果,桃,梨`")
chain = prompt | llm

# 配置 API 接口详情
add_routes(
    app=app,
    runnable=chain,
    path="/query",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)