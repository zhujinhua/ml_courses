# 引入 OpenAI 支持库
from langchain_openai import ChatOpenAI
# 连接信息
# 自己的vllm部署的 服务地址
base_url = "http://direct.virtaicloud.com:28408/v1"
# 自己部署的模型暂无
api_key = "xxxx"
# model 名称
model_name = "qwen-vl-chat"


# 获取自定以模型连接对象
def get_self_model_connect(base_url=base_url, api_key=api_key, model_name=model_name, stream_option=None):
    """
    
    """
    # 连接大模型
    llm = ChatOpenAI(base_url=base_url,
                    api_key=api_key,
                    model=model_name,
                    temperature=0.01,
                    max_tokens=512,
                    
                    stream_options={"include_usage": stream_option}
                    )
    

    return llm



