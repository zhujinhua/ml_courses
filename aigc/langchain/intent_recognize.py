"""
Author: jhzhu
Date: 2024/9/9
Description: 
"""
from langchain.schema import OutputParserException
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.prompts import SystemMessagePromptTemplate

from aigc.traning.model_connect_util import get_self_model_connect

chat = get_self_model_connect(base_url='http://direct.virtaicloud.com:20925/v1', stream_option=False)


def intent_recognize():
    messages = [
        SystemMessagePromptTemplate.from_template(template='''
        你是一个{role}，根据用户的输入识别用户的意图。意图分为两类：
        1. 用户需要推荐药品（用户需要了解或查询具体的药品信息），
        2. 用户不需要推荐药品（用户没有询问具体药品相关的信息）。
        输出以 Json 格式输出，输出模版：
        {{
            "intention": 1,
            "reason": "用户询问了治疗咽喉痛的常用感冒药，属于推荐药品范畴，应该推荐药品。"
        }}
    '''
                                                  ),
        HumanMessagePromptTemplate.from_template(
            template='''用户输入:{content}''')
    ]
    prompt = ChatPromptTemplate.from_messages(messages=messages)
    chain = prompt | chat
    result = chain.invoke(input={"role": "意图识别专家", "content": "能推荐常用的感冒药吗？"})
    parser = JsonOutputParser()
    try:
        parsed_output = parser.parse(result.content)
        # print(parsed_output)
        return parsed_output

    except OutputParserException as e:
        print(f"Parse Failed: {e}")

def main():
    intent_recognize()


if __name__ == "__main__":
    main()
