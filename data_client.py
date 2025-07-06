from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.prompts import load_mcp_prompt
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
import os

load_dotenv()
os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(model="gpt-4o")

server_params = StdioServerParameters(
    command="python",
    args=["./data_server.py"],
)

#session이 있어야
# 툴을 로드할 수 있고 (load_mcp_tools)
# 프롬프트를 서버로부터 받아올 수 있고 (load_mcp_prompt)
# MCP 프로토콜에 맞는 통신이 가능

# stdio_client는 단순 연결만 담당
# ClientSession이 MCP 서버와의 세션(상태) 관리와 통신 로직을 담당
async def run():
    async with stdio_client(server_params) as (read, write):       
        async with ClientSession(read, write) as session:
            await session.initialize()

            ##### AGENT #####
            tools = await load_mcp_tools(session)
            agent = create_react_agent(model, tools)

            while True: 
                try:
                    ##### REQUEST & REPOND #####
                    user_input = input("질문을 입력하세요: ")
                    if user_input.lower() in ["quit","exit","q"]:
                        print("종료합니다.")
                        break
                    print("=====PROMPT=====")
                    prompts = await load_mcp_prompt(
                        session, "default_prompt", arguments={"message": user_input}
                    )
                    print("prompts : ", prompts)
                    response = await agent.ainvoke({"messages": prompts})
                    # response = await agent.ainvoke({"messages": user_input})

                    print(response)
                    print("=====RESPONSE=====")
                    print(response["messages"][-1].content)
                except:
                    print("종료합니다.")
                    break;


import asyncio

asyncio.run(run())