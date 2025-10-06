from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# 載入.env 檔案中的環境變數
load_dotenv()

# 1. 初始化模型
model = ChatOpenAI(model="gpt-4o-mini")

# 2. 建立提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一個旅遊助理。"),
    ("human", "請推薦 {city} 的三個景點。")
])

# 3. 建立輸出解析器
output_parser = StrOutputParser()

# 4. 使用 LCEL 組合鏈
chain = prompt | model | output_parser

# 5. 執行鏈
response = chain.invoke({"city": "Paris"})
print("--- Invoke Response ---")
print(response)

# 6. 串流執行鏈
print("\n--- Stream Response ---")
for chunk in chain.stream({"city": "Tokyo"}):
    print(chunk, end="", flush=True)
print()