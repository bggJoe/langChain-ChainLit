import os
from dotenv import load_dotenv
from fastapi import FastAPI

# 載入.env 檔案中的環境變數
load_dotenv()
hello_key = os.getenv("HELLO_KEY")

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": f"{hello_key}, FastAPI!"}
