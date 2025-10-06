"""
主應用程式檔案：使用 Chainlit、LangChain Agents (MCP) 和 RAG 建立一個智能問答機器人。

這個應用程式的架構分為三個主要部分：
1. 配置層 (CONFIGURATION): 集中管理所有可設定的參數，如模型名稱和資料夾路徑。
2. 服務層 (SERVICES - CORE LOGIC): 封裝了所有與 LangChain 相關的複雜邏輯，
   包括 RAG 檢索器的建立、Agent 工具的定義，以及 Agent Executor 的組裝和執行。
   ChatbotService 類別是這個應用的核心引擎。
3. 表現層 (CHAINLIT UI - PRESENTATION LAYER): 處理前端 UI 的互動邏輯，
   包括啟動應用、接收使用者訊息，並將其傳遞給服務層處理，最後將結果呈現給使用者。
"""

import chainlit as cl
import logging
import glob
import os
import tempfile
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from langchain_core.agents import AgentAction, AgentFinish

# --- 1. 配置層 (CONFIGURATION) ---

load_dotenv()
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

RAG_DATA_FOLDER = "rag_data/"
LLM_MODEL_NAME = "gpt-4o-mini"


# --- 2. 服務層 (SERVICES - CORE LOGIC) ---

class ChatbotService:
    """
    封裝了聊天機器人的核心商業邏輯。
    """
    def __init__(self, llm: ChatOpenAI, embeddings: OpenAIEmbeddings, base_retriever: Optional[BaseRetriever]):
        """
        初始化 ChatbotService。
        """
        self.llm = llm
        self.embeddings = embeddings
        self.base_retriever = base_retriever
        self.chat_history = []

    async def process_message(self, user_message: str, new_files: Optional[List[cl.File]] = None) -> str:
        """
        處理單一的使用者訊息，並返回 AI 的最終答案。
        """
        logging.debug(f"--- Agent 執行開始：動態決策 ---")
        logging.debug(f"[輸入] 使用者問題: {user_message}")

        # 建立一個消息物件，但尚未發送
        msg = cl.Message(content="")

        final_answer = ""
        try:
            # 建立工具列表
            tools = []
            if self.base_retriever:
                base_tool = _get_rag_tool(
                    self.base_retriever, 
                    "preloaded_document_retriever", 
                    "當你需要回答關於預載的背景知識（如故事、設定）的問題時，請使用這個工具。"
                )
                tools.append(base_tool)
            
            if new_files:
                logging.info(f"偵測到 {len(new_files)} 個新上傳的檔案，正在進行即時索引...")
                await cl.Message(content=f"正在為您上傳的 {len(new_files)} 個檔案建立索引...", author="處理中").send()
                temp_retriever = await _create_temp_retriever_from_files(new_files, self.embeddings)
                temp_tool = _get_rag_tool(
                    temp_retriever, 
                    "uploaded_file_retriever", 
                    "當你需要回答關於使用者剛剛上傳的檔案內容的問題時，請優先使用這個工具。"
                )
                tools.append(temp_tool)

            agent_executor, _ = _create_agent_executor(self.llm, tools)

            # 首先發送消息
            await msg.send()

            # 使用 astream_events 來獲取 Agent 執行的所有內部事件
            streamed_response = ""
            final_answer = ""

            # 修正 LangChain 最新版本 astream_events 的事件結構
            async for event in agent_executor.astream_events(
                {"input": user_message, "chat_history": self.chat_history},
                version="v1"
            ):
                # 首先檢查事件結構，並獲取正確的事件類型
                if "event" in event:  # 新版本格式
                    kind = event["event"]
                    data = event.get("data", {})
                elif "event_type" in event:  # 舊版本格式
                    kind = event["event_type"]
                    data = event.get("data", {})
                else:
                    logging.warning(f"未知的事件結構: {event}")
                    continue

                if kind == "on_agent_action" or kind == "agent_action":
                    action = data.get("action")
                    if isinstance(action, AgentAction):
                        # 在 Log 中顯示 Agent 選擇了哪個工具
                        logging.info(f"[決策] Agent 決定使用工具: `{action.tool}`")
                        logging.debug(f"[決策] 傳遞給工具的輸入: {action.tool_input}")
                        # 向用戶顯示工具選擇，使用 Markdown 實現視覺縮進
                        await cl.Message(
                            content=f"&nbsp;&nbsp;➤ 正在使用工具: `{action.tool}` 來查找資料。",
                            author="系統"
                        ).send()

                elif kind == "on_tool_start" or kind == "tool_start":
                    tool_name = event.get("name", data.get("name", "未知工具"))
                    await cl.Message(
                        content=f"&nbsp;&nbsp;⟳ 開始使用工具: `{tool_name}` 來查找資料。",
                        author="系統"
                    ).send()

                elif kind == "on_tool_end" or kind == "tool_end":
                    tool_name = event.get("name", data.get("name", "未知工具"))
                    await cl.Message(
                        content=f"&nbsp;&nbsp;✓ 工具 `{tool_name}` 執行完畢。",
                        author="系統"
                    ).send()

                elif kind == "on_chat_model_stream" or kind == "llm_chunk":
                    chunk = data.get("chunk", {})
                    content = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if content:
                        # 記錄流式內容，但不實時顯示
                        streamed_response += content
                        logging.debug(f"流式回應片段: {content}")

                elif kind == "on_agent_finish" or kind == "agent_finish":
                    agent_finish_event = data.get("output")
                    if isinstance(agent_finish_event, AgentFinish):
                        answer = agent_finish_event.return_values.get("output")
                        if answer:
                            final_answer = answer
                            logging.info(f"[最終答案] Agent 完成執行，總結的答案: {final_answer}")

            # 決定最終要顯示的答案：優先使用 on_agent_finish 的輸出
            full_response = final_answer if final_answer else streamed_response

            # 直接發送最終回應，而非更新原消息
            if full_response:
                await cl.Message(content=full_response).send()
                # 刪除之前的空消息
                await msg.remove()
            else:
                await msg.update(content="抱歉，我無法生成回答。")

        except Exception as e:
            logging.error(f"Agent 執行時發生錯誤: {e}", exc_info=True)
            await cl.Message(content=f"處理您的問題時發生錯誤: {e}").send()
            # 刪除之前的空消息
            try:
                await msg.remove()
            except:
                pass
            full_response = ""

        # 更新對話歷史
        if full_response:
            self.chat_history.append(("human", user_message))
            self.chat_history.append(("ai", full_response))

        # 此處返回的 full_response 主要是為了保持函式簽名的一致性，UI 的更新已在上面完成
        return full_response


async def create_chatbot_service() -> ChatbotService:
    """
    一個工廠函式，負責建立和組裝 ChatbotService 所需的所有依賴項。
    """
    llm = ChatOpenAI(model=LLM_MODEL_NAME, streaming=True)
    embeddings = OpenAIEmbeddings()
    
    base_retriever = None
    try:
        base_retriever = await _create_rag_retriever(RAG_DATA_FOLDER, embeddings)
    except FileNotFoundError as e:
        logging.warning(f"{e} - 將不會載入預設知識庫。")
    
    return ChatbotService(llm, embeddings, base_retriever)


# --- 輔助函式 (Helper Functions) ---

async def _create_rag_retriever(data_folder: str, embeddings: OpenAIEmbeddings) -> BaseRetriever:
    """
    (內部函式) 從資料夾建立 RAG 檢索器。
    """
    file_paths = glob.glob(f"{data_folder}*.txt")
    if not file_paths: raise FileNotFoundError(f"在資料夾 '{data_folder}' 中找不到任何 .txt 檔案。")
    
    logging.info(f"正在從 {data_folder} 載入 {len(file_paths)} 個檔案... ")
    all_docs = []
    for path in file_paths:
        try:
            loader = TextLoader(path, encoding='utf-8')
            all_docs.extend(loader.load())
        except Exception as e:
            logging.warning(f"讀取檔案 {path} 時發生錯誤: {e}")

    if not all_docs: raise ValueError(f"無法從 {data_folder} 中的檔案載入任何內容。")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(all_docs)
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore.as_retriever()

async def _create_temp_retriever_from_files(files: List[cl.File], embeddings: OpenAIEmbeddings) -> BaseRetriever:
    """
    (內部函式) 從使用者上傳的 Chainlit 檔案物件中，建立一個臨時的 RAG 檢索器。
    """
    documents = []
    temp_files_to_delete = []
    try:
        for file in files:
            actual_file_path = None
            if file.path:
                actual_file_path = file.path
            elif file.content is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as temp_file:
                    temp_file.write(file.content)
                    actual_file_path = temp_file.name
                    temp_files_to_delete.append(actual_file_path)
            
            if not actual_file_path: 
                logging.warning(f"檔案 {file.name} 既無路徑也無內容，已略過。")
                continue

            if file.name.endswith(".pdf"): loader = PyPDFLoader(actual_file_path)
            elif file.name.endswith(".csv"): loader = CSVLoader(actual_file_path)
            else: loader = TextLoader(actual_file_path, encoding='utf-8')
            
            try: documents.extend(loader.load())
            except Exception as e: 
                logging.error(f"使用 {loader.__class__.__name__} 處理檔案 {file.name} 時出錯: {e}")
                await cl.Message(content=f"處理檔案 `{file.name}` 時出錯，已略過。原因: {e}").send()

        if not documents: raise ValueError("無法從上傳的檔案中讀取任何可處理的內容。")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        return vectorstore.as_retriever()
    finally:
        for path in temp_files_to_delete: 
            if os.path.exists(path): 
                os.unlink(path)
                logging.debug(f"已刪除臨時檔案: {path}")

def _get_rag_tool(retriever: BaseRetriever, name: str, description: str) -> Tool:
    """
    (內部函式) 將 RAG 檢索器包裝成一個可自訂名稱和描述的 LangChain 工具。
    """
    return Tool(name=name, description=description, func=retriever.invoke)

def _create_agent_executor(llm: ChatOpenAI, tools: List[Tool]) -> Tuple[AgentExecutor, ChatPromptTemplate]:
    """
    (內部函式) 建立並返回一個 LangChain AgentExecutor 以及它所使用的提示模板。
    """

    NEW_SYSTEM_PROMPT = """
   **角色與目標 (Role & Goal):**
您是一位具備世界級水平的知識庫檢索與問答專家，專門為使用者提供精確、可靠且整合了 RAG 檢索結果的答案。
您的核心目標是：

1. 準確判斷使用者問題是否需要外部工具 (RAG) 支援。

2. 在使用工具後，以流暢、自然的台灣正體中文，將檢索到的知識與對話內容整合，提供最終且完整的答案。

3. 嚴格遵守您被賦予的工具描述來決定工具的使用優先級。

**工具使用策略 (Tool Usage Strategy):**
您擁有以下兩種知識檢索工具。請根據問題的內容和時間性判斷：

1. **uploaded_file_retriever (優先級高):** 如果問題明顯是關於**最近上傳**或**特定的臨時文件**內容，請優先使用此工具。

2. **preloaded_document_retriever (背景知識):** 如果問題是關於**預先載入的背景知識、一般設定或故事線**，請使用此工具。

3. **如果您的工具不足以回答問題，請根據您自身的通用知識直接回答。**

**輸出與格式要求 (Output Requirements):**

* **語言：** 必須使用簡潔、專業且流利的台灣正體中文。

* **最終回答：** 您的最終答案 (Final Answer) 應該是使用者可以直接閱讀的，不需要額外的評論或思考標籤。

* **答案完整性：** 答案必須在單一的 `on_agent_finish` 步驟中完成總結。

* **避免冗餘：** 僅在需要檢索資訊時使用工具；對於簡單的通用問題（例如「你好嗎？」），請直接回答。
    """

    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", NEW_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, tools, agent_prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return executor, agent_prompt


# --- 3. 表現層 (CHAINLIT UI - PRESENTATION LAYER) ---

@cl.on_chat_start
async def start_chat():
    """
    當新的聊天會話開始時觸發。
    """
    try:
        chatbot_service = await create_chatbot_service()
        cl.user_session.set("chatbot_service", chatbot_service)
        await cl.Message(content=f"你好！我已經讀取了 '{RAG_DATA_FOLDER}' 資料夾中的知識，現在你可以問我關於這些文件的問題了。也歡迎上傳檔案來提問！").send()

    except Exception as e:
        logging.error(f"初始化 ChatbotService 時發生錯誤: {e}", exc_info=True)
        await cl.Message(f"初始化時發生嚴重錯誤，請檢查 Log: {e}").send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    當使用者發送訊息時觸發。
    """
    chatbot_service = cl.user_session.get("chatbot_service")
    if not chatbot_service:
        await cl.Message(content="系統尚未準備好，請重新整理頁面。").send()
        return

    # 將使用者的問題和可能上傳的檔案，一併傳遞給服務層處理
    # 注意：UI 層現在不直接處理訊息的發送，而是由服務層的 process_message 負責
    await chatbot_service.process_message(message.content, message.elements)
