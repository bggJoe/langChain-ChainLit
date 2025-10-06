# 智能問答機器人 (RAG + MCP Agent)

這是一個基於 Chainlit、LangChain Agents (MCP) 和 RAG 技術構建的進階智能問答機器人。它不僅能理解和回答關於預先提供的文件內容的問題，還支援使用者**即時上傳檔案**進行問答，並能自主判斷何時需要查閱文件，何時使用通用知識進行回答。

## 功能特色

- **網頁使用者介面 (Web UI)**: 使用 [Chainlit](https://docs.chainlit.io/get-started/overview) 快速構建美觀、可互動的聊天介面。
- **雙重知識庫 (Dual RAG)**:
    - **預載知識庫**: 應用程式在啟動時，會自動讀取 `rag_data/` 資料夾下的所有 `.txt` 文件，建立基礎知識庫。
    - **即時檔案問答**: 支援使用者在對話中上傳 `.txt`, `.pdf`, `.csv` 格式的檔案，並針對這些檔案的內容進行即時問答。
- **智能決策核心 (MCP Agent)**: 採用 LangChain 的 Agent Executor 作為決策大腦。Agent 擁有兩種檢索工具，能夠根據使用者問題的語意，自主決定是查詢預載知識、查詢上傳的檔案，還是直接回答。
- **可配置的系統提示**: Agent 的核心行為準則（系統提示）已從程式碼中分離至 `prompts/chatbot_system_prompt.txt`，讓您可以輕鬆地調整 Agent 的角色、目標和輸出風格，而無需修改程式碼。
- **多輪對話記憶**: 能夠記住當前對話的上下文，進行流暢的多輪問答。
- **詳細的執行日誌**: 在後台 Log 中，可以清晰地看到 Agent 的每一步思考、決策和工具呼叫過程，極具教學和除錯價值。

## 技術架構

本專案採用了關注點分離 (Separation of Concerns) 的設計原則，將程式碼劃分為三個邏輯層次：

1.  **表現層 (Presentation Layer - Chainlit UI)**
    -   **職責**: 處理所有與前端 UI 相關的互動邏輯。
    -   **實現**: 使用 `@cl.on_chat_start` 和 `@cl.on_message` 裝飾器來捕捉使用者事件。這一層的程式碼非常簡潔，只負責將請求傳遞給服務層。

2.  **服務層 (Service Layer)**
    -   **職責**: 作為 UI 層和核心 AI 邏輯之間的中介，封裝了應用的主要商業邏輯。
    -   **實現**: `ChatbotService` 類別。它管理著 Agent Executor 和對話歷史，並提供一個 `process_message` 方法供 UI 層呼叫，該方法能同時處理文字和上傳的檔案。

3.  **核心邏輯層 (Core Logic Layer - RAG & MCP)**
    -   **職責**: 實現所有與 LangChain 相關的複雜功能。
    -   **RAG (檢索增強生成)**:
        - `_create_rag_retriever`: 從 `rag_data/` 資料夾建立**預載知識**的檢索器。
        - `_create_temp_retriever_from_files`: 從使用者上傳的檔案動態建立**臨時知識**的檢索器。
    -   **MCP (代理人決策)**: 由 `_create_agent_executor` 函式實現。其核心是「設計配方」：
        1.  **大腦**: `ChatOpenAI` 語言模型。
        2.  **工具箱**: 將上述兩種 RAG 檢索器分別包裝成 `preloaded_document_retriever` 和 `uploaded_file_retriever` 兩個 `Tool`。
        3.  **工作說明書**: Agent 的系統提示（System Prompt）從 `prompts/chatbot_system_prompt.txt` 檔案動態載入，用以明確指示 Agent 的行為準則和決策邏輯。

## 環境準備

在開始之前，請確保您已安裝以下軟體：

- Python 3.10 或更高版本
- Git

## 安裝與設定

1.  **克隆專案**
    ```bash
    git clone <your-repository-url>
    cd langchain_ChainLit
    ```

2.  **建立並啟用虛擬環境**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS / Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **安裝依賴套件**
    ```bash
    pip install -r requirements.txt
    ```

4.  **設定環境變數**
    -   建立一個名為 `.env` 的新檔案。
    -   在 `.env` 檔案中，填入您的 OpenAI API 金鑰。

5.  **準備知識庫與提示 (可選)**
    -   在 `rag_data` 資料夾中，放入您希望 AI 預先學習的 `.txt` 文件。
    -   (可選) 編輯 `prompts/chatbot_system_prompt.txt` 來客製化您 Agent 的行為。

## 如何啟動

本專案提供了多種啟動方式，請根據您的作業系統和偏好選擇：

### 方式一：使用 Makefile (推薦)

```bash
make start
```

### 方式二：使用 Shell 腳本 (macOS / Linux / Git Bash)

```bash
./start.sh
```

### 方式三：使用 Batch 腳本 (Windows CMD / PowerShell)

```bash
start.bat
```

應用程式啟動後，請在您的瀏覽器中開啟 `http://localhost:8000`。

## 專案結構

```
.langchain_ChainLit/
├── .git/               # Git 版本控制資料夾
├── .gitignore          # Git 忽略清單
├── Makefile            # make 指令設定檔
├── README.md           # 本說明文件
├── langchain_chainlit.py # 主應用程式碼
├── project_prompt.txt  # 專案初始生成提示
├── prompts/
│   └── chatbot_system_prompt.txt # Agent 的系統提示
├── rag_data/           # RAG 知識庫來源資料夾
│   └── ...
├── requirements.txt    # Python 依賴套件清單
├── start.bat           # Windows 啟動腳本
├── start.sh            # Linux/macOS 啟動腳本
└── venv/               # Python 虛擬環境
```