# 智能問答機器人 (RAG + MCP Agent)

這是一個基於 Chainlit、LangChain Agents (MCP) 和 RAG 技術構建的進階智能問答機器人。它不僅能理解和回答關於預先提供的文件內容的問題，還能自主判斷何時需要查閱文件，何時使用通用知識進行回答。

## 功能特色

- **網頁使用者介面 (Web UI)**: 使用 [Chainlit](https://docs.chainlit.io/get-started/overview) 快速構建美觀、可互動的聊天介面。
- **可擴展的知識庫 (RAG)**: 應用程式在啟動時，會自動讀取 `rag_data/` 資料夾下的所有 `.txt` 文件，並將其內容建立成一個可供檢索的向量資料庫。
- **智能決策核心 (MCP)**: 採用 LangChain 的 Agent Executor 作為決策大腦。Agent 能夠根據使用者問題的語意，自主決定是直接回答，還是呼叫 RAG 工具來查閱知識庫。
- **多輪對話記憶**: 能夠記住當前對話的上下文，進行流暢的多輪問答。
- **詳細的執行日誌**: 在後台 Log 中，可以清晰地看到 Agent 的每一步思考、決策和工具呼叫過程，極具教學和除錯價值。

## 技術架構

本專案採用了關注點分離 (Separation of Concerns) 的設計原則，將程式碼劃分為三個邏輯層次：

1.  **表現層 (Presentation Layer - Chainlit UI)**
    -   **職責**: 處理所有與前端 UI 相關的互動邏輯。
    -   **實現**: 使用 `@cl.on_chat_start` 和 `@cl.on_message` 裝飾器來捕捉使用者事件。這一層的程式碼非常簡潔，只負責將請求傳遞給服務層，並將最終結果呈現出來。

2.  **服務層 (Service Layer)**
    -   **職責**: 作為 UI 層和核心 AI 邏輯之間的中介，封裝了應用的主要商業邏輯。
    -   **實現**: `ChatbotService` 類別。它管理著 Agent Executor 和對話歷史，並提供一個簡單的 `process_message` 方法供 UI 層呼叫。

3.  **核心邏輯層 (Core Logic Layer - RAG & MCP)**
    -   **職責**: 實現所有與 LangChain 相關的複雜功能。
    -   **RAG (檢索增強生成)**: 由 `_create_rag_retriever` 函式實現。它負責：
        1.  載入 `rag_data/` 中的文件。
        2.  將文件切割成小區塊 (Chunks)。
        3.  使用 `OpenAIEmbeddings` 將區塊向量化。
        4.  將向量存入 `FAISS` 向量資料庫中，並建立一個 `retriever`。
    -   **MCP (代理人決策)**: 由 `_create_agent_executor` 和 `_get_rag_tool` 函式實現。其核心是「設計配方」：
        1.  **大腦**: `ChatOpenAI` 語言模型。
        2.  **工具箱**: 將 RAG 的 `retriever` 包裝成一個 `Tool`，並用清晰的 `description` 告知 Agent 何時使用它。
        3.  **工作說明書**: 使用 `ChatPromptTemplate` 明確指示 Agent 的行為準則和決策邏輯。

## 環境準備

在開始之前，請確保您已安裝以下軟體：

- Python 3.10 或更高版本
- Git

## 安裝與設定

1.  **克隆專案**
    ```bash
    git clone <your-repository-url>
    cd langchain_project
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
    本專案使用 `requirements.txt` 來管理依賴。執行以下指令以安裝所有必要的套件：
    ```bash
    pip install -r requirements.txt
    ```

4.  **設定環境變數**
    -   將 `.env.example` (如果有的話) 複製為 `.env`，或者直接建立一個名為 `.env` 的新檔案。
    -   在 `.env` 檔案中，填入您的 OpenAI API 金鑰：
        ```
        OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        ```

5.  **準備知識庫**
    -   在專案根目錄下，建立一個名為 `rag_data` 的資料夾。
    -   將您希望 AI 學習的所有 `.txt` 格式的文件放入此資料夾中。

## 如何啟動

本專案提供了多種啟動方式，請根據您的作業系統和偏好選擇：

### 方式一：使用 Makefile (推薦)

這是最標準且跨平台的方式。只需在終端機中執行：

```bash
make start
```

*(注意：Windows 用戶可能需要透過 Git Bash 或 WSL 來執行 `make` 指令。)*

### 方式二：使用 Shell 腳本 (macOS / Linux / Git Bash)

```bash
# 如果是第一次執行，可能需要先給予執行權限
# chmod +x start.sh

./start.sh
```

### 方式三：使用 Batch 腳本 (Windows CMD / PowerShell)

```bash
# 直接執行
start.bat

# 或者
.\start.bat
```

應用程式啟動後，請在您的瀏覽器中開啟 `http://localhost:8500`。

## 專案結構

```
.langchain_project/
├── .git/               # Git 版本控制資料夾
├── .gitignore          # Git 忽略清單
├── Makefile            # make 指令設定檔
├── README.md           # 本說明文件
├── langchain_chainlit.py # 主應用程式碼
├── rag_data/           # RAG 知識庫來源資料夾
│   ├── story_1.txt
│   └── ...
├── requirements.txt    # Python 依賴套件清單
├── start.bat           # Windows 啟動腳本
├── start.sh            # Linux/macOS 啟動腳本
└── venv/               # Python 虛擬環境 (已被 .gitignore 忽略)
```
