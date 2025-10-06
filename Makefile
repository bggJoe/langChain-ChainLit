# .PHONY 宣告 'start' 是一個虛擬目標，它是一個指令，而不是一個檔案。
.PHONY: start

# 'start' 目標執行啟動 Chainlit 伺服器的指令。
# 重要提示：此行開頭的縮排必須是 Tab 鍵，而不是空格。
start:
	chainlit run langchain_chainlit.py --port 8500
