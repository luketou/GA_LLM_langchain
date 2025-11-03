import os
try:
    from langchain_xai import ChatXAI # 假設這是您用來與 Grok 模型互動的類別
except ImportError:
    print("錯誤：找不到 `langchain_xai` 模組。")
    print("請確保它已安裝，或者將 `ChatXAI` 替換為您實際使用的 LLM 類別。")
    exit()

def test_grok_models():
    """
    一個簡單的腳本，用於測試不同的 Grok 模型。
    執行前請確保已設定 XAI_API_KEY 環境變數。
    """
    api_key = os.getenv("CEREBRAS_API_KEY", None)

    if not api_key:
        print("錯誤：未設定 XAI_API_KEY 環境變數。")
        print("請在執行腳本前設定它，例如：export XAI_API_KEY='your_api_key_here'")
        return

    model_names = [
        "grok-3-fast",  # 假設的模型名稱
        "grok-3-mini"   # 假設的模型名稱
    ]

    test_prompt = "請給我講一個適合在工作場合聽的短笑話。"

    for model_name in model_names:
        print(f"\n--- 測試模型：{model_name} ---")
        try:
            # 初始化 LLM
            # 您可以根據需要調整 temperature 和 max_tokens
            llm = ChatXAI(
                model_name=model_name,
                api_key=api_key,
                temperature=0.7,
                max_tokens=200  # 限制回應長度
            )
            print(f"成功初始化模型：{model_name}。")

            # 發送提示
            print(f"正在發送提示：\"{test_prompt}\"")
            response = llm.invoke(test_prompt)

            # 印出回應
            print(f"來自 {model_name} 的回應：")
            if hasattr(response, 'content'):  # 適用於 LangChain AIMessage 類型的物件
                print(response.content)
            elif isinstance(response, str): # 適用於純字串回應
                print(response)
            else: # 其他可能的回應類型
                print(str(response))

        except ValueError as ve:
             print(f"初始化 {model_name} 時發生 ValueError：{ve}")
        except Exception as e:
            print(f"測試模型 {model_name} 時發生錯誤：{e}")
            print(f"這可能是由於無效的模型名稱、API 問題或其他配置問題。")
        finally:
            print("--- 模型測試結束 ---")

if __name__ == "__main__":
    test_grok_models()