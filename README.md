# GA_LLM_LangChain

智慧型藥物分子設計代理（Agent），結合 **Graph Genetic Algorithm (Graph GA)**、**Cerebras LLM** 與 **GuacaMol** 基準任務。系統透過 LangChain 標準化工具介面，讓代理在演化過程中可自動切換「圖形式基因演算法」與「語言模型生成」兩種探索手段，同時動態調整 GA 參數並追蹤 Oracle 成本。

## 系統流程

```
┌────────────┐      ┌────────────────────┐
│  main.py   │ ───► │ ZeroShotAgent/工具 │
└────────────┘      └────────┬───────────┘
     ▲  初始化 API Key、任務          │
     │  建立記憶體/提示               │
     │                             │
     │                      ┌──────▼───────┐
     │                      │ agent.py     │
     │                      │ 參數調控工具 │
     │                      └──────┬───────┘
     │                             │ 建構 LL M 提示解析歷史
     │                             │
     │                      ┌──────▼───────┐
     │                      │ tools/       │
     │                      │ generator.py │
     │                      └──────┬───────┘
     │                             │
     │                  ┌──────────▼──────────┐
     │                  │ Graph GA / Cerebras │
     │                  │ LLM 生 成 / Oracle  │
     │                  │ Scoring             │
     │                  └──────────┬──────────┘
     │                             │
     └───────────結果紀錄──────────┘
```

1. `main.py` 載入 `.env` 的 `CEREBRAS_API_KEY`，建立 LangChain Agent (`ZeroShotAgent`)。
2. Agent 可呼叫三個工具：
   - `graph_ga_generate`：以 `GB_GA_Generator` 執行圖形式 GA。
   - `llm_generate`：透過 Cerebras LLM 生成 SMILES，並以 Oracle 評分。
   - `parameter_tuner`：交由 `agent.py` 的 Cerebras LLM 分析歷史紀錄、動態調整 GA 參數。
3. 所有結果透過 `utils/record_generation.py` 與 `utils/file_config.py` 序列化到 `results/`，並由 `utils/logger_config.py` 產生日誌。

## 核心模組

- **`main.py`**：專案入口。負責
  - 載入 `.env`、解析 CLI 參數（`--task`）。
  - 初始化 `CerebrasChatLLM`，並將它注入 Agent、生成器與參數調控模組。
  - 建立工具列表、LangChain 記憶體、客製化 ZeroShot 提示。
  - 觸發代理迭代並存檔。
- **`agent.py`**
  - 管理 Oracle 預算、格式化歷史資料 (`format_history_for_llm`)。
  - `initialize_parameter_llm` 以 Cerebras LLM 建立參數分析模型。
  - `parameter_tuner_tool` 解析歷史或直接接收 LLM 建議，輸出 GA 可用的設定。
  - `GAParameters` 模型定義於 `prompt.py`，確保 LLM 回傳值可解析。
- **`tools/generator.py`**
  - `graph_ga_generate`：整合 `GB_GA_Generator`、Oracle 評分與結果記錄。
  - `llm_generate`：根據歷史資料、當前任務評分標準組出提示，呼叫 Cerebras LLM 產生 SMILES，並透過 `score_smiles` 評分、更新 Oracle 預算與精英池。
  - `initialize_llm_with_key`：建立 `CerebrasChatLLM` 實例供生成器使用。
- **`tools/oracle_scoring.py`**
  - 將 GuacaMol 任務映射至對應基準函式（包含 logP/TPSA 參數化任務與特定 SMILES 的相似度任務）。
  - 提供 `set_task`、`score_smiles` 以供工具與 Agent 使用。
- **`prompt.py`**
  - 各任務的評分敘述、LLM 生成模板與 GA 參數資料模型。
- **`utils/record_generation.py` / `utils/file_config.py`**
  - 統一管理結果輸出（CSV、JSON、精英池紀錄）與執行時間戳記。
- **Shell 腳本 (`*.sh`)**
  - 提供 HPC 提交範例：建立 Conda 環境、載入 `.env`、呼叫 `python main.py --task ...`，並把輸出寫入 `log/<task>.log`。

## 安裝與環境設定

1. **建立 Conda 環境**
   ```bash
   conda create -n agent_predictor python=3.12
   conda activate agent_predictor
   ```
2. **安裝依賴**
   ```bash
   pip install -r requirements.txt
   ```
   > 專案依賴 RDKit、GuacaMol、LangChain、Cerebras SDK 等套件。部分套件需使用 Conda/自帶 whl 以確保相容性。
3. **設定 API Key**
   - 在專案根目錄建立 `.env`：
     ```
     CEREBRAS_API_KEY=your_key_here
     ```
   - api key 申請網站：https://cloud.cerebras.ai/platform/org_jyy6f2eferyhxmxrde6ch28p/playground
   - model suggest : gpt-oss-120b	, qwen-3-235b-a22b-thinking-2507	
   - 此project 沒有api router ，可自建api router
   - 所有 Python 模組與 Shell 腳本會自動載入 `.env`。
4. **資料檔案**
   - Graph GA 預設使用 `/home/luketou/LLM_AI_agent/Agent_predictor/data/guacamol_v1_all.txt` 作為初始 SMILES。若路徑不同，請在呼叫工具時於 `params` 中覆蓋 `smi_file`。

## 執行方式

### 直接執行代理

```bash
python main.py --task celecoxib
```

- `--task` 支援的任務列於 `main.py` 的 `choices`（例如 `celecoxib`, `troglitazone`, `amlodipine`, `isomer_c11h24`, `median1`, `qed`, `logp_2.5` 等）。
- 程式會在 `results/<task>_<timestamp>/` 建立輸出，並印出代理的最終回答。

### 使用 HPC/JOB 腳本

以 `celecoxib.sh` 為例：

```bash
bash celecoxib.sh
```

腳本會：
- 啟動對應 Conda 環境。
- 載入 `.env`，確保 `CEREBRAS_API_KEY` 可用。
- 執行 `python main.py --task celecoxib` 並把輸出寫入 `log/celecoxib.log`。

其他任務（如 `troglitazone.sh`, `amlodipine.sh`, `median1.sh`, `isomers_c11h24.sh`）亦遵循同樣模式。

## 結果與紀錄

- **結果目錄**：`results/<task>_<timestamp>/`
  - `gen_999_results.json`：完整的代理輸出、工具中間步驟。
  - `each_generation_record_<task>_<timestamp>.csv`：各代統計（最大/平均分數、oracle 使用量、GA 參數）。
  - `generation_details_<task>_<timestamp>.json`：每代詳列 SMILES 與分數。
  - 精英池 CSV：保留最高分的前 100 個分子。
- **日誌**：儲存在 `utils/logs/`，分別有 `main_*.log`, `agent_*.log`, `oracle_*.log`, `generator_*.log`, `graph_ga_*.log`。
  - 可在 `utils/logger_config.py` 調整日誌等級或輸出位置。

## 測試

專案提供基礎測試樣板：

```bash
pytest tests/test_generator.py
pytest tests/test_ga.py
```

- `test_generator.py` 透過 mock 檢查 LLM 生成功能與 scoring。
- `test_ga.py` 示範如何在獨立腳本中呼叫 `graph_ga_generate`，並記錄 GA 的演化歷史。

## 客製化建議

- **新增任務**：在 `tools/oracle_scoring.py` 裡加入新的 benchmark 對應，並在 `prompt.py` 的 `TASK_SCORING_CRITERIA` 與 `main.py` 的 CLI `choices` 列出即可。
- **調整提示**：可修改 `prompt.py` 的 `system_message`、`initial_prompt`、`llm_generate_template`，或自訂 `parameter_optimization_template`。
- **更換 LLM**：`utils/cerebras_llm.py` 提供 LangChain 介面，若要接其他模型，可繼承 `BaseChatModel` 或直接替換該模組。
- **資料輸入輸出**：`tools/generator.py` 中的 `create_scoring_wrapper` 與結果記錄函式提供擴充點，可整合自訂評分器或資料庫。

## 疑難排解

- **API Key 相關錯誤**
  - 確認 `.env` 已設置且腳本能讀取；可於終端手動 `echo $CEREBRAS_API_KEY` 驗證。
- **GuacaMol 資料缺失**
  - 若 `guacamol_v1_all.txt` 路徑不符，需手動下載並在執行參數或程式碼中指定。
- **RDKit / CUDA 依賴**
  - HPC 腳本已範例化 CUDA 環境變數。若在本地執行，可刪除或改成對應 GPU/CPU 設定。

---

透過上述流程與模組說明，新進人員只需設定好環境與 API Key，即可重現整個「GA + LLM」藥物生成管線，並針對不同任務快速評估分子設計策略。歡迎依需求擴充任務、替換模型或整合額外分析元件。祝開發順利！
