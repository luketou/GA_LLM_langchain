import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# 將專案根目錄加到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 匯入需要測試的模組
from tools.generator import llm_generate, initialize_llm_with_key
from tools.oracle_scoring import get_benchmark, score_smiles, set_task


class TestLLMGenerate(unittest.TestCase):
    
    def setUp(self):
        """測試前的準備工作"""
        # 設定任務為 celecoxib
        set_task("median1")
        
        # 建立模擬的 LLM 回應
        self.mock_response = MagicMock()
        self.mock_response.content = """
        Here are the molecules:
        <SMILES>CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F</SMILES>
        <SMILES>CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)Cl</SMILES>
        <SMILES>CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)Br</SMILES>
        """
        
        # 初始化全域變數
        self.original_llm = None
    
    def tearDown(self):
        """測試後的清理工作"""
        from tools.generator import generation_llm
        if generation_llm != self.original_llm:
            from tools.generator import generation_llm
            generation_llm = self.original_llm
    
    @patch('tools.generator.generation_llm')
    @patch('tools.generator.get_remaining_oracle_budget')
    @patch('tools.generator.update_remaining_oracle_budget')
    def test_llm_generate_basic(self, mock_update_budget, mock_get_budget, mock_llm):
        """測試基本的 LLM 生成功能"""
        # 保存原來的 LLM
        from tools.generator import generation_llm
        self.original_llm = generation_llm
        
        # 設定 mock
        mock_get_budget.return_value = 1000
        mock_llm.invoke.return_value = self.mock_response
        
        # 執行 llm_generate 函數
        result = llm_generate({"count": 3})
        
        # 檢查呼叫次數
        mock_llm.invoke.assert_called_once()
        mock_get_budget.assert_called_once()
        mock_update_budget.assert_called_once()
        
        # 檢查結果
        self.assertEqual(len(result), 3, "應該生成 3 個分子")
        self.assertTrue(all('smiles' in x and 'score' in x for x in result), "每個結果都應該有 smiles 和 score")
        
        # 檢查分子是否包含原始 SMILES
        found_original = False
        for item in result:
            if item['smiles'] == "CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F":
                found_original = True
                break
        self.assertTrue(found_original, "結果應該包含原始的 celecoxib SMILES")
        
    @patch('tools.generator.generation_llm')
    def test_llm_generate_with_history(self, mock_llm):
        """測試有歷史記錄的 LLM 生成功能"""
        # 設定 mock
        mock_llm.invoke.return_value = self.mock_response
        
        # 創建測試歷史
        history = [
            {'smiles': 'CCCCC', 'score': 0.5},
            {'smiles': 'CCCCO', 'score': 0.7},
            {'smiles': 'CCCCS', 'score': 0.3},
            {'smiles': 'CCCCP', 'score': 0.2},
            {'smiles': 'CCCCN', 'score': 0.8},
            {'smiles': 'CCCCF', 'score': 0.1}
        ]
        
        # 執行 llm_generate 函數
        result = llm_generate({"count": 3, "history": history})
        
        # 檢查結果
        self.assertEqual(len(result), 3, "應該生成 3 個分子")
        
    def test_score_smiles_direct(self):
        """直接測試評分函數"""
        # 獲取 benchmark
        benchmark = get_benchmark("celecoxib")
        
        # 測試用的分子
        test_smiles = [
            "CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F",  # celecoxib
            "CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)Cl",  # 變體
            "c1ccccc1"  # 乙醇 (非相似物)
        ]
        
        # 使用 score_smiles 函數評分
        scores = score_smiles(test_smiles, benchmark)
        
        # 檢查分數
        self.assertEqual(len(scores), len(test_smiles), "分數列表長度應與 SMILES 列表相同")
        self.assertGreater(scores[0], 0.8, "Celecoxib 自身應該有很高的分數")
        self.assertGreater(scores[1], 0.5, "類似物結構應該有中等分數")
        self.assertLess(scores[2], 0.5, "不相關的結構應該有低分數")
    
    @patch('tools.generator.get_benchmark')
    def test_benchmark_objective_handling(self, mock_get_benchmark):
        """測試處理帶有 objective 的 benchmark"""
        # 創建模擬的 benchmark 和 objective
        mock_objective = MagicMock()
        mock_objective.__call__ = MagicMock(return_value=0.9)  # 模擬評分函數
        
        mock_benchmark = MagicMock()
        mock_benchmark.objective = mock_objective
        mock_get_benchmark.return_value = mock_benchmark
        
        # 執行 llm_generate
        with patch('tools.generator.generation_llm') as mock_llm:
            mock_llm.invoke.return_value = self.mock_response
            result = llm_generate({"count": 1})
        
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]['score'], 0.9)
        mock_objective.__call__.assert_called_once()


if __name__ == '__main__':
    # 初始化測試用的 LLM
    mock_llm = MagicMock()
    mock_llm.invoke = MagicMock()
    from tools.generator import generation_llm
    generation_llm = mock_llm
    
    unittest.main()