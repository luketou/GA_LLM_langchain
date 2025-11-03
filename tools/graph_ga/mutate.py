import random

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

from . import crossover as co
from .crossover import mol_ok, ring_OK
rdBase.DisableLog('rdApp.error')


def delete_atom():
    choices = ['[*:1]~[D1:2]>>[*:1]', '[*:1]~[D2:2]~[*:3]>>[*:1]-[*:3]',
               '[*:1]~[D3:2](~[*;!H0:3])~[*:4]>>[*:1]-[*:3]-[*:4]',
               '[*:1]~[D4:2](~[*;!H0:3])(~[*;!H0:4])~[*:5]>>[*:1]-[*:3]-[*:4]-[*:5]',
               '[*:1]~[D4:2](~[*;!H0;!H1:3])(~[*:4])~[*:5]>>[*:1]-[*:3](-[*:4])-[*:5]']
    p = [0.25, 0.25, 0.25, 0.1875, 0.0625]

    return np.random.choice(choices, p=p)


def append_atom():
    choices = [['single', ['C', 'N', 'O', 'F', 'S', 'Cl', 'Br'], 7 * [1.0 / 7.0]],
               ['double', ['C', 'N', 'O'], 3 * [1.0 / 3.0]],
               ['triple', ['C', 'N'], 2 * [1.0 / 2.0]]]
    p_BO = [0.60, 0.35, 0.05]

    index = np.random.choice(list(range(3)), p=p_BO)

    BO, atom_list, p = choices[index]
    new_atom = np.random.choice(atom_list, p=p)

    if BO == 'single':
        rxn_smarts = '[*;!H0:1]>>[*:1]X'.replace('X', '-' + new_atom)
    if BO == 'double':
        rxn_smarts = '[*;!H0;!H1:1]>>[*:1]X'.replace('X', '=' + new_atom)
    if BO == 'triple':
        rxn_smarts = '[*;H3:1]>>[*:1]X'.replace('X', '#' + new_atom)

    return rxn_smarts


def insert_atom():
    choices = [['single', ['C', 'N', 'O', 'S'], 4 * [1.0 / 4.0]],
               ['double', ['C', 'N'], 2 * [1.0 / 2.0]],
               ['triple', ['C'], [1.0]]]
    p_BO = [0.60, 0.35, 0.05]

    index = np.random.choice(list(range(3)), p=p_BO)

    BO, atom_list, p = choices[index]
    new_atom = np.random.choice(atom_list, p=p)

    if BO == 'single':
        rxn_smarts = '[*:1]~[*:2]>>[*:1]X[*:2]'.replace('X', new_atom)
    if BO == 'double':
        rxn_smarts = '[*;!H0:1]~[*:2]>>[*:1]=X-[*:2]'.replace('X', new_atom)
    if BO == 'triple':
        rxn_smarts = '[*;!R;!H1;!H0:1]~[*:2]>>[*:1]#X-[*:2]'.replace('X', new_atom)

    return rxn_smarts


def change_bond_order():
    choices = ['[*:1]!-[*:2]>>[*:1]-[*:2]', '[*;!H0:1]-[*;!H0:2]>>[*:1]=[*:2]',
               '[*:1]#[*:2]>>[*:1]=[*:2]', '[*;!R;!H1;!H0:1]~[*:2]>>[*:1]#[*:2]']
    p = [0.45, 0.45, 0.05, 0.05]

    return np.random.choice(choices, p=p)


def delete_cyclic_bond():
    return '[*:1]@[*:2]>>([*:1].[*:2])'


def add_ring():
    choices = ['[*;!r;!H0:1]~[*;!r:2]~[*;!r;!H0:3]>>[*:1]1~[*:2]~[*:3]1',
               '[*;!r;!H0:1]~[*!r:2]~[*!r:3]~[*;!r;!H0:4]>>[*:1]1~[*:2]~[*:3]~[*:4]1',
               '[*;!r;!H0:1]~[*!r:2]~[*:3]~[*:4]~[*;!r;!H0:5]>>[*:1]1~[*:2]~[*:3]~[*:4]~[*:5]1',
               '[*;!r;!H0:1]~[*!r:2]~[*:3]~[*:4]~[*!r:5]~[*;!r;!H0:6]>>[*:1]1~[*:2]~[*:3]~[*:4]~[*:5]~[*:6]1']
    p = [0.05, 0.05, 0.45, 0.45]

    return np.random.choice(choices, p=p)


def change_atom(mol):
    """
    修改分子中的一個原子類型。
    
    Args:
        mol: RDKit分子對象
        
    Returns:
        str: 原子替換的SMARTS反應式
        
    說明:
        - 從常見原子類型列表中隨機選擇一個存在於分子中的原子
        - 然後選擇另一個不同的原子類型替換它
        - 原子類型使用SMARTS表示: #6=C, #7=N, #8=O, #9=F, #16=S, #17=Cl, #35=Br
    """
    # 定義原子類型和相應的概率
    atom_types = ['#6', '#7', '#8', '#9', '#16', '#17', '#35']  # C, N, O, F, S, Cl, Br
    probabilities = [0.15, 0.15, 0.14, 0.14, 0.14, 0.14, 0.14]  # 各原子類型選擇概率
    
    # 優化: 預先篩選出分子中存在的原子類型
    valid_atoms = []
    valid_probs = []
    
    for atom_type, prob in zip(atom_types, probabilities):
        if mol.HasSubstructMatch(Chem.MolFromSmarts('[' + atom_type + ']')):
            valid_atoms.append(atom_type)
            valid_probs.append(prob)
    
    # 如果分子中沒有可替換的原子類型，返回原始分子
    if not valid_atoms:
        # 生成一個無效反應，不會改變分子
        return '[C:1]>>[C:1]'
    
    # 重新歸一化概率
    sum_probs = sum(valid_probs)
    valid_probs = [p/sum_probs for p in valid_probs]
    
    # 選擇源原子類型
    source_atom = np.random.choice(valid_atoms, p=valid_probs)
    
    # 選擇目標原子類型 (確保不同於源原子)
    target_atoms = [a for a in atom_types if a != source_atom]
    target_probs = [probabilities[atom_types.index(a)] for a in target_atoms]
    sum_target_probs = sum(target_probs)
    target_probs = [p/sum_target_probs for p in target_probs]
    
    target_atom = np.random.choice(target_atoms, p=target_probs)
    
    # 構建SMARTS反應式 '[X:1]>>[Y:1]'
    return f'[{source_atom}:1]>>[{target_atom}:1]'


def mutate(mol, mutation_rate):
    if random.random() > mutation_rate:
        return mol

    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except ValueError:
        return mol

    p = [0.15, 0.14, 0.14, 0.14, 0.14, 0.14, 0.15]
    for i in range(10):
        rxn_smarts_list = 7 * ['']
        rxn_smarts_list[0] = insert_atom()
        rxn_smarts_list[1] = change_bond_order()
        rxn_smarts_list[2] = delete_cyclic_bond()
        rxn_smarts_list[3] = add_ring()
        rxn_smarts_list[4] = delete_atom()
        rxn_smarts_list[5] = change_atom(mol)
        rxn_smarts_list[6] = append_atom()
        rxn_smarts = np.random.choice(rxn_smarts_list, p=p)

        # print 'mutation',rxn_smarts

        rxn = AllChem.ReactionFromSmarts(rxn_smarts)

        new_mol_trial = rxn.RunReactants((mol,))

        new_mols = []
        for m in new_mol_trial:
            m = m[0]
            # print Chem.MolToSmiles(mol),mol_ok(mol)
            if mol_ok(m) and ring_OK(m):
                new_mols.append(m)

        if len(new_mols) > 0:
            return random.choice(new_mols)

    return None
