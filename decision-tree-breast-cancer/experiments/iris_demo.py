"""
鸢尾花数据集入门演示实验
========================
对应论文第5部分（入门演示）

使用自实现的 ID3 决策树算法对鸢尾花数据集进行分类，
验证算法的正确性并展示决策树的构建过程。

运行方式：
    python experiments/iris_demo.py

预期输出：
    - 鸢尾花数据集基本信息
    - 决策树结构（字典嵌套格式）
    - 测试集分类准确率
"""

import sys
import os
# 将项目根目录加入 sys.path，确保可以导入 src 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.id3_tree import Id3Tree


def main():
    print('=' * 60)
    print('  鸢尾花数据集 — ID3 决策树演示实验')
    print('  对应论文：入门演示部分')
    print('=' * 60)

    # ===== 1. 加载数据集 =====
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    print(f'\n数据集信息:')
    print(f'  样本数: {X.shape[0]}')
    print(f'  特征数: {X.shape[1]}')
    print(f'  类别:   {target_names}')

    # ===== 2. 将连续特征离散化 =====
    # ID3 算法仅支持离散特征，这里使用均值作为阈值进行二值化
    X_discrete = (X > X.mean(axis=0)).astype(int)

    # ===== 3. 划分训练集/测试集 =====
    X_train, X_test, y_train, y_test = train_test_split(
        X_discrete, y, test_size=0.3, random_state=42, stratify=y
    )

    # ===== 4. 组装 ID3 所需的数据格式 =====
    # 每行：特征值列表 + 类别标签
    train_data = [list(X_train[i]) + [y_train[i]] for i in range(len(X_train))]
    test_data = [list(X_test[i]) + [y_test[i]] for i in range(len(X_test))]

    labels = list(feature_names)

    # ===== 5. 训练 ID3 决策树 =====
    print(f'\n训练 ID3 决策树...')
    tree = Id3Tree()
    tree.fit(train_data, labels)

    print(f'\n生成的决策树结构:')
    tree.print_tree()

    # ===== 6. 分类测试 =====
    y_pred = []
    for sample in test_data:
        pred = tree.predict(sample[:-1])
        y_pred.append(pred)

    acc = accuracy_score(y_test, y_pred)
    print(f'\n测试集分类准确率: {acc:.2%} ({sum(y_pred == y_test)}/{len(y_test)})')

    # ===== 7. 输出每类准确率 =====
    print(f'\n各类别分类准确率:')
    for i, name in enumerate(target_names):
        mask = (y_test == i)
        if mask.sum() > 0:
            class_acc = accuracy_score(y_test[mask], [y_pred[j] for j in range(len(y_pred)) if y_test[j] == i])
            print(f'  {name:12s}: {class_acc:.2%} ({mask.sum()} 个样本)')

    print(f'\n{"=" * 60}')
    print(f'  鸢尾花演示实验完成！')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
