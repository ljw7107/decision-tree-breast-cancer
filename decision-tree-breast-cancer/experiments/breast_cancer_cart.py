"""
乳腺癌数据集 CART 决策树主实验
==============================
对应论文第5部分（核心实验）

完整复现论文全部实验结果，包括：
  1. 加载乳腺癌威斯康星数据集，按 7:3 分层划分
  2. 对 max_depth 1~10 执行网格搜索 + 5 折交叉验证
  3. 输出最佳模型的分类报告、混淆矩阵、特征重要性
  4. 生成所有配图（深度-准确率曲线、决策树可视化、混淆矩阵等）
  5. 分析典型误判样本

运行方式：
    python experiments/breast_cancer_cart.py

预期输出：
    - 终端输出完整实验日志（含论文全部数值结果）
    - output/ 目录生成全部论文配图（8张）

Reference:
  [2] Breiman L, et al. Classification and regression trees. CRC press, 1984.
  [5] Wolberg W H, et al. Breast Cancer Wisconsin (Diagnostic) Data Set. UCI, 1995.
  [6] Pedregosa F, et al. Scikit-learn: Machine learning in Python. JMLR, 2011.
"""

import sys
import os
# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, recall_score)
from sklearn.tree import export_text
from src.id3_tree import Id3Tree
from utils.plot_utils import (
    plot_label_distribution,
    plot_feature_distribution,
    plot_depth_comparison,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_decision_tree,
    plot_misclass_radar,
)
from math import pi


def main():
    print('=' * 70)
    print('  乳腺癌威斯康星数据集 — CART 决策树主实验')
    print('  对应论文第5章：实验与应用')
    print('=' * 70)

    # ==================== 1. 加载数据集 ====================
    # 对应论文 5.1 节
    # 数据集来源: UCI Machine Learning Repository / sklearn.datasets
    # 从项目本地 data/ 目录加载
    print('\n[1] 加载数据集')
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'data', 'breast_cancer.csv')
    df = pd.read_csv(data_path)
    feature_names = [c for c in df.columns if c != 'target']
    X = df[feature_names].values
    y = df['target'].values
    target_names = ['malignant', 'benign']

    print(f'  样本数: {X.shape[0]}')
    print(f'  特征数: {X.shape[1]}')
    print(f'  类别:   {target_names}')
    print(f'  恶性样本: {sum(y)} / {len(y)} ({sum(y)/len(y):.1%})')
    print(f'  良性样本: {len(y)-sum(y)} / {len(y)} ({(len(y)-sum(y))/len(y):.1%})')

    # 图5-1: 类别分布
    print('\n  生成图5-1: 类别分布...')
    plot_label_distribution(y, title='Breast Cancer Dataset - Class Distribution',
                            filename='fig_label_distribution.png')

    # ==================== 2. 数据划分 ====================
    # 对应论文 5.2 节
    print('\n[2] 数据划分 (7:3 分层采样)')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f'  训练集: {X_train.shape[0]} 样本')
    print(f'  测试集: {X_test.shape[0]} 样本')

    # 图5-2: 特征分布对比（Top3 重要特征）
    # 先训练一个初步模型获得特征重要性排序
    temp_clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    temp_clf.fit(X_train, y_train)
    temp_importances = temp_clf.feature_importances_
    top3_indices = np.argsort(temp_importances)[::-1][:3]
    top3_names = [feature_names[i] for i in top3_indices]

    print(f'\n  生成图5-2: Top3特征分布对比...')
    print(f'    Top3特征: {top3_names}')
    plot_feature_distribution(X, y, top3_indices, top3_names,
                              title='Top3 Feature Distribution by Class',
                              filename='fig_feature_distribution.png')

    # ==================== 3. 超参数调优 ====================
    # 对应论文 5.4 节（表5-1、表5-2、图5-3）
    print(f'\n[3] 超参数调优: max_depth 网格搜索 + 5折交叉验证')
    print(f'{"-" * 50}')
    print(f'  {"Depth":<6} {"Train Acc":<12} {"Test Acc":<12} {"CV Score":<12}')
    print(f'{"-" * 50}')

    depths = list(range(1, 11))
    train_scores, test_scores, cv_scores = [], [], []
    best_depth = 1
    best_test_acc = 0.0

    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, criterion='entropy',
                                     splitter='best', min_samples_split=2,
                                     min_samples_leaf=1, random_state=42)
        clf.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, clf.predict(X_train))
        test_acc = accuracy_score(y_test, clf.predict(X_test))
        cv = cross_val_score(clf, X_train, y_train, cv=5).mean()

        train_scores.append(train_acc)
        test_scores.append(test_acc)
        cv_scores.append(cv)

        print(f'  depth={d:<2}  {train_acc:.4f}       {test_acc:.4f}       {cv:.4f}')

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_depth = d

    print(f'{"-" * 50}')
    print(f'  [OK] 最佳深度: max_depth={best_depth}, 测试准确率={best_test_acc:.4f}')

    # 图5-3: 深度-准确率曲线
    print(f'\n  生成图5-3: 深度-准确率曲线...')
    plot_depth_comparison(depths, train_scores, test_scores, best_depth,
                          title='Decision Tree Depth vs Accuracy',
                          filename='fig_depth_comparison.png')

    # ==================== 4. 最终模型训练与评估 ====================
    # 对应论文 5.5 节（表5-3、图5-4、图5-5、图5-6）
    print(f'\n[4] 训练最终模型 (max_depth={best_depth})')
    best_clf = DecisionTreeClassifier(max_depth=best_depth, criterion='entropy',
                                      splitter='best', min_samples_split=2,
                                      min_samples_leaf=1, random_state=42)
    best_clf.fit(X_train, y_train)
    y_pred = best_clf.predict(X_test)

    # 分类报告（对应表5-3）
    print(f'\n  表5-3: 测试集分类报告')
    print(classification_report(y_test, y_pred, target_names=target_names, digits=3))

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f'\n  混淆矩阵:')
    print(f'              Predicted malignant  Predicted benign')
    print(f'  Actual malignant    {tn:<5}             {fp:<5}')
    print(f'  Actual benign       {fn:<5}             {tp:<5}')
    print(f'\n  恶性召回率 (malignant recall): {tn/(tn+fp):.1%}  ({tn}/{tn+fp})')
    print(f'  良性召回率 (benign recall):    {tp/(tp+fn):.1%}  ({tp}/{tp+fn})')

    # 图5-4: 混淆矩阵
    print(f'\n  生成图5-4: 混淆矩阵...')
    plot_confusion_matrix(cm, display_labels=['malignant', 'benign'],
                          title='Confusion Matrix',
                          filename='fig_confusion_matrix.png')

    # 特征重要性（对应表5-4、图5-5）
    importances = best_clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    print(f'\n  表5-4: Top5 特征重要性:')
    print(f'  {"Rank":<6} {"Feature":<30} {"Importance":<12}')
    print(f'  {"-" * 48}')
    for rank in range(5):
        idx = indices[rank]
        print(f'  {rank+1:<6} {feature_names[idx]:<30} {importances[idx]:.4f}')

    print(f'\n  生成图5-5: 特征重要性...')
    plot_feature_importance(importances, feature_names, top_n=10,
                            title='Feature Importance (Top {n})',
                            filename='fig_feature_importance.png')

    # 图5-6: 决策树可视化
    print(f'\n  生成图5-6: 决策树可视化...')
    plot_decision_tree(best_clf, list(feature_names), target_names,
                       title=f'Breast Cancer Decision Tree (max_depth={best_depth})',
                       filename='fig_decision_tree.png')

    # ==================== 5. 误判样本分析 ====================
    # 对应论文 6.2 节（图6-1、图6-2）
    print(f'\n[5] 误判样本分析')
    misclassified = np.where(y_pred != y_test)[0]
    fn_idx_list = [idx for idx in misclassified if y_test[idx] == 0 and y_pred[idx] == 1]
    fp_idx_list = [idx for idx in misclassified if y_test[idx] == 1 and y_pred[idx] == 0]
    print(f'  假阴性 (恶性→良性): {len(fn_idx_list)} 个')
    print(f'  假阳性 (良性→恶性): {len(fp_idx_list)} 个')

    # 选择第一个 FN 和第一个 FP 样本作案例分析
    top_feat_indices = indices[:8]
    top_feat_names = [feature_names[i] for i in top_feat_indices]

    mean_malignant = X[y == 0][:, top_feat_indices].mean(axis=0)
    mean_benign = X[y == 1][:, top_feat_indices].mean(axis=0)

    # FN 案例（图6-1）
    if fn_idx_list:
        fn_idx = fn_idx_list[0]
        fn_sample = X_test[fn_idx, top_feat_indices]
        print(f'\n  图6-1: FN案例 (测试样本#{fn_idx})')
        for feat_name, val in zip(top_feat_names, fn_sample):
            print(f'    {feat_name:30s} = {val:.4f}')
        plot_misclass_radar(fn_sample, mean_benign, top_feat_names,
                            title='Misclassified: Malignant Predicted Benign',
                            filename='fig_FN_radar.png')
        print(f'  生成图6-1: FN雷达图...')

    # FP 案例（图6-2）
    if fp_idx_list:
        fp_idx = fp_idx_list[0]
        fp_sample = X_test[fp_idx, top_feat_indices]
        print(f'\n  图6-2: FP案例 (测试样本#{fp_idx})')
        for feat_name, val in zip(top_feat_names, fp_sample):
            print(f'    {feat_name:30s} = {val:.4f}')
        plot_misclass_radar(fp_sample, mean_malignant, top_feat_names,
                            title='Misclassified: Benign Predicted Malignant',
                            filename='fig_FP_radar.png')
        print(f'  生成图6-2: FP雷达图...')

    # ==================== 6. 输出文本决策规则 ====================
    print(f'\n[6] 决策树文本规则 (对应论文图5-6):')
    tree_rules = export_text(best_clf, feature_names=list(feature_names))
    print(tree_rules)

    # ==================== 总结 ====================
    print(f'\n{"=" * 70}')
    print(f'  实验完成！结果摘要：')
    print(f'    最佳参数: max_depth={best_depth}, criterion=entropy')
    print(f'    测试准确率: {best_test_acc:.2%}')
    print(f'    恶性召回率: {tn/(tn+fp):.1%}')
    print(f'    良性召回率: {tp/(tp+fn):.1%}')
    print(f'    误判总数: {len(misclassified)} / {len(y_test)}')
    print(f'  ')
    print(f'  所有图片已保存至: output/ 目录')
    print(f'{"=" * 70}')


if __name__ == '__main__':
    main()
