"""
绘图工具模块
=============
封装统一的图表绘制函数，规范论文配图样式。
所有图片统一保存至项目根目录下 output 文件夹。

对应论文配图：
  - 图5-1 类别分布（plot_label_distribution）
  - 图5-2 Top3特征分布（plot_feature_distribution）
  - 图5-3 深度-准确率曲线（plot_depth_comparison）
  - 图5-4 混淆矩阵（plot_confusion_matrix）
  - 图5-5 特征重要性（plot_feature_importance）
  - 图5-6 决策树可视化（plot_decision_tree）
  - 图6-1/6-2 误判样本雷达图（plot_misclass_radar）
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from math import pi

# 全局设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')


def _ensure_output_dir():
    """确保 output 目录存在"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _save_fig(fig, filename, dpi=200):
    """保存图片至 output 目录"""
    _ensure_output_dir()
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f'  图片已保存: {filepath}')
    plt.close(fig)


# ==================== 图5-1: 类别分布柱状图 ====================

def plot_label_distribution(y, title='Dataset Class Distribution', filename='fig_label_distribution.png'):
    """
    绘制数据集类别分布柱状图
    对应论文图5-1
    """
    classes = sorted(set(y))
    counts = [sum(y == c) for c in classes]
    labels = [f'Class {c}' for c in classes]

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ['#d66c6c', '#6c9ed6'][:len(classes)]
    bars = ax.bar(labels, counts, color=colors, edgecolor='white', width=0.5)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(count), ha='center', va='bottom', fontsize=13, fontweight='bold')

    ax.set_ylabel('Sample Count', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    _save_fig(fig, filename)


# ==================== 图5-2: 特征分布对比 ====================

def plot_feature_distribution(X, y, feature_indices, feature_names,
                               title='Feature Distribution by Class',
                               filename='fig_feature_distribution.png'):
    """
    绘制指定特征在各类别下的分布对比直方图
    对应论文图5-2
    """
    n_features = len(feature_indices)
    fig, axes = plt.subplots(1, n_features, figsize=(5 * n_features, 4.5))

    if n_features == 1:
        axes = [axes]

    for i, (idx, name) in enumerate(zip(feature_indices, feature_names)):
        ax = axes[i]
        classes = sorted(set(y))
        colors = ['#d66c6c', '#6c9ed6']
        labels_map = {0: 'malignant', 1: 'benign'}

        for c in classes:
            data = X[y == c, idx]
            ax.hist(data, bins=20, alpha=0.7, color=colors[c],
                    label=labels_map.get(c, f'class_{c}'), density=True)

        ax.set_xlabel(name, fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.legend(fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    _save_fig(fig, filename)


# ==================== 图5-3: 深度-准确率曲线 ====================

def plot_depth_comparison(depths, train_scores, test_scores, best_depth,
                           title='Decision Tree Depth vs Accuracy',
                           filename='fig_depth_comparison.png'):
    """
    绘制不同树深度下的训练/测试准确率对比曲线
    对应论文图5-3
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(depths, train_scores, 'b-o', label='Training Accuracy',
            linewidth=2, markersize=7)
    ax.plot(depths, test_scores, 'r-s', label='Testing Accuracy',
            linewidth=2, markersize=7)
    ax.axvline(x=best_depth, color='gray', linestyle=':', alpha=0.6, linewidth=1.5)
    ax.annotate(f'Best depth={best_depth}', xy=(best_depth + 0.2, 0.96),
                fontsize=11, color='gray')

    ax.set_xlabel('Max Depth', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(depths)
    ax.set_ylim(0.88, 1.01)
    ax.legend(fontsize=11, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, filename)


# ==================== 图5-4: 混淆矩阵 ====================

def plot_confusion_matrix(cm, display_labels,
                           title='Confusion Matrix',
                           filename='fig_confusion_matrix.png'):
    """
    绘制混淆矩阵热力图
    对应论文图5-4
    """
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(False)
    plt.tight_layout()
    _save_fig(fig, filename)


# ==================== 图5-5: 特征重要性 ====================

def plot_feature_importance(importances, feature_names, top_n=10,
                             title='Feature Importance (Top {n})',
                             filename='fig_feature_importance.png'):
    """
    绘制特征重要性水平条形图
    对应论文图5-5
    """
    indices = np.argsort(importances)[::-1][:top_n]
    vals = importances[indices][::-1]
    lbls = [feature_names[i] for i in indices[::-1]]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n), vals, color='#4682B4', edgecolor='white')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(lbls, fontsize=10)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(title.format(n=top_n), fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    _save_fig(fig, filename)


# ==================== 图5-6: 决策树可视化 ====================

def plot_decision_tree(clf, feature_names, class_names,
                        title='Decision Tree Visualization',
                        filename='fig_decision_tree.png'):
    """
    绘制 sklearn 决策树结构图
    对应论文图5-6
    """
    from sklearn.tree import plot_tree

    fig, ax = plt.subplots(figsize=(18, 10))
    plot_tree(clf, filled=True, rounded=True,
              feature_names=feature_names,
              class_names=class_names,
              fontsize=9, ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save_fig(fig, filename)


# ==================== 图6-1/6-2: 误判样本雷达图 ====================

def plot_misclass_radar(sample_features, comp_features, feature_names,
                         title, filename):
    """
    绘制误判样本与对照均值的特征对比雷达图
    对应论文图6-1（FN案例）、图6-2（FP案例）
    """
    n_features = len(feature_names)
    angles = [n / float(n_features) * 2 * pi for n in range(n_features)]
    angles += angles[:1]

    # 归一化到 [0, 1]
    all_vals = np.vstack([sample_features, comp_features])
    mins = all_vals.min(axis=0)
    maxs = all_vals.max(axis=0)

    def normalize(vals):
        return (vals - mins) / (maxs - mins + 1e-10)

    sample_norm = normalize(sample_features)
    comp_norm = normalize(comp_features)

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names, fontsize=9)

    values = list(sample_norm) + [sample_norm[0]]
    comp = list(comp_norm) + [comp_norm[0]]

    ax.plot(angles, values, 'r-', linewidth=2, label='Misclassified Sample')
    ax.fill(angles, values, 'r', alpha=0.1)
    ax.plot(angles, comp, 'b--', linewidth=1.5, label='Comparison Mean')
    ax.fill(angles, comp, 'b', alpha=0.05)

    ax.set_title(title, fontsize=11, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    _save_fig(fig, filename)
