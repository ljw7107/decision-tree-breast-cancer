# 决策树算法复现 — 《人工智能》课程论文配套代码

## 项目简介

本项目为《人工智能》课程论文《决策树算法的原理复现与实验分析》的配套源代码。项目完整实现了 **ID3** 和 **CART** 两种经典决策树算法，并在 **乳腺癌威斯康星数据集**（Breast Cancer Wisconsin Diagnostic Dataset）上完成了二分类实验，系统验证了决策树算法在医疗诊断领域的实用价值。

项目从零实现了 ID3 决策树（含信息熵、信息增益、递归建树全流程），同时基于 scikit-learn 框架构建了 CART 决策树，对不同树深度进行了网格搜索与交叉验证，最终在测试集上达到 **95.32%** 的分类准确率。

**核心结论：**
- 最佳超参数：`max_depth=4`, `criterion='entropy'`
- 测试集准确率：**95.32%**
- 恶性（malignant）召回率：**92.2%**
- 良性（benign）召回率：**97.2%**
- 最关键特征：**worst radius**（重要性占比 63.93%）

## 环境要求

- **Python** >= 3.8
- **操作系统**：Windows / Linux / macOS 均可
- **主要依赖库**：

| 库名 | 最低版本 | 用途 |
|------|---------|------|
| scikit-learn | 1.0.0 | 决策树模型、数据集、评估指标 |
| numpy | 1.20.0 | 数值计算 |
| pandas | 1.3.0 | 数据处理 |
| matplotlib | 3.4.0 | 可视化（决策树图、混淆矩阵、曲线图等） |

## 安装步骤

### 1. 克隆或下载项目

```bash
git clone https://github.com/ljw7107/decision-tree-breast-cancer.git
cd decision-tree-breast-cancer
```

或直接下载 ZIP 压缩包并解压。

### 2. 创建虚拟环境（推荐）

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 数据准备

本实验使用的 **乳腺癌威斯康星数据集** 已预先导出并存储在项目 `data/` 目录下，**无需额外下载**。

数据集信息：
- **来源**：美国威斯康星大学医学院（UCI Machine Learning Repository）
- **文件**：`data/breast_cancer.csv`（569 行 × 31 列，含 30 个特征 + 目标标签）
- **样本数**：569 个
- **特征数**：30 个（细胞核形态学特征）
- **类别**：0 = 恶性（malignant）、1 = 良性（benign）
- **文献引用**：Wolberg W H, et al. Breast Cancer Wisconsin (Diagnostic) Data Set. UCI, 1995.
- **元数据**：`data/dataset_info.txt`（含完整特征说明）

## 运行指南

### 1. ID3 决策树演示实验

使用鸢尾花数据集验证自实现 ID3 算法的正确性：

```bash
python experiments/iris_demo.py
```

**预期输出：**
- 鸢尾花数据集基本信息（150 样本、4 特征、3 类别）
- ID3 决策树的字典结构
- 测试集分类准确率

### 2. 乳腺癌 CART 决策树主实验

完整复现论文全部实验结果：

```bash
python experiments/breast_cancer_cart.py
```

**预期输出（终端）：**
- 数据集基本信息与类别分布
- max_depth 1~10 网格搜索对比表（训练准确率、测试准确率、5 折交叉验证得分）
- 最佳模型配置与测试集分类报告（精确率、召回率、F1 分数）
- 混淆矩阵详细数值
- Top5 特征重要性排名
- 误判样本分析（假阴性、假阳性各一例）
- 决策树文本规则

**生成文件（output/ 目录）：**

| 文件名 | 对应论文 | 说明 |
|--------|---------|------|
| `fig_label_distribution.png` | 图5-1 | 数据集类别分布柱状图 |
| `fig_feature_distribution.png` | 图5-2 | Top3 特征分布对比直方图 |
| `fig_depth_comparison.png` | 图5-3 | 决策树深度-准确率曲线 |
| `fig_confusion_matrix.png` | 图5-4 | 混淆矩阵热力图 |
| `fig_feature_importance.png` | 图5-5 | 特征重要性水平条形图 |
| `fig_decision_tree.png` | 图5-6 | 决策树结构可视化 |
| `fig_FN_radar.png` | 图6-1 | 假阴性误判样本雷达图 |
| `fig_FP_radar.png` | 图6-2 | 假阳性误判样本雷达图 |

## 项目结构

```
decision-tree-breast-cancer/
│
├── README.md                     # 项目说明文档（本文件）
├── requirements.txt              # Python 依赖清单
│
├── data/                         # 实验数据集（CSV 格式，从 sklearn 导出）
│   ├── breast_cancer.csv          # 乳腺癌威斯康星数据集（569 × 31）
│   └── dataset_info.txt           # 数据说明与特征元数据
│
├── src/                          # 核心算法源码
│   ├── __init__.py
│   └── id3_tree.py               # 从零实现的 ID3 决策树算法
│       ├── Id3Tree._calc_shannon_ent()   # 信息熵计算
│       ├── Id3Tree._choose_best_feature() # 信息增益最优特征选择
│       ├── Id3Tree._create_tree()         # 递归构建决策树
│       ├── Id3Tree.fit()                  # 训练接口
│       └── Id3Tree.predict()              # 预测接口
│
├── experiments/                  # 实验运行脚本
│   ├── __init__.py
│   ├── iris_demo.py              # 鸢尾花数据集 ID3 演示实验
│   └── breast_cancer_cart.py     # 乳腺癌数据集 CART 主实验（论文核心）
│       ├── 网格搜索 + 交叉验证
│       ├── 分类报告 + 混淆矩阵
│       ├── 特征重要性分析
│       ├── 误判样本案例分析
│       └── 自动生成全部配图
│
├── utils/                        # 工具函数
│   ├── __init__.py
│   └── plot_utils.py             # 统一绘图工具封装（7 个图表函数）
│
└── output/                       # 运行后自动生成，存放所有图片
    ├── fig_label_distribution.png
    ├── fig_feature_distribution.png
    ├── fig_depth_comparison.png
    ├── fig_confusion_matrix.png
    ├── fig_feature_importance.png
    ├── fig_decision_tree.png
    ├── fig_FN_radar.png
    └── fig_FP_radar.png
```

## 核心实验结果

### 最佳模型参数

| 参数 | 取值 |
|------|------|
| algorithm | CART (Classification and Regression Tree) |
| criterion | entropy（信息熵） |
| splitter | best（最优分裂） |
| max_depth | 4 |
| min_samples_split | 2 |
| min_samples_leaf | 1 |

### 测试集分类性能

| 类别 | 精确率 | 召回率 | F1 分数 | 样本数 |
|------|--------|--------|---------|-------|
| malignant | 0.952 | 0.922 | 0.937 | 64 |
| benign | 0.954 | 0.972 | 0.963 | 107 |
| **总体** | **0.953** | **0.953** | **0.953** | **171** |

### 混淆矩阵

| | 预测恶性 | 预测良性 |
|---|:-------:|:--------:|
| 实际恶性 | 59 | 5 |
| 实际良性 | 3 | 104 |

### 特征重要性 Top 5

| 排名 | 特征 | 重要性 |
|:---:|------|:------:|
| 1 | worst radius | 0.6393 |
| 2 | worst concave points | 0.1726 |
| 3 | mean concavity | 0.0551 |
| 4 | worst texture | 0.0339 |
| 5 | mean texture | 0.0308 |

## 参考资料

1. Quinlan J R. Induction of decision trees[J]. Machine learning, 1986, 1(1): 81-106. — **ID3 算法原始论文**
2. Breiman L, Friedman J, Olshen R, et al. Classification and regression trees[M]. CRC press, 1984. — **CART 算法原始论文**
3. 周志华. 机器学习[M]. 北京: 清华大学出版社, 2016. — **机器学习教材**
4. Jack-Cherish. Machine-Learning[EB/OL]. GitHub, https://github.com/Jack-Cherish/Machine-Learning, 2017. — **参考的开源项目（ID3 自实现参考）**
5. Wolberg W H, Mangasarian O L, Street W N, et al. Breast Cancer Wisconsin (Diagnostic) Data Set[DS]. UCI Machine Learning Repository, 1995. — **实验使用数据集**
6. Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: Machine learning in Python[J]. Journal of machine learning research, 2011, 12(Oct): 2825-2830. — **scikit-learn 库文档**

## License

本项目仅供学习交流使用，代码基于 MIT 协议开源。
