"""
ID3 决策树算法从零实现
========================
对应论文章节：4.2（信息熵与信息增益）、4.3（ID3算法）

该模块从零实现了 ID3 决策树算法，包含：
  - 信息熵（Shannon Entropy）计算
  - 信息增益（Information Gain）计算
  - 基于信息增益的最优特征选择
  - 递归构建决策树
  - 对新样本进行分类预测

Reference:
  Quinlan J R. Induction of decision trees[J]. Machine learning, 1986, 1(1): 81-106.
"""

from math import log
import operator


class Id3Tree:
    """
    ID3 决策树分类器

    使用信息增益作为特征选择准则，递归构建决策树。
    支持离散特征的分类任务。

    Attributes:
        tree (dict): 训练完成后生成的决策树（字典嵌套结构）
        feat_labels (list): 训练过程中选择的最优特征标签（按选择顺序）
    """

    def __init__(self):
        self.tree = None
        self.feat_labels = None

    # ==================== 信息熵计算 ====================
    # 对应论文 4.2 节：H(D) = -Σ p_k · log₂(p_k)

    @staticmethod
    def _calc_shannon_ent(data_set):
        """
        计算数据集的经验熵（香农熵）

        参数:
            data_set (list): 数据集，每行为一条样本，最后一列为类别标签

        返回:
            float: 经验熵值 H(D)
        """
        num_entries = len(data_set)
        label_counts = {}

        # 统计各类别出现的次数
        for feat_vec in data_set:
            current_label = feat_vec[-1]
            if current_label not in label_counts:
                label_counts[current_label] = 0
            label_counts[current_label] += 1

        shannon_ent = 0.0
        for key in label_counts:
            prob = float(label_counts[key]) / num_entries  # p_k
            shannon_ent -= prob * log(prob, 2)             # -p_k · log₂(p_k)

        return shannon_ent

    # ==================== 数据划分 ====================

    @staticmethod
    def _split_data_set(data_set, axis, value):
        """
        按照指定特征的值划分数据集

        参数:
            data_set (list): 待划分的数据集
            axis (int): 特征索引
            value: 特征取值（保留该取值的样本）

        返回:
            list: 划分后的子集（已去掉 axis 特征列）
        """
        ret_data_set = []
        for feat_vec in data_set:
            if feat_vec[axis] == value:
                # 去掉 axis 列，保留其余列
                reduced_vec = feat_vec[:axis]
                reduced_vec.extend(feat_vec[axis + 1:])
                ret_data_set.append(reduced_vec)
        return ret_data_set

    # ==================== 最优特征选择 ====================
    # 对应论文 4.2 节：g(D, A) = H(D) - H(D|A)

    @staticmethod
    def _choose_best_feature(data_set):
        """
        通过信息增益选择最优特征

        参数:
            data_set (list): 数据集

        返回:
            int: 最优特征的索引值
        """
        num_features = len(data_set[0]) - 1  # 特征数量
        base_entropy = Id3Tree._calc_shannon_ent(data_set)  # H(D)
        best_info_gain = 0.0
        best_feature = -1

        for i in range(num_features):
            # 获取当前特征的所有取值
            feat_list = [example[i] for example in data_set]
            unique_vals = set(feat_list)

            new_entropy = 0.0
            # 计算条件熵 H(D|A)
            for value in unique_vals:
                sub_data_set = Id3Tree._split_data_set(data_set, i, value)
                prob = len(sub_data_set) / float(len(data_set))
                new_entropy += prob * Id3Tree._calc_shannon_ent(sub_data_set)

            info_gain = base_entropy - new_entropy  # 信息增益 g(D, A)

            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = i

        return best_feature

    # ==================== 多数投票 ====================

    @staticmethod
    def _majority_cnt(class_list):
        """
        统计 class_list 中出现次数最多的类别标签（多数投票）

        参数:
            class_list (list): 类别标签列表

        返回:
            出现次数最多的类别标签
        """
        class_count = {}
        for vote in class_list:
            if vote not in class_count:
                class_count[vote] = 0
            class_count[vote] += 1

        # 按出现次数降序排列
        sorted_class_count = sorted(
            class_count.items(), key=operator.itemgetter(1), reverse=True
        )
        return sorted_class_count[0][0]

    # ==================== 递归构建决策树 ====================
    # 对应论文 4.3 节：ID3 算法流程

    def _create_tree(self, data_set, labels, feat_labels):
        """
        递归构建决策树

        参数:
            data_set (list): 当前子数据集
            labels (list): 当前剩余特征标签
            feat_labels (list): 已选特征标签（用于递归传参）

        返回:
            dict: 以字典嵌套形式表示的决策树

        递归终止条件：
            (1) 当前子集所有样本属于同一类别 → 返回该类别
            (2) 特征用尽 → 返回多数投票结果
        """
        class_list = [example[-1] for example in data_set]

        # 终止条件(1)：所有样本类别相同
        if class_list.count(class_list[0]) == len(class_list):
            return class_list[0]

        # 终止条件(2)：遍历完所有特征
        if len(data_set[0]) == 1 or len(labels) == 0:
            return self._majority_cnt(class_list)

        # 选择信息增益最大的特征
        best_feat = self._choose_best_feature(data_set)
        best_feat_label = labels[best_feat]
        feat_labels.append(best_feat_label)

        # 以该特征为根节点构建子树
        my_tree = {best_feat_label: {}}
        del labels[best_feat]

        # 获取该特征的所有取值
        feat_values = [example[best_feat] for example in data_set]
        unique_vals = set(feat_values)

        # 对每个取值递归构建子树
        for value in unique_vals:
            sub_labels = labels[:]
            my_tree[best_feat_label][value] = self._create_tree(
                self._split_data_set(data_set, best_feat, value),
                sub_labels, feat_labels
            )

        return my_tree

    # ==================== 训练接口 ====================

    def fit(self, data_set, labels):
        """
        训练 ID3 决策树模型

        参数:
            data_set (list): 训练数据集，每行最后一项为类别标签
            labels (list): 特征名称列表，与 data_set 各列一一对应

        返回:
            self: 训练完成的 ID3 决策树对象
        """
        labels_copy = labels[:]
        self.feat_labels = []
        self.tree = self._create_tree(data_set, labels_copy, self.feat_labels)
        return self

    # ==================== 预测接口 ====================

    def predict(self, test_vec):
        """
        对单个测试样本进行分类预测

        参数:
            test_vec (list): 测试样本的特征值列表（按 feat_labels 顺序）

        返回:
            预测的类别标签
        """
        return self._classify(self.tree, self.feat_labels, test_vec)

    def _classify(self, input_tree, feat_labels, test_vec):
        """递归遍历决策树进行分类"""
        first_str = next(iter(input_tree))
        second_dict = input_tree[first_str]
        feat_index = feat_labels.index(first_str)

        for key in second_dict.keys():
            if test_vec[feat_index] == key:
                if isinstance(second_dict[key], dict):
                    return self._classify(second_dict[key], feat_labels, test_vec)
                else:
                    return second_dict[key]

        # 若未匹配到任何分支（测试数据特征值不在训练范围内），返回 None
        return None

    # ==================== 树结构可视化 ====================

    @staticmethod
    def _convert_tree(tree):
        """递归将树中的 numpy 类型转换为 Python 原生类型，确保 JSON 可序列化"""
        if isinstance(tree, dict):
            return {str(k): Id3Tree._convert_tree(v) for k, v in tree.items()}
        elif isinstance(tree, list):
            return [Id3Tree._convert_tree(item) for item in tree]
        elif hasattr(tree, 'item'):  # numpy 数值类型
            return tree.item()
        return tree

    def print_tree(self):
        """打印决策树的字典结构"""
        import json
        if self.tree:
            converted = self._convert_tree(self.tree)
            print(json.dumps(converted, ensure_ascii=False, indent=2))
        else:
            print("模型尚未训练，请先调用 fit() 方法。")
