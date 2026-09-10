"""A small ID3-style decision tree for nominal classification features."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Hashable

import numpy as np


@dataclass
class Node:
    """One node in a multiway decision tree."""

    # feature：当前节点用于分裂数据的属性索引；叶节点没有分裂属性，所以是 None。
    feature: int | None = None
    # children：nominal 属性的每个取值对应一个子树。
    # 例如 {0: child_0, 1: child_1, 2: child_2}，因此不能只假设 left/right。
    children: dict[Hashable, "Node"] = field(default_factory=dict)
    # gain / entropy：记录当前分裂的信息增益和分裂前的熵，用于输出树结构。
    gain: float | None = None
    # value：叶节点最终预测的多数类别；内部节点为 None。
    value: Hashable | None = None
    # class_counts：当前节点中各类别的样本数量，例如 {0: 10, 1: 5}。
    class_counts: dict[Hashable, int] = field(default_factory=dict)
    entropy: float | None = None
    # default_class：预测时遇到训练集中没出现过的 nominal value 时使用的回退类别。
    default_class: Hashable | None = None

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


class DecisionTree:
    """ID3-style classifier for nominal features.

    Each split creates one child for every observed value of the selected feature.
    Features are not reused on the same root-to-leaf path.
    """

    def __init__(
        self,
        min_samples: int = 2,
        max_depth: int | None = None,
        min_information_gain: float = 1e-5,
    ) -> None:
        if min_samples < 1:
            raise ValueError("min_samples must be at least 1")
        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth must be non-negative or None")

        # 内部节点继续分裂所需的最小样本数。
        self.min_samples = min_samples
        # 根节点深度为 0；达到 max_depth 后生成叶节点。
        self.max_depth = max_depth
        # 信息增益太小时不再继续分裂，避免没有意义的子树。
        self.min_information_gain = min_information_gain
        self.root: Node | None = None

    @staticmethod
    def entropy(labels: np.ndarray) -> float:
        """Return Shannon entropy for arbitrary nominal class labels."""
        if len(labels) == 0:
            return 0.0
        # 先统计每个类别的数量，再转成概率；entropy = -sum(p * log2(p))。
        _, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        return float(-np.sum(probabilities * np.log2(probabilities)))

    @staticmethod
    def _class_counts(labels: np.ndarray) -> dict[Hashable, int]:
        values, counts = np.unique(labels, return_counts=True)
        return {
            value.item() if isinstance(value, np.generic) else value: int(count)
            for value, count in zip(values, counts)
        }

    @staticmethod
    def _majority_class(labels: np.ndarray) -> Hashable:
        # 叶节点使用数量最多的类别作为预测；平票时 np.unique 的较小值优先。
        values, counts = np.unique(labels, return_counts=True)
        winner = values[int(np.argmax(counts))]
        return winner.item() if isinstance(winner, np.generic) else winner

    @staticmethod
    def split_data(dataset: np.ndarray, feature: int) -> dict[Hashable, np.ndarray]:
        """Partition a dataset once for every observed nominal feature value."""
        partitions: dict[Hashable, np.ndarray] = {}
        # nominal feature 不是按照 <= threshold 和 > threshold 分左右两支。
        # 每个不同取值都生成一个独立分支，这对 rtg_C 的 att3 尤其重要。
        for value in np.unique(dataset[:, feature]):
            key = value.item() if isinstance(value, np.generic) else value
            partitions[key] = dataset[dataset[:, feature] == value]
        return partitions

    def information_gain(
        self,
        parent_labels: np.ndarray,
        child_labels: list[np.ndarray],
    ) -> float:
        """Return entropy reduction produced by a multiway nominal split."""
        # 信息增益 = 分裂前的熵 - 所有子节点熵的加权平均。
        # 多路分支中，每个子节点的权重是该分支样本数 / 父节点样本数。
        weighted_child_entropy = sum(
            len(labels) / len(parent_labels) * self.entropy(labels)
            for labels in child_labels
        )
        return self.entropy(parent_labels) - weighted_child_entropy

    def best_split(
        self,
        dataset: np.ndarray,
        available_features: tuple[int, ...],
    ) -> dict[str, object] | None:
        """Find the nominal feature with the largest multiway information gain."""
        parent_labels = dataset[:, -1]
        best_gain = self.min_information_gain
        best: dict[str, object] | None = None

        # 从所有仍可用的属性中寻找信息增益最大的属性。
        # 按索引顺序遍历；信息增益相同时保留编号较小的属性，保证结果稳定。
        for feature in available_features:
            partitions = self.split_data(dataset, feature)
            # 当前节点中只有一个取值的属性无法产生有效分裂。
            if len(partitions) < 2:
                continue

            gain = self.information_gain(
                parent_labels,
                [partition[:, -1] for partition in partitions.values()],
            )
            # 只有真正超过当前最佳值时才更新，不能忘记同步 best_gain。
            if gain > best_gain:
                best_gain = gain
                best = {
                    "feature": feature,
                    "partitions": partitions,
                    "gain": gain,
                }

        return best

    def build_tree(
        self,
        dataset: np.ndarray,
        current_depth: int = 0,
        available_features: tuple[int, ...] | None = None,
    ) -> Node:
        """Recursively build a multiway tree from a feature-plus-label dataset."""
        labels = dataset[:, -1]
        counts = self._class_counts(labels)
        majority_class = self._majority_class(labels)

        if available_features is None:
            available_features = tuple(range(dataset.shape[1] - 1))

        depth_limit_reached = (
            self.max_depth is not None and current_depth >= self.max_depth
        )
        # 停止条件：类别已经纯净、样本太少、达到最大深度，或者没有剩余属性。
        should_stop = (
            len(np.unique(labels)) == 1
            or len(dataset) < self.min_samples
            or depth_limit_reached
            or not available_features
        )
        if should_stop:
            return Node(
                value=majority_class,
                class_counts=counts,
                default_class=majority_class,
            )

        split = self.best_split(dataset, available_features)
        if split is None:
            return Node(
                value=majority_class,
                class_counts=counts,
                default_class=majority_class,
            )

        feature = int(split["feature"])
        # nominal ID3 在同一条根到叶路径上不重复使用已经分裂过的属性。
        remaining_features = tuple(
            candidate for candidate in available_features if candidate != feature
        )
        # 对该属性的每个 nominal value 分别递归建立一棵子树。
        children = {
            value: self.build_tree(
                partition,
                current_depth + 1,
                remaining_features,
            )
            for value, partition in split["partitions"].items()
        }

        return Node(
            feature=feature,
            children=children,
            gain=float(split["gain"]),
            entropy=self.entropy(labels),
            class_counts=counts,
            default_class=majority_class,
        )

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "DecisionTree":
        """Fit the tree and return self."""
        # 将 X 与最后一列标签 y 合并，保持原作业代码的数据表示方式。
        features = np.asarray(features)
        labels = np.asarray(labels)
        if features.ndim != 2:
            raise ValueError("features must be a two-dimensional array")
        if labels.ndim != 1:
            raise ValueError("labels must be a one-dimensional array")
        if len(features) == 0 or len(features) != len(labels):
            raise ValueError("features and labels must have the same non-zero length")

        dataset = np.column_stack((features, labels))
        self.root = self.build_tree(dataset)
        return self

    def _predict_one(self, sample: np.ndarray) -> Hashable:
        if self.root is None:
            raise RuntimeError("fit must be called before predict")

        node = self.root
        while not node.is_leaf:
            # 根据样本在当前 nominal 属性上的具体取值选择对应子树。
            raw_value = sample[node.feature]
            value = raw_value.item() if isinstance(raw_value, np.generic) else raw_value
            child = node.children.get(value)
            if child is None:
                # 测试数据出现训练集中未见过的取值时，安全回退到当前节点多数类。
                return node.default_class
            node = child
        return node.value

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict one class label for every row."""
        features = np.asarray(features)
        if features.ndim != 2:
            raise ValueError("features must be a two-dimensional array")
        return np.asarray([self._predict_one(sample) for sample in features])

    def score(self, features: np.ndarray, labels: np.ndarray) -> float:
        return float(np.mean(self.predict(features) == np.asarray(labels)))

    @staticmethod
    def _format_value(value: Hashable) -> str:
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    def _tree_lines(self, node: Node, indent: str = "") -> list[str]:
        if node.is_leaf:
            # 叶节点同时输出类别分布与最后的预测类别。
            counts = ", ".join(
                f"{self._format_value(label)}: {count}"
                for label, count in node.class_counts.items()
            )
            return [
                f"{indent}leaf {{{counts}}} -> class {self._format_value(node.value)}"
            ]

        lines = [
            f"{indent}feature {node.feature} "
            f"(IG: {node.gain:.4f}, Entropy: {node.entropy:.4f})"
        ]
        try:
            values = sorted(node.children)
        except TypeError:
            values = sorted(node.children, key=str)

        # 每个取值显示为 "feature i == value"，与实际预测条件完全一致。
        for value in values:
            lines.append(
                f"{indent}-- feature {node.feature} == {self._format_value(value)} --"
            )
            lines.extend(self._tree_lines(node.children[value], indent + "    "))
        return lines

    def print_tree(self, output_file: str | Path | None = None) -> str:
        """Print the fitted tree and optionally save the same text to a file."""
        if self.root is None:
            raise RuntimeError("fit must be called before print_tree")
        text = "\n".join(self._tree_lines(self.root))
        print(text)
        if output_file is not None:
            Path(output_file).write_text(text + "\n", encoding="utf-8")
        return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a multiway decision tree for nominal features."
    )
    parser.add_argument("train_file", type=Path, help="CSV file with label in last column")
    parser.add_argument("output_file", type=Path, help="Where to save the text tree")
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--min-samples", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # CSV 的最后一列是类别标签，其余列都是 nominal features。
    data = np.loadtxt(args.train_file, delimiter=",", skiprows=1)
    features, labels = data[:, :-1], data[:, -1]

    classifier = DecisionTree(
        max_depth=args.max_depth,
        min_samples=args.min_samples,
    ).fit(features, labels)
    # 这里是训练集准确率，只用于检查建树行为，不代表测试集泛化性能。
    training_accuracy = classifier.score(features, labels)
    print(
        f"Training accuracy: {training_accuracy:.4f} "
        f"({training_accuracy * 100:.2f}%)"
    )
    classifier.print_tree(args.output_file)
    print(f"Decision tree saved to {args.output_file}")


if __name__ == "__main__":
    main()
