import numpy as np
import pandas as pd


def calculate_entropy(column):
    p_data = column.value_counts(normalize=True)
    return -np.sum(p_data * np.log2(p_data))


def calculate_info_gain(df, feature_column, target_column):

    parent_entropy = calculate_entropy(df[target_column])

    child_entropy = 0
    total_rows = len(df)

    for value in df[feature_column].unique():

        subset = df[df[feature_column] == value]

        weights = len(subset) / total_rows

        child_entropy += weights * (calculate_entropy(subset[target_column]))

    return parent_entropy - child_entropy


def find_best_feature(df, n_features_names, target_column):

    info_gains = [
        calculate_info_gain(df, name, target_column) for name in n_features_names
    ]

    idx = np.argmax(info_gains)

    return n_features_names[idx]


def most_frequent_class(column):

    return column.value_counts().idxmax()


class Node:

    def __init__(self, feature=None, label=None, parent_Node_Class=None):
        self.feature = feature
        self.label = label
        self.children = {}
        self.parent_Node_Class = parent_Node_Class

    def is_leaf_Node(self):
        return self.feature == None


class DecisionTree:
    def __init__(self, target_attribute_name="target"):
        self.target_attribute_name = target_attribute_name
        self.root = None

    def fit(self, data):
        # extract the feature names from data frame
        feature_names = [
            col for col in data.columns if col != self.target_attribute_name
        ]
        self.root = self.id3(data, data, feature_names)

    def id3(self, original_data, data, feature_names, parent_Node_Class=None):

        target = self.target_attribute_name

        if len(np.unique(data[target])) <= 1:
            # if all the column contain only one class then it decides it as leaf node
            return Node(label=data[target].iloc[0])
        elif len(data) == 0:
            majority_class = most_frequent_class(original_data[target])
            return Node(label=majority_class)
        elif len(feature_names) == 0:
            return Node(label=parent_Node_Class)

        else:

            best_feature = find_best_feature(data, feature_names, target)

            remaining_features = [
                feature for feature in feature_names if feature != best_feature
            ]

           

            parent_Node_Class = most_frequent_class(data[target])
            node = Node(feature=best_feature,parent_Node_Class=parent_Node_Class)

            for value in data[best_feature].unique():

                subset = data[data[best_feature] == value]
                # recursive call
                child = self.id3(
                    original_data, subset, remaining_features, parent_Node_Class
                )

                node.children[value] = child

            return node

    def predict_one(self, sample):

        node = self.root

        while not node.is_leaf_Node():

            feature_value = sample[node.feature]

            if feature_value in node.children:
                node = node.children[feature_value]
            else:
                return node.parent_Node_Class

        return node.label

    def predict(self, data):
        return data.apply(self.predict_one, axis=1)
    
    def print_tree(self, node=None, indent=""):
        if node is None:
            node = self.root

        if node.is_leaf_Node():
            print(indent + "Label:", node.label)
            return

        print(indent + "Feature:", node.feature)
        for value, child in node.children.items():
            print(indent + f"  {value} ->")
            self.print_tree(child, indent + "    ")


df=pd.read_csv("Profits_Classification.csv")


