"""
    This module visualises the decision tree.
"""
import sys

import decision_tree_analysis
import matplotlib.pyplot as plt


class VisualNode:
    """Represents a visual node of a decision tree."""

    def __init__(self, name, x, y, width=2, height=0.75, left=None, right=None):
        """Constructor to store the attributes of a visual node."""
        self.name = name
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.left = left
        self.right = right


def get_name(node):
    """Returns a text representation of a node, matching the example in the specification."""
    if decision_tree_analysis.is_leaf_node(node):
        return "leaf:" + str(node["value"])
    else:
        return "[X" + str(node["attribute"]) + " < " + str(node["value"]) + "]"


def draw_node(name, position, parent):
    """Draws an annotation to represent a tree node with a line beginning from the parent."""
    # Offset line origin to be below the node's position.
    parent_line_origin = (parent.get_position()[0], parent.get_position()[1] - 0.13)
    ax.annotate("", xy=position, xycoords='data', xytext=parent_line_origin,
                textcoords='data', arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),)
    return ax.annotate(name, xy=position, xycoords="data", va="center",
                       ha="center", bbox=dict(boxstyle="round", fc="w"))


def draw_root_node(name, position):
    """Draws an annotation to represent a single tree node."""
    return ax.annotate(name, xy=position, xycoords="data", va="center",
                       ha="center", bbox=dict(boxstyle="round", fc="w"))


def calculate_subtree_widths(tree_node, is_left):
    """Calculates the widths of the subtree so that it can be positioned as compactful as possible."""
    if (decision_tree_analysis.is_leaf_node(tree_node)):
        # Position the leaf nodes closer to the parent for compactness.
        return VisualNode(get_name(tree_node), (-1*is_left)+0.5, -0.5)
    else:
        l_node = calculate_subtree_widths(tree_node["left"], True)
        r_node = calculate_subtree_widths(tree_node["right"], False)
        # Width of the current node is the sum of the subtrees.
        width = l_node.width + r_node.width
        # Take the max height of the sub trees and add one for the current nodes height.
        height = max(l_node.height, r_node.height) + 1
        return VisualNode(get_name(tree_node), 1, 0.5, width, height, l_node, r_node)


def draw_tree_inner(node, parent=None, is_left=False):
    """Draw the inner tree."""
    if not node.left and not node.right:
        # Draw leaf node relative to the parents position.
        draw_node(node.name, (parent.get_position()[0] + node.x, parent.get_position()[1] + node.y), parent)
    elif parent:
        # Calculate the position of current node relative to parent.
        parent_x_offset = parent.get_position()[0]
        # Position dependent on whether current node is a left child or right child.
        x = parent_x_offset - (node.right.width/2) if is_left else parent_x_offset + (node.left.width/2)
        # Position the current inner node one layer below parent based upon heights of nodes.
        y = parent.get_position()[1] - 1
        # Draw current node and the children.
        an = draw_node(node.name, (x, y), parent)
        draw_tree_inner(node.left, an, True)
        draw_tree_inner(node.right, an, False)
    else:
        # Draw the root node.
        x = node.left.width/2
        y = max(node.left.height, node.right.height)
        an = draw_root_node(node.name, (x, y))
        draw_tree_inner(node.left, an, True)
        draw_tree_inner(node.right, an, False)


def draw_tree(tree, cur_depth, offset=0, left=1, parent=None):
    """Draws the entire tree."""
    # Calculate the width of the subtrees visual nodes.
    node_positions = calculate_subtree_widths(tree, True)
    # Calculate positions with visual nodes and plot on matlotlib.
    draw_tree_inner(node_positions)


def run():
    """Runs the script."""
    # Capture and parse user input.
    if len(sys.argv) != 2:
        print("usage: python3 -m decision_tree_visualisation path_to_dataset")
        sys.exit(0)

    # Generate the tree.
    print("Generating decision tree")
    dataset = decision_tree_analysis.import_dataset(sys.argv[1])
    tree, depth = decision_tree_analysis.decision_tree_learning(dataset)

    # Set the width and height of the image.
    global ax
    fwidth = int(45 * 1.5 ** (depth - 14))
    fheight = depth
    fig, ax = plt.subplots(figsize=(fwidth, fheight))
    ax.set_ylim([0, fheight])
    ax.set_xlim([0, fwidth])

    # Draw the tree.
    print("Generating visualisation")
    draw_tree(tree, 0)

    # Save the tree.
    plt.savefig("tree_visualisation", dpi=200)
    print("Saved visualisation to tree_visualisation.png")


if __name__ == '__main__':
    run()
