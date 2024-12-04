# Import your Node class and initialization function
from src.node import initialize_nodes


def test_node():
    """
    Main function to test node initialization and basic operations for all nodes.
    """
    # Initialize the nodes
    nodes = initialize_nodes()

    # Iterate through all nodes
    for node_id, node in nodes.items():
        print(f"Node {node_id} Details:")
        print(f"  State: {node.state}")
        print(f"  Transitions: {node.transitions}")
        print(f"  Rewards: {node.rewards}")

        if not node.transitions:  # Check if the node is terminal
            print(f"  Node {node_id} is terminal. No actions available.")
        else:
            # Test: Randomly select an action from the current node
            action = node.select_action()
            print(f"  Randomly selected action: {action}")

            # Test: Get the next node based on the selected action
            next_node_id = node.get_next_node(action)
            print(f"  Next node on action {action}: {next_node_id}")
            print("-" * 40)  # Separator for readability

if __name__ == "__main__":
    test_node()
