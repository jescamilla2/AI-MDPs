# Import your Node class and initialization function
from node import Node, initialize_nodes


def run_episode(start_node, nodes):
    """
    Simulate a single episode starting from the given node.
    :param start_node: The starting Node object.
    :param nodes: Dictionary of all nodes.
    :return: Sequence of experiences [(node_id, action, reward)], total reward.
    """
    current_node = start_node
    experiences = []
    total_reward = 0

    while current_node.id != 10:  # Assuming Node 10 is the terminal node
        # Select an action
        action = current_node.select_action()

        # Determine the next node
        next_node_id = current_node.get_next_node(action)

        # Get reward of selecting action
        reward = current_node.rewards[(current_node.id, action, next_node_id)]

        total_reward += reward

        # Record the experience (store node ID)
        experiences.append((current_node.id, action, reward))

        # Move to the next node
        current_node = nodes[next_node_id]

    return experiences, total_reward


def monte_carlo_update(visited_nodes, total_reward, nodes, alpha=0.1):
    """
    Perform first-visit Monte Carlo updates on visited nodes.
    :param visited_nodes: List of visited node IDs in the episode.
    :param total_reward: The total reward obtained in the episode.
    :param nodes: Dictionary of all nodes.
    :param alpha: Learning rate for value updates.
    """
    visited_set = set()  # To track first-visit states
    for node_id in visited_nodes:
        if node_id not in visited_set:
            visited_set.add(node_id)
            node = nodes[node_id]
            # Update node value using first-visit MC formula
            node.update_value(node.value + alpha * (total_reward - node.value))


def run_experiment(start_node, nodes, episodes=50, alpha=0.1):
    """
    Run the MDP simulation for a specified number of episodes.
    :param start_node: The starting Node object.
    :param nodes: Dictionary of all nodes.
    :param episodes: Number of episodes to simulate.
    :param alpha: Learning rate for Monte Carlo updates.
    """
    total_rewards = []
    for episode in range(episodes):
        # Simulate one episode
        experiences, total_reward = run_episode(start_node, nodes)
        total_rewards.append(total_reward)

        # Extract visited node IDs for first-visit Monte Carlo
        visited_nodes = [node_id for node_id, _, _ in experiences]

        # Perform Monte Carlo update
        monte_carlo_update(visited_nodes, total_reward, nodes, alpha)

        # Print episode details
        print(f"Episode {episode + 1}:")
        for node_id, action, reward in experiences:
            print(f"  Node ({nodes[node_id].state[0]}{nodes[node_id].state[1]} {nodes[node_id].state[2]:>3s}), Action: {action}, Reward: {reward:>2d}")
        print(f"  Total Reward: {total_reward}")
        print("-" * 40)

    # Print final state values and average episode reward
    print("\nFinal Node Values:")
    for node in nodes.values():
        print(f"Node ({node.state[0]}{node.state[1]} {node.state[2]:>3s}) Value: {node.value: .4f}")

    average_reward = sum(total_rewards) / episodes
    print(f"\nAverage Reward per Episode: {average_reward}")


# run through
if __name__ == "__main__":
    nodes = initialize_nodes()  # Make sure this initializes Node objects correctly
    run_experiment(nodes[0], nodes)
