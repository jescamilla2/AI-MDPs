from src.monte_carlo import *
from src.mdp import initialize_nodes

# Main method to run Monte Carlo
if __name__ == "__main__":
    """
    Main method to set up the MDP, run Monte Carlo, and display results.
    """
    # Step 1: Initialize nodes and their transitions/rewards
    nodes = initialize_nodes()

    # Step 2: Run Monte Carlo
    print("Starting Monte Carlo...\n")
    run_experiment(nodes[0], nodes)

    # Step 3: Display final results
    print("\nFinal Results Monte Carlo:")
    print("-" * 50)
    for node in nodes.values():
        print(f"Node ({node.state[0]}{node.state[1]} {node.state[2]:>3s}) Value: {node.value: .4f}")
    print("-" * 50)

