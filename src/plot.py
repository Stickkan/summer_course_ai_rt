import pandas as pd
import matplotlib.pyplot as plt


def plot(data: str) -> None:
    # Load the CSV file without headers
    df = pd.read_csv(data, header=None)
    
    # Plot Input data and Chosen output state vs Current time
    plt.figure(figsize=(10, 6))
    # plt.plot(df[0], df[2], df[3], label="Input data")
    plt.plot(df[0], df[18], label="Chosen output state")

    plt.xlabel("Current time")
    plt.ylabel("Value")
    plt.title("Input Data and Chosen Output State vs Current Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
