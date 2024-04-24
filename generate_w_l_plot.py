import pandas as pd
import matplotlib.pyplot as plt

rob_data = pd.read_csv("results/ai_vs_rob_100.csv")
jake_data = pd.read_csv("results/ai_vs_jake_100.csv")

# Assume the column of interest is named 'DataColumn' in both CSVs
# If it's an index, use data1.iloc[:, column_index] instead
plt.figure(figsize=(10, 5))  # Set the size of the plot
rob_cumm_data = rob_data["ai_winnings"].cumsum()
jake_cumm_data = jake_data["ai_winnings"].cumsum()

# Plotting the data
plt.plot(rob_cumm_data, label="Against Rob")
plt.plot(jake_cumm_data, label="Against Jake")


# Adding titles and labels
plt.title("AI Winnings vs. Human Opponents")
plt.xlabel("Hand #")
plt.ylabel("Profit (chips)")

# Adding legend
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5, color="gray", alpha=0.5)

# Show the plot
plt.savefig("results/ai_vs_human_100.png")
