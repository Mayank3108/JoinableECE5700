import matplotlib.pyplot as plt

# Data
k_values = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
top_k_accuracies = [0.7960, 0.8307, 0.8515, 0.8619, 0.8764,
                    0.9071, 0.9304, 0.9403, 0.9460, 0.9486,
                    0.9517, 0.9522, 0.9533, 0.9548, 0.9548]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(k_values, top_k_accuracies, marker='o')
plt.title("Top-k Accuracy on Test Set")
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()
plt.savefig("top_k_accuracy_plot.png") 
plt.show()
