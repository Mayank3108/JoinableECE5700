import matplotlib.pyplot as plt
labels = ['Accuracy', 'IoU', 'Top-1', 'Top-1 (Holes)', 'Top-1 (No Holes)']
values = [0.8920, 0.1157, 0.7960, 0.8015, 0.7695]

plt.figure(figsize=(10, 6))
bars = plt.bar(labels, values, color='skyblue')
plt.title("Final Test Set Metrics")
plt.ylabel("Value")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("test_metrics_barplot.png")
plt.show()
