import pandas as pd
import numpy as np


label = pd.read_csv("/data2/datasets/kaggle/diabetic-retinopathy-detection-2015/label/similar_to_2019.csv")
print(label.head())


counts = label.groupby("label").count()["image"].values
print(counts)


desire_prob = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
prob_of_each_cls = desire_prob / counts
print(prob_of_each_cls)

for i in range(5):
    label.loc[label.label == i, "prob"] = prob_of_each_cls[i]

print(label)
print(f"sum up to: {label.prob.sum()}")

print(np.bincount(np.random.choice(label.label, size=10000, p=label.prob)))

label.to_csv(
    "/data2/datasets/kaggle/diabetic-retinopathy-detection-2015/label/similar_to_2019_with_prob.csv", index=False
)
