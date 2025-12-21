import pandas as pd

# Load the CSV
df = pd.read_csv("trauma_inference_log.csv")  # Update path if needed

# Parse label strings into dictionaries
def parse_labels(label_str):
    if pd.isna(label_str):
        return {}
    parts = label_str.split(", ")
    parsed = {}
    for part in parts:
        if ": " in part:
            key, value = part.split(": ", 1)
            parsed[key.strip()] = value.strip()
    return parsed

df["Predicted"] = df["Predicted Label"].apply(parse_labels)
df["GroundTruth"] = df["Ground Truth"].apply(parse_labels)

# Define regions
regions = ["Head", "Torso", "Upper Extremities", "Lower Extremities"]

# Initialize counters
correct_counts = {region: 0 for region in regions}
total_counts = {region: 0 for region in regions}
total_correct = 0
total_labels = len(df) * len(regions)

# Initialize score counters
score_per_region = {region: 0 for region in regions}
total_score = 0

# Compare each region in prediction vs ground truth
for _, row in df.iterrows():
    for region in regions:
        pred = row["Predicted"].get(region)
        truth = row["GroundTruth"].get(region)

        if pred == truth:
            correct_counts[region] += 1
            total_correct += 1
            score_per_region[region] += 1   # +1 for correct
            total_score += 1
        else:
            score_per_region[region] -= 1   # -1 for wrong
            total_score -= 1

        total_counts[region] += 1

# Compute accuracy per region
accuracy_per_region = {
    region: correct_counts[region] / total_counts[region]
    for region in regions
}

# Compute overall accuracy
overall_accuracy = total_correct / total_labels

# Print results
print("Accuracy per Region:")
for region, acc in accuracy_per_region.items():
    print(f"{region}: {acc:.2%}")

print(f"\nOverall Accuracy: {overall_accuracy:.2%}")

# Print score results
print("\nScore per Region (raw):")
for region, score in score_per_region.items():
    print(f"{region}: {score}")

print(f"\nTotal Score (raw): {total_score}")

# Normalized scores (divide by total sentences)
print("\nScore per Region (per sentence):")
for region, score in score_per_region.items():
    print(f"{region}: {score / len(df):.2f}")

print(f"\nTotal Score (per sentence): {total_score / len(df):.2f}")

# Optional: Exact Match Accuracy
if "Exact Match" in df.columns:
    exact_match_accuracy = df["Exact Match"].sum() / len(df)
    print(f"\nExact Match Accuracy: {exact_match_accuracy:.2%}")
