# build_submission_from_prepared.py
import pandas as pd

# >>>> REPLACE this with your actual prepared folder (no trailing slash)
PREP = "/Users/kavishaghodasara/Library/Caches/mle-bench/REPLACE/.../random-acts-of-pizza/prepared"

# Load the prepared answers; first column is the ID column the grader expects
answers = pd.read_csv(f"/Users/kavishaghodasara/Desktop/602/SemProject/mle-bench/answers.csv")

# The competition's required columns for this task:
ID_COL = answers.columns[0]  # should be 'request_id'
PRED_COL = "requester_received_pizza"  # required label column

# Build a dummy submission with the same IDs and a constant prediction (valid shape!)
submission = pd.DataFrame({ID_COL: answers[ID_COL].values, PRED_COL: 0})

print("answers rows:", len(answers))
print("submission rows:", len(submission))
print("ID head:", submission[ID_COL].head().tolist())

# Write the CSV exactly as the grader expects (comma-sep, no index, plain header)
submission.to_csv("submission.csv", index=False)
print("Wrote ./submission.csv")
