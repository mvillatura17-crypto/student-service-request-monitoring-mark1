"""Generate a SYNTHETIC dataset that follows the midterm schema.

Use this only for demos and testing. Replace it with your own authorized
midterm CSV and re-run train_model.py to reproduce your midterm results.
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 400

SERVICES = {
    # service: (office, SLA days, probability, breach effect)
    "Academic Records":    ("Registrar", 5, 0.26, 0.0),
    "Certification":       ("Registrar", 3, 0.15, 0.7),
    "Scholarship Inquiry": ("Student Affairs", 7, 0.17, -0.2),
    "Financial Services":  ("Finance Office", 5, 0.20, 0.1),
    "IT Support":          ("IT Helpdesk", 2, 0.22, -0.3),
}
names = list(SERVICES)
service = rng.choice(names, N, p=[SERVICES[n][2] for n in names])
office = [SERVICES[s][0] for s in service]
sla = [SERVICES[s][1] for s in service]
svc_effect = np.array([SERVICES[s][3] for s in service])

requester = rng.choice(["Student", "Faculty", "Staff"], N, p=[0.75, 0.12, 0.13])
channel = rng.choice(["Portal", "Email", "Walk-In"], N, p=[0.55, 0.25, 0.20])
priority = rng.choice(["High", "Normal", "Low"], N, p=[0.25, 0.50, 0.25])
queue = np.clip(rng.poisson(9, N), 0, 30)
queue[rng.random(N) < 0.18] = 0
docs = rng.choice(["Yes", "No"], N, p=[0.82, 0.18])
status = rng.choice(["Completed", "Pending", "In Progress", "Cancelled"], N,
                    p=[0.76, 0.09, 0.09, 0.06])
reopened = rng.choice(["No", "Yes"], N, p=[0.88, 0.12])
dates = pd.to_datetime("2026-01-01") + pd.to_timedelta(rng.integers(0, 181, N), unit="D")

logit = (-0.1 + 0.22 * (queue - 8) + 1.6 * (docs == "No") + svc_effect
         + 0.35 * (priority == "Low") - 0.25 * (priority == "High")
         + rng.normal(0, 0.4, N))
breach = rng.random(N) < 1 / (1 + np.exp(-logit))

processing = np.full(N, np.nan)
satisfaction = np.full(N, np.nan)
for i in range(N):
    if status[i] != "Completed":
        continue
    if breach[i]:
        processing[i] = sla[i] + rng.integers(1, 5)
        satisfaction[i] = rng.choice([1, 2, 3], p=[0.2, 0.4, 0.4])
    else:
        processing[i] = max(1, sla[i] - rng.integers(0, 3))
        satisfaction[i] = rng.choice([3, 4, 5], p=[0.2, 0.4, 0.4])

df = pd.DataFrame({
    "Request_ID": [f"REQ-{1001 + i}" for i in range(N)],
    "Service_Type": service,
    "Requester_Type": requester,
    "Submission_Date": dates.strftime("%Y-%m-%d"),
    "Requested_Channel": channel,
    "Priority": priority,
    "Assigned_Office": office,
    "Queue_Length": queue,
    "Required_Documents_Complete": docs,
    "Processing_Days": processing,
    "SLA_Days": sla,
    "Status": status,
    "Reopened": reopened,
    "Satisfaction_Score": satisfaction,
})

# Add a little messy text so the cleaning step has something to do
for col in ["Service_Type", "Assigned_Office", "Requested_Channel"]:
    idx = rng.choice(N, 40, replace=False)
    df.loc[idx, col] = df.loc[idx, col].str.lower()
idx = rng.choice(N, 15, replace=False)
df.loc[idx, "Priority"] = " " + df.loc[idx, "Priority"] + " "

out = Path(__file__).resolve().parent.parent / "data" / "sample_requests.csv"
df.to_csv(out, index=False)
print("Saved", out, df.shape)
