# Credresolve_Prob_Predictor
This project develops a LightGBM classification model to predict a continuous target variable, adapted for a binary classification task. The solution leverages a comprehensive Customer-360 approach by integrating various customer interaction data points

📌 Overview

This project builds a machine learning system to predict the probability that a debt-recovery action (WhatsApp, AI Voice Bot, Human Call, Field Visit) will successfully result in payment.
Each test row contains a lead_code (borrower) and a suggested_action.
Your model must output TARGET = P(success | borrower, action).

Kaggle evaluates submissions by computing ROI using action cost vs predicted success probability.

📂 Dataset Components

The competition provides:

Core Files

train.csv – lead_code, suggested_action, TARGET

test.csv – same structure without TARGET

metaData.csv – static borrower info (total_due, dpd_bucket, state)

sample_submission.csv

Interaction Logs

Used to build a Customer-360 profile:

whatsapp_activity.csv

teleco_call_back.csv (AI bot calls)

call_placed.csv (human calls)

mobile_app_data.csv (field visits)

AI_sms_callback.csv (SMS)

All logs keyed using lead_code.

🧠 Approach Summary
1. Base Merging

Merge metaData with train/test

Create simple features:

log_total_due

Label encoding of suggested_action, dpd_bucket, state

2. Build Customer-360 Features

Aggregate each log at lead_code level:

WhatsApp:
count sent/read/delivered, reply count, read/reply rate, days since last message

AI Bot:
call counts, answer rate, avg/max duration, last call time, intent & sentiment extracted from transcript_json

Human Calls:
call counts, answer rate, durations, days since last human call

Field Visits:
visit outcomes (MET, DOOR_LOCKED, SHIFTED), mean lat/long, last visit date, keyword flags from remarks

SMS:
delivery flag

Add global metrics:

total interactions

number of active channels

days since last interaction

Merge all into a single full_features table.

3. Model Training

Merge full_features into train/test

Select all numeric/categorical engineered features

Use LightGBM with 5-fold Stratified KFold

Train → predict OOF + test

Average fold predictions

Clip probabilities to [0,1]

4. Submission

Create a CSV with:

id, TARGET


and upload to Kaggle.

🚀 Result

This pipeline creates a robust, behavior-driven predictive model that identifies the ROI-optimal next action for debt recovery using a Customer-360 ML approach
