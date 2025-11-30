import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier, early_stopping
import scipy.stats as stats

# --- 1. Load and Initial Merge ---

# Load the datasets
train_df = pd.read_csv('/content/train.csv')
test_df = pd.read_csv('/content/test.csv')
meta_df = pd.read_csv('/content/metaData.csv')

# Merge meta_df with train_df
train_df = pd.merge(train_df, meta_df, on='lead_code', how='left')

# Merge meta_df with test_df
test_df = pd.merge(test_df, meta_df, on='lead_code', how='left')

print("DataFrames loaded and merged successfully.")

# --- 2. Create Basic Static Features ---

# Create 'log_total_due' feature
train_df['log_total_due'] = np.log1p(train_df['total_due'])
test_df['log_total_due'] = np.log1p(test_df['total_due'])

# Identify categorical columns for encoding
categorical_cols = ['suggested_action', 'dpd_bucket', 'state']

# Apply one-hot encoding to categorical columns
train_df_encoded = pd.get_dummies(train_df, columns=categorical_cols, drop_first=False)
test_df_encoded = pd.get_dummies(test_df, columns=categorical_cols, drop_first=False)

# Align columns to ensure consistency between train_df and test_df after encoding
all_columns = list(set(train_df_encoded.columns) | set(test_df_encoded.columns))
train_df_encoded = train_df_encoded.reindex(columns=all_columns, fill_value=0)
test_df_encoded = test_df_encoded.reindex(columns=all_columns, fill_value=0)

train_df_encoded = train_df_encoded[sorted(all_columns)]
test_df_encoded = test_df_encoded[sorted(all_columns)]

train_df = train_df_encoded
test_df = test_df_encoded

print("New feature 'log_total_due' created and categorical features one-hot encoded.")

# --- 3. Aggregate WhatsApp Data ---

whatsapp_df = pd.read_csv('/content/whatsapp_activity.csv')
whatsapp_df['sent_at'] = pd.to_datetime(whatsapp_df['sent_at'])

# Create binary indicators for message status
whatsapp_df['is_sent'] = whatsapp_df['status'].apply(lambda x: 1 if x in ['sent', 'DELIVERED', 'READ', 'REPLIED'] else 0)
whatsapp_df['is_delivered'] = whatsapp_df['status'].apply(lambda x: 1 if x in ['DELIVERED', 'READ', 'REPLIED'] else 0)
whatsapp_df['is_read'] = whatsapp_df['status'].apply(lambda x: 1 if x in ['READ', 'REPLIED'] else 0)
whatsapp_df['is_replied'] = whatsapp_df['status'].apply(lambda x: 1 if x == 'REPLIED' else 0)

# Group by 'lead_code' and calculate aggregated features
whatsapp_agg_df = whatsapp_df.groupby('lead_code').agg(
    whatsapp_total_messages=('lead_code', 'count'),
    whatsapp_sent_messages=('is_sent', 'sum'),
    whatsapp_delivered_messages=('is_delivered', 'sum'),
    whatsapp_read_messages=('is_read', 'sum'),
    whatsapp_replied_messages=('is_replied', 'sum'),
    whatsapp_last_message_date=('sent_at', 'max')
).reset_index()

# Calculate read and reply rates, handling division by zero
whatsapp_agg_df['whatsapp_read_rate'] = whatsapp_agg_df.apply(
    lambda row: row['whatsapp_read_messages'] / row['whatsapp_sent_messages'] if row['whatsapp_sent_messages'] > 0 else 0,
    axis=1
)
whatsapp_agg_df['whatsapp_reply_rate'] = whatsapp_agg_df.apply(
    lambda row: row['whatsapp_replied_messages'] / row['whatsapp_sent_messages'] if row['whatsapp_sent_messages'] > 0 else 0,
    axis=1
)

# Get the latest date from the entire dataset for 'days_since_last_message'
latest_date_whatsapp = whatsapp_df['sent_at'].max()
whatsapp_agg_df['whatsapp_days_since_last_message'] = (latest_date_whatsapp - whatsapp_agg_df['whatsapp_last_message_date']).dt.days

print("Whatsapp activity data loaded, preprocessed, and aggregated successfully.")

# --- 4. Aggregate Bot & SMS Data ---

ai_sms_df = pd.read_csv('/content/AI_sms_callback.csv')
ai_sms_df.columns = ai_sms_df.columns.str.strip()

# Group by 'lead_code' and calculate aggregated features
ai_sms_agg_df = ai_sms_df.groupby('lead_code').agg(
    ai_sms_total_activities=('lead_code', 'count'),
    ai_sms_delivered_count=('status', lambda x: (x == 'DELIVERED').sum())
).reset_index()

# Calculate delivery rate, handling division by zero
ai_sms_agg_df['ai_sms_delivered_rate'] = ai_sms_agg_df.apply(
    lambda row: row['ai_sms_delivered_count'] / row['ai_sms_total_activities'] if row['ai_sms_total_activities'] > 0 else 0,
    axis=1
)

print("Aggregated AI SMS data successfully.")

# --- 5. Aggregate Human Call Data ---

call_placed_df = pd.read_csv('/content/call_placed.csv')
telleco_callback_df = pd.read_csv('/content/teleco_call_back.csv')

# Convert 'start_time' to datetime objects
call_placed_df['start_time'] = pd.to_datetime(call_placed_df['start_time'])
telleco_callback_df['start_time'] = pd.to_datetime(telleco_callback_df['start_time'])

# Create 'is_answered_call_placed' for call_placed_df
call_placed_df['is_answered_call_placed'] = (call_placed_df['disposition'] == 'ANSWERED').astype(int)
# Ensure 'duration' is numeric for call_placed_df
call_placed_df['duration'] = pd.to_numeric(call_placed_df['duration'], errors='coerce').fillna(0)

# Create 'is_answered_callback' for telleco_callback_df
telleco_callback_df['is_answered_callback'] = (telleco_callback_df['disposition'] == 'ANSWERED').astype(int)
# Ensure 'duration' is numeric for telleco_callback_df
telleco_callback_df['duration'] = pd.to_numeric(telleco_callback_df['duration'], errors='coerce').fillna(0)

call_placed_agg_df = call_placed_df.groupby('lead_code').agg(
    call_placed_total_calls=('lead_code', 'count'),
    call_placed_answered_calls=('is_answered_call_placed', 'sum'),
    call_placed_avg_duration=('duration', 'mean'),
    call_placed_max_duration=('duration', 'max'),
    call_placed_last_call_date=('start_time', 'max')
).reset_index()

telleco_callback_agg_df = telleco_callback_df.groupby('lead_code').agg(
    teleco_total_callbacks=('lead_code', 'count'),
    teleco_answered_callbacks=('is_answered_callback', 'sum'),
    teleco_avg_duration=('duration', 'mean'),
    teleco_max_duration=('duration', 'max'),
    teleco_last_callback_date=('start_time', 'max')
).reset_index()

# Calculate answer rates, handling division by zero
call_placed_agg_df['call_placed_answer_rate'] = call_placed_agg_df.apply(
    lambda row: row['call_placed_answered_calls'] / row['call_placed_total_calls'] if row['call_placed_total_calls'] > 0 else 0,
    axis=1
)
telleco_callback_agg_df['teleco_callback_answer_rate'] = telleco_callback_agg_df.apply(
    lambda row: row['teleco_answered_callbacks'] / row['teleco_total_callbacks'] if row['teleco_total_callbacks'] > 0 else 0,
    axis=1
)

# Merge the aggregated DataFrames
human_calls_agg_df = pd.merge(
    call_placed_agg_df,
    telleco_callback_agg_df,
    on='lead_code',
    how='outer'
)

human_calls_agg_df['human_calls_last_interaction_date'] = human_calls_agg_df[['call_placed_last_call_date', 'teleco_last_callback_date']].max(axis=1)

overall_latest_human_call_date = pd.concat([call_placed_df['start_time'], telleco_callback_df['start_time']]).max()

human_calls_agg_df['human_calls_days_since_last_interaction'] = (overall_latest_human_call_date - human_calls_agg_df['human_calls_last_interaction_date']).dt.days

numeric_cols_to_fill_zero_human_calls = [
    'call_placed_total_calls', 'call_placed_answered_calls', 'call_placed_avg_duration', 'call_placed_max_duration', 'call_placed_answer_rate',
    'teleco_total_callbacks', 'teleco_answered_callbacks', 'teleco_avg_duration', 'teleco_max_duration', 'teleco_callback_answer_rate',
    'human_calls_days_since_last_interaction'
]

for col in numeric_cols_to_fill_zero_human_calls:
    if col in human_calls_agg_df.columns:
        human_calls_agg_df[col] = human_calls_agg_df[col].fillna(0)

print("Human call data aggregated and processed.")

# --- 6. Merge All Aggregated Features ---

temp_merged_df = pd.merge(whatsapp_agg_df, ai_sms_agg_df, on='lead_code', how='outer')
full_features = pd.merge(temp_merged_df, human_calls_agg_df, on='lead_code', how='outer')

print("Full features DataFrame created with merged aggregated data.")

# --- 7. Add Global Features & Handle Missing Values ---

numeric_cols_to_fill_zero_full = [
    'whatsapp_total_messages', 'whatsapp_sent_messages', 'whatsapp_delivered_messages', 'whatsapp_read_messages',
    'whatsapp_replied_messages', 'whatsapp_read_rate', 'whatsapp_reply_rate', 'whatsapp_days_since_last_message',
    'ai_sms_total_activities', 'ai_sms_delivered_count', 'ai_sms_delivered_rate',
    'call_placed_total_calls', 'call_placed_answered_calls', 'call_placed_avg_duration', 'call_placed_max_duration', 'call_placed_answer_rate',
    'teleco_total_callbacks', 'teleco_answered_callbacks', 'teleco_avg_duration', 'teleco_max_duration', 'teleco_callback_answer_rate',
    'human_calls_days_since_last_interaction'
]

for col in numeric_cols_to_fill_zero_full:
    if col in full_features.columns:
        full_features[col] = full_features[col].fillna(0)

# Calculate 'total_interactions'
full_features['total_interactions'] = \
    full_features['whatsapp_total_messages'] + \
    full_features['ai_sms_total_activities'] + \
    full_features['call_placed_total_calls'] + \
    full_features['teleco_total_callbacks']

# Calculate 'number_of_active_channels'
full_features['human_call_interactions'] = full_features['call_placed_total_calls'] + full_features['teleco_total_callbacks']
channel_cols = ['whatsapp_total_messages', 'ai_sms_total_activities', 'human_call_interactions']
full_features['number_of_active_channels'] = full_features[channel_cols].apply(
    lambda x: (x > 0).sum(), axis=1
)

# Create 'last_interaction_time'
full_features['last_interaction_time'] = full_features[['whatsapp_last_message_date', 'human_calls_last_interaction_date']].max(axis=1)

# Determine overall latest date (global_latest_date) from all raw activity dataframes
global_latest_date = pd.concat([
    whatsapp_df['sent_at'],
    call_placed_df['start_time'],
    teleco_callback_df['start_time']
]).max()

# Calculate the difference in days
full_features['days_since_last_interaction'] = (global_latest_date - full_features['last_interaction_time']).dt.days

# Fill any resulting NaN values in 'days_since_last_interaction' with 0
full_features['days_since_last_interaction'] = full_features['days_since_last_interaction'].fillna(0)

# Drop the individual date columns and intermediate human_call_interactions
date_cols_to_drop = [
    'whatsapp_last_message_date',
    'call_placed_last_call_date',
    'teleco_last_callback_date',
    'human_calls_last_interaction_date'
]
full_features = full_features.drop(columns=date_cols_to_drop, errors='ignore')
full_features = full_features.drop(columns=['human_call_interactions'], errors='ignore')

print("Global features created and missing values handled in 'full_features'.")

# --- 8. Final Feature Set Preparation ---

# Merge full_features with train_df and test_df
train_df = pd.merge(train_df, full_features, on='lead_code', how='left')
test_df = pd.merge(test_df, full_features, on='lead_code', how='left')

# Drop 'lead_code' column from both train_df and test_df
train_df = train_df.drop('lead_code', axis=1)
test_df = test_df.drop('lead_code', axis=1)

# Drop 'id' column from both train_df and test_df
train_df = train_df.drop('id', axis=1)
test_df = test_df.drop('id', axis=1)

# Separate the TARGET column from train_df into y_train and the remaining features into X_train.
y_train = train_df['TARGET']
X_train = train_df.drop('TARGET', axis=1)

# Create X_test by dropping the TARGET column from test_df.
X_test = test_df.drop('TARGET', axis=1)

# Identify common columns between X_train and X_test.
common_cols = list(set(X_train.columns) & set(X_test.columns))

# Reindex both X_train and X_test to keep only these common columns, ensuring they have the same features in the same order.
X_train = X_train[common_cols]
X_test = X_test[common_cols]

# Ensure columns are in the same order
X_train = X_train.reindex(columns=sorted(X_train.columns))
X_test = X_test.reindex(columns=sorted(X_test.columns))

# Drop any remaining non-numeric columns from X_train and X_test (e.g., date columns like last_interaction_time).
non_numeric_cols_train = X_train.select_dtypes(exclude=np.number).columns
non_numeric_cols_test = X_test.select_dtypes(exclude=np.number).columns

X_train = X_train.drop(columns=non_numeric_cols_train, errors='ignore')
X_test = X_test.drop(columns=non_numeric_cols_test, errors='ignore')

print("Target variable separated and feature sets prepared with consistent columns.")

# --- 9. Train LightGBM Model with Stratified K-Fold ---

N_SPLITS = 5
RANDOM_STATE = 42
TARGET_THRESHOLD = 0.5

y_train_binned = pd.qcut(y_train, q=N_SPLITS, labels=False, duplicates='drop')
label_encoder = LabelEncoder()
y_train_binned_encoded = label_encoder.fit_transform(y_train_binned)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

oof_preds = np.zeros(X_train.shape[0])
test_preds = np.zeros(X_test.shape[0])

feature_importances = pd.DataFrame(index=X_train.columns)

print(f"Starting {N_SPLITS}-fold Stratified K-Fold cross-validation...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train_binned_encoded)):
    print(f"\n--- Fold {fold + 1}/{N_SPLITS} ---")

    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

    y_train_binary = (y_train_fold > TARGET_THRESHOLD).astype(int)
    y_val_binary = (y_val_fold > TARGET_THRESHOLD).astype(int)

    lgbm = LGBMClassifier(
        objective='binary',
        metric='auc',
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        reg_alpha=0.1,
        reg_lambda=0.1,
        verbose=-1
    )

    lgbm.fit(X_train_fold, y_train_binary,
             eval_set=[(X_val_fold, y_val_binary)],
             eval_metric='auc',
             callbacks=[early_stopping(100, verbose=False)])

    oof_preds[val_idx] = lgbm.predict_proba(X_val_fold)[:, 1]

    test_preds += lgbm.predict_proba(X_test)[:, 1] / N_SPLITS

    feature_importances[f'Fold_{fold+1}'] = lgbm.feature_importances_

print("\nStratified K-Fold cross-validation completed.")


# --- 10. Prepare for Hyperparameter Tuning ---

param_dist = {
    'n_estimators': stats.randint(100, 1001),
    'learning_rate': stats.uniform(0.01, 0.19),
    'num_leaves': stats.randint(20, 61),
    'max_depth': stats.randint(3, 16),
    'reg_alpha': stats.uniform(0.0, 0.5),
    'reg_lambda': stats.uniform(0.0, 0.5),
    'subsample': stats.uniform(0.6, 0.4),
    'colsample_bytree': stats.uniform(0.6, 0.4)
}

scoring_metric = 'roc_auc'

print("Hyperparameter search space and scoring metric defined.")

# --- 11. Implement RandomizedSearchCV ---

y_train_binary = (y_train > TARGET_THRESHOLD).astype(int)

lgbm_base = LGBMClassifier(
    objective='binary',
    metric='auc',
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1
)

random_search = RandomizedSearchCV(
    estimator=lgbm_base,
    param_distributions=param_dist,
    n_iter=50,
    scoring=scoring_metric,
    cv=skf,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1
)

print("Starting RandomizedSearchCV...")
random_search.fit(X_train, y_train_binary)
print("RandomizedSearchCV completed.")

print("\nBest parameters found:")
print(random_search.best_params_)
print("\nBest AUC score:")
print(random_search.best_score_)

# --- 12. Train Model with Best Parameters ---

best_params = random_search.best_params_

final_lgbm_model = LGBMClassifier(
    objective='binary',
    metric='auc',
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
    **best_params
)

print("Training final LightGBM model with best parameters...")
final_lgbm_model.fit(X_train, y_train_binary)
print("Final LightGBM model trained successfully.")

# --- 13. Evaluate Tuned Model and Generate Submission ---

test_predictions = final_lgbm_model.predict_proba(X_test)[:, 1]
clipped_predictions = np.clip(test_predictions, 0, 1)

original_test_df = pd.read_csv('/content/test.csv')

submission_df = pd.DataFrame({
    'id': original_test_df['id'],
    'TARGET': clipped_predictions
})

submission_df.to_csv('submission.csv', index=False)

print("Predicted probabilities generated and clipped.")
print("Submission file 'submission.csv' created successfully.")