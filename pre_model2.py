import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import OneHotEncoder
from scipy import stats

def run_model2(year_num, txt_name, pth_name):
    # Load data
    file_path = 'all_data_final.xlsx'
    data = pd.read_excel(file_path)

    # Delete rows where 'Primary Site - labeled_new_classified' column values are specified values
    #data = data[~data['Primary Site - labeled_new_classified'].isin([13,14,16,17,18,21,22,23,25,26,27,30,31,33,34,36,37,39,40,43,44,45,47])]

    # Convert four-class classification to binary classification
    data['Survival months_new_classified'] = data['Survival months_new_classified'].apply(lambda x: 1 if x >= year_num else 0)

    # One-hot encode 'Primary Site - labeled_new_classified' column
    encoder = OneHotEncoder(sparse=False)
    encoded_features = encoder.fit_transform(data[['Primary Site - labeled_new_classified']])
    encoded_feature_names = encoder.get_feature_names_out(['Primary Site - labeled_new_classified'])

    # Add one-hot encoded results to the original data
    encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)
    data = pd.concat([data, encoded_df], axis=1).drop(columns=['Primary Site - labeled_new_classified'])

    # Drop rows with NaN values
    data = data.dropna()

    # Check the number of rows in data
    row_count = data.shape[0]
    print(f"Data has {row_count} rows.")

    # Split the dataset
    X = data.drop(columns=['Survival months_new_classified'])
    y = data['Survival months_new_classified']

    # Set up cross-validation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # To store the results of each validation
    auc_scores = []
    accuracy_scores = []

    # To store feature weights after each validation
    primary_site_weights_all_folds = []

    # Open file to write results
    with open(txt_name, 'w') as f:
        def print_to_file(*args, **kwargs):
            print(*args, **kwargs, file=f)
            print(*args, **kwargs)  # Optional: also print to console

        for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            # Calculate sample weights
            class_weights = compute_class_weight(class_weight='balanced', classes=[0, 1], y=y_train)
            class_weights_dict = {0: class_weights[0], 1: class_weights[1]}

            # Model definition, adding L2 regularization
            model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])

            # Learning rate scheduler
            initial_learning_rate = 0.001
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate,
                decay_steps=10000,
                decay_rate=0.9,
                staircase=True
            )

            # Compile the model
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

            # Set up callbacks, save the best model
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                pth_name + 'best_model_fold_{}.h5'.format(fold + 1), monitor='val_loss', save_best_only=True, mode='min'
            )
            early_stopping_callback = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=20, restore_best_weights=True
            )

            # Train the model
            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=1000,
                batch_size=32,
                class_weight=class_weights_dict,
                callbacks=[checkpoint_callback, early_stopping_callback],
                verbose=2
            )

            # Prediction and evaluation
            y_val_pred = model.predict(X_val).ravel()
            y_val_pred_binary = (y_val_pred > 0.5).astype(int)

            auc = roc_auc_score(y_val, y_val_pred)
            accuracy = accuracy_score(y_val, y_val_pred_binary)

            auc_scores.append(auc)
            accuracy_scores.append(accuracy)

            # Extract model weights
            layer_weights = model.layers[0].get_weights()[0]
            primary_site_weights = [weight for name, weight in zip(X.columns, layer_weights) if 'Primary Site - labeled_new_classified' in name]
            primary_site_weights_all_folds.append(primary_site_weights)

            # Output 'Primary Site - labeled_new_classified' feature weights in each fold
            print_to_file(f"\nFold {fold + 1} 'Primary Site - labeled_new_classified' Feature Weights:")
            for name, weight in zip(encoded_feature_names, primary_site_weights):
                print_to_file(f"{name}: {np.mean(weight):.4f}")

        # Calculate average weights across all folds
        primary_site_average_weights = np.mean(np.array(primary_site_weights_all_folds), axis=0)

        # Output average weights
        print_to_file("\n'Primary Site - labeled_new_classified' Average Feature Weights:")
        for name, weight in zip(encoded_feature_names, primary_site_average_weights):
            # Output the average weight
            average_weight = np.mean(weight)
            print_to_file(f"{name}: {average_weight:.4f}")

        # Calculate mean and 95% confidence interval
        def mean_ci(scores):
            mean = np.mean(scores)
            ci = stats.norm.interval(0.95, loc=mean, scale=stats.sem(scores))
            return mean, ci

        metrics = {
            "AUC": auc_scores,
            "Accuracy": accuracy_scores
        }

        for metric_name, scores in metrics.items():
            mean, ci = mean_ci(scores)
            print_to_file(f"{metric_name}: Mean = {mean:.4f}, 95% CI = [{ci[0]:.4f}, {ci[1]:.4f}]")

        # Output detailed metrics for all folds
        print_to_file("\nCross-Validation Detailed Metrics:")
        print_to_file(f"{'Fold':<6} {'AUC':<8} {'C-index':<10} {'F1-score':<10} {'Accuracy':<10} {'Recall':<10} {'Precision':<12} {'Specificity':<12} {'Sensitivity':<12}")
        for i in range(len(auc_scores)):
            print_to_file(f"{i+1:<6} {auc_scores[i]:<8.4f} {accuracy_scores[i]:<10.4f}")
