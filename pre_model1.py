import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from scipy import stats
from tensorflow.keras.callbacks import TensorBoard
import datetime

def run_model1(year_num, txt_name, pth_name):
    # Read data
    file_path = 'all_data_final.xlsx'
    data = pd.read_excel(file_path)

    # Convert four-class classification to binary classification
    data['Survival months_new_classified'] = data['Survival months_new_classified'].apply(lambda x: 1 if x >= year_num else 0)

    # Split the dataset
    X = data.drop(columns=['Survival months_new_classified'])
    y = data['Survival months_new_classified']

    # Set up cross-validation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Create TensorBoard log directory
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Store results for each validation
    auc_scores = []
    accuracy_scores = []

    # Store feature weights after each validation
    feature_weights = []

    # Open file for writing
    with open(txt_name, 'w') as f:
        # Redirect print function output to file
        def print_to_file(*args, **kwargs):
            print(*args, **kwargs, file=f)
            print(*args, **kwargs)  # Optional: Also print to console

        for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            # Compute class weights
            class_weights = compute_class_weight(class_weight='balanced', classes=[0, 1], y=y_train)
            class_weights_dict = {0: class_weights[0], 1: class_weights[1]}

            # Adjusted model structure, increasing complexity
            model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],),
                                      kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.35),
                tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.35),
                tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.35),
                tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])

            # Learning rate warmup
            initial_learning_rate = 0.00005
            warmup_steps = 10
            warmup_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                boundaries=[warmup_steps],
                values=[initial_learning_rate / warmup_steps, initial_learning_rate]
            )

            # Cosine annealing
            cosine_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=initial_learning_rate,
                first_decay_steps=300,
                t_mul=1.0,
                m_mul=1.0,
                alpha=0.000001
            )

            # Combined learning rate scheduler
            def lr_schedule(epoch):
                if epoch < warmup_steps:
                    return warmup_schedule(epoch)
                else:
                    return cosine_schedule(epoch - warmup_steps)

            lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

            # Compile the model
            optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

            # Set callbacks to save the best model
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                pth_name + 'best_model_fold_{}.h5'.format(fold + 1), monitor='val_loss', save_best_only=True, mode='min'
            )
            early_stopping_callback = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=60, restore_best_weights=True
            )

            # Train the model, extending training time
            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=1500,
                batch_size=32,
                class_weight=class_weights_dict,
                callbacks=[tensorboard_callback, checkpoint_callback, early_stopping_callback, lr_callback],
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
            feature_weights.append(layer_weights)

            # Output feature weights for each fold
            print_to_file(f"\nFold {fold + 1} Feature Weights:")
            for name, weight in zip(X.columns, layer_weights):
                print_to_file(f"{name}: {np.mean(weight):.4f}")

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

        # Output detailed metrics for each cross-validation
        print_to_file("\nCross-Validation Detailed Metrics:")
        print_to_file(f"{'Fold':<6} {'AUC':<8} {'C-index':<10} {'F1-score':<10} {'Accuracy':<10} {'Recall':<10} {'Precision':<12} {'Specificity':<12} {'Sensitivity':<12}")
        for i in range(len(auc_scores)):
            print_to_file(
                f"{i + 1:<6} {auc_scores[i]:<8.4f}  {accuracy_scores[i]:<10.4f}")
