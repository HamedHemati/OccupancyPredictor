import argparse
import os
from sklearn.model_selection import RandomizedSearchCV

from room_occupancy.data_utils import load_dataset
from room_occupancy.model import get_model
from room_occupancy.train_utils import get_default_search_space
from room_occupancy.train_utils import calculate_metrics
from room_occupancy.plot_utils import plot_and_save_roc


def train(args):
    # Load train and test sets
    X_train, y_train = load_dataset(args.trainset_path)
    X_test, y_test = load_dataset(args.testset_path)

    # Initialize model (XGBoost)
    model = get_model(random_seed=args.seed)

    # Perform Random Search with cross-validation
    search_space = get_default_search_space()
    random_search = RandomizedSearchCV(
        model,
        param_distributions=search_space,
        n_iter=20,
        scoring="roc_auc",
        cv=3,
        verbose=True,
        random_state=args.seed,
        n_jobs=-1,  # Use all available processors
    )

    # Fit best hyparameters using RandomizedSearchCV
    random_search.fit(X_train, y_train)

    # Get the best estimator from the random search
    best_model = random_search.best_estimator_

    # Train the best model on the entire training set using the best parameters
    best_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_logits = best_model.predict_proba(X_test)[:, 1]
    y_pred = [1 if pred > 0.5 else 0 for pred in y_logits]

    # Calculate metrics
    metrics = calculate_metrics(y_test, y_logits, y_pred)

    # Plot ROC curve using calculated metrics
    plt_roc = plot_and_save_roc(y_test, y_logits, metrics)

    # Save model and plots
    os.makedirs(args.save_path, exist_ok=True)
    best_model.get_booster().save_model(os.path.join(args.save_path, "model.bin"))
    plt_roc.savefig(os.path.join(args.save_path, "roc_curve.png"))


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainset_path", type=str, default="./data/original/datatraining.txt")
    parser.add_argument("--testset_path", type=str, default="./data/original/datatest.txt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="./output")
    args = parser.parse_args()

    train(args)
