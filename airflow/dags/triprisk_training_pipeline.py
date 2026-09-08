from datetime import timedelta

from airflow.sdk import dag, task

from scripts.airflow_training import (
    prepare_dataset,
    register_champion,
    select_best,
    train_candidate,
)

DATA_PATH = "/opt/airflow/data/triprisk_dataset.npz"


@dag(
    dag_id="triprisk_training_pipeline",
    schedule=None,
    catchup=False,
    tags=[
        "ml",
        "training",
        "mlflow",
    ],
)
def triprisk_training_pipeline():
    @task
    def prepare_data() -> str:
        return prepare_dataset(DATA_PATH)

    @task(
        retries=1,
        retry_delay=timedelta(seconds=15),
    )
    def train_logistic(
        data_path: str,
    ) -> dict[str, object]:
        return train_candidate(
            data_path,
            "logistic-regression",
        )

    @task(
        retries=1,
        retry_delay=timedelta(seconds=15),
    )
    def train_random_forest(
        data_path: str,
    ) -> dict[str, object]:
        return train_candidate(
            data_path,
            "random-forest",
        )

    @task(
        retries=1,
        retry_delay=timedelta(seconds=15),
    )
    def train_hist_gradient_boosting(
        data_path: str,
    ) -> dict[str, object]:
        return train_candidate(
            data_path,
            "hist-gradient-boosting",
        )

    @task
    def choose_best(
        logistic: dict[str, object],
        random_forest: dict[str, object],
        hist_gradient_boosting: (dict[str, object]),
    ) -> dict[str, object]:
        return select_best(
            [
                logistic,
                random_forest,
                hist_gradient_boosting,
            ]
        )

    @task
    def register_best(
        best: dict[str, object],
    ) -> dict[str, str]:
        return register_champion(best)

    dataset_path = prepare_data()

    logistic = train_logistic(dataset_path)

    random_forest = train_random_forest(dataset_path)

    hist_gradient_boosting = train_hist_gradient_boosting(dataset_path)

    best = choose_best(
        logistic,
        random_forest,
        hist_gradient_boosting,
    )

    register_best(best)


triprisk_training_pipeline()
