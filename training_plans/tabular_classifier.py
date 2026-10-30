"""Fed-BioMed training plan for tabular classification."""

import numpy as np
from declearn.optimizer.modules import ScaffoldClientModule
from declearn.optimizer.regularizers import FedProxRegularizer
from fedbiomed.common.datamanager import DataManager
from fedbiomed.common.dataset import TabularDataset
from fedbiomed.common.optimizers.optimizer import Optimizer
from fedbiomed.common.training_plans import FedSGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score


class TabularClassifierTrainingPlan(FedSGDClassifier):
    """Predict an integer-encoded class column from standardized feature columns."""

    def init_dependencies(self):
        return [
            "import numpy as np",
            "from fedbiomed.common.datamanager import DataManager",
            "from fedbiomed.common.dataset import TabularDataset",
            "from fedbiomed.common.optimizers.optimizer import Optimizer",
            "from declearn.optimizer.modules import ScaffoldClientModule",
            "from declearn.optimizer.regularizers import FedProxRegularizer",
            "from sklearn.metrics import accuracy_score, balanced_accuracy_score",
            "from sklearn.metrics import roc_auc_score",
        ]

    def init_optimizer(self, optimizer_args):
        if "lr" not in optimizer_args:
            # use the default scikit-learn internal optimizer
            return None

        modules = [ScaffoldClientModule()] if optimizer_args.get("scaffold") else []
        regularizers = (
            [FedProxRegularizer(alpha=optimizer_args["fedprox_mu"])]
            if optimizer_args.get("fedprox_mu")
            else []
        )
        return Optimizer(
            lr=optimizer_args["lr"], modules=modules, regularizers=regularizers
        )

    def set_data_loaders(self, train_data_loader, test_data_loader):
        if train_data_loader is None:
            # no training data when the node only tests the model
            # used in held-out node validation
            self.training_data_loader = None
            self.testing_data_loader = test_data_loader
        else:
            super().set_data_loaders(train_data_loader, test_data_loader)

    def training_data(self):
        model_args = self.model_args()
        columns = list(model_args["feature_columns"])
        mean = np.asarray(model_args["feature_mean"], dtype=float)
        std = np.asarray(model_args["feature_std"], dtype=float)
        if not len(columns) == len(mean) == len(std) == model_args["n_features"]:
            raise ValueError(
                "feature_columns, feature_mean and feature_std must all have"
                f" length n_features={model_args['n_features']}"
            )

        dataset = TabularDataset(
            input_columns=columns,
            target_columns=[model_args["target_column"]],
            transform=lambda x: (x - mean) / std,
        )
        return DataManager(dataset=dataset, shuffle=True)

    def testing_step(self, data, target):
        target = np.ravel(target)
        prediction = self.model().predict(data)
        if len(np.unique(target)) < 2:
            # AUROC is undefined if the split has a single class
            auroc = float("nan")
        else:
            auroc = roc_auc_score(target, self.model().decision_function(data))
        return {
            "ACCURACY": accuracy_score(target, prediction),
            "BALANCED_ACCURACY": balanced_accuracy_score(target, prediction),
            "AUROC": auroc,
        }
