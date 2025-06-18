from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import os
import asyncio

target_names = {'Black or African American': 0,
 'East/Southeast Asian': 1,
 'Hispanic, Latino, or Spanish': 2,
 'White': 3
 }



class ModelEvaluator:
    def __init__(self, models, results_dir='./results/'):
        self.models = models
        self.results_dir = results_dir
        os.makedirs(os.path.dirname(self.results_dir), exist_ok=True)

    def get_classification_report(self, true, pred):
        return classification_report(
            true,
            pred,
            digits=4,
            target_names=list(target_names.keys()),
            output_dict=True
        )

    def plot_confusion_matrix(self, true, pred, model_name, target_names, sample=None):
        cm = confusion_matrix(true, pred)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names, yticklabels=target_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {model_name}')

        if sample is None:
            return cm

        part, split = sample.split('/')
        path = os.path.join(os.path.dirname(self.results_dir), split,
                            part, 'confusion matrices', f"{split}-{model_name.replace('Classifier', '')}.jpg")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.tight_layout()
        plt.xticks(rotation=45, ha='right')
        plt.savefig(path, dpi=500)
        plt.close()
        return cm, path

    async def _evaluate_single_model(self, model, X_train, X_test, y_train, y_test, sample):
        name = model.__class__.__name__
        print(f"Evaluating Model {name} on {sample}")

        def sync_evaluation():
            model.fit(X_train, y_train)
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            train_report = self.get_classification_report(y_train, train_pred)
            test_report = self.get_classification_report(y_test, test_pred)
            cm, path = self.plot_confusion_matrix(y_test, test_pred, name, target_names=target_names, sample=sample)
            return name, {
                'confusion_matrix': cm,
                'report': {'train': train_report, 'test': test_report},
                'plots': {'confusion_matrix': path}
            }

        return await asyncio.to_thread(sync_evaluation)

    async def evaluate(self, X_train, X_test, y_train, y_test, sample=None):
        tasks = [self._evaluate_single_model(model, X_train, X_test, y_train, y_test, sample) for model in self.models]
        results = await asyncio.gather(*tasks)
        return {name: result for name, result in results}