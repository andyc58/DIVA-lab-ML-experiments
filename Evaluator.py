from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, log_loss
import seaborn as sns
from sklearn.preprocessing import label_binarize
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
        
    def __build(self, name ,sample, model_name):
        part, split = sample.split('/')
        path = os.path.join(os.path.dirname(self.results_dir), part, split, name, f"{split}-{model_name.replace('Classifier', '')}.jpg")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def plot_confusion_matrix(self, true, pred, model, target_names, sample):
        cm = confusion_matrix(true, pred)
        model_name = model.__class__.__name__

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names, yticklabels=target_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {model_name}')

        path = self.__build('confusion matrix', sample ,model_name)
        plt.tight_layout()
        plt.xticks(rotation=45, ha='right')
        plt.savefig(path, dpi=500)
        plt.close()
        return cm, path
    
    
    
    def plot_roc(self, model, X_test, y_test, target_classes, sample):
        y_test_bin = label_binarize(y_test, classes=sorted(y_test.unique()))
        n_classes = y_test_bin.shape[1]
        model_name = model.__class__.__name__

        # ROC for Random Forest
        y_score_rf = model.predict_proba(X_test)
        fpr_rf = dict()
        tpr_rf = dict()
        roc_auc_rf = dict()
        for i in range(n_classes):
            fpr_rf[i], tpr_rf[i], _ = roc_curve(y_test_bin[:, i], y_score_rf[:, i])
            roc_auc_rf[i] = auc(fpr_rf[i], tpr_rf[i])

        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        colors = ['blue', 'red', 'green', 'purple']
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr_rf[i], tpr_rf[i], color=color, lw=2,
                    label='ROC curve of class {0} (area = {1:0.2f})'.format(list(target_classes.keys())[i], roc_auc_rf[i]))
            

        path = self.__build('roc curve',sample, model_name)
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.savefig(path, dpi=500)
        plt.close()
        return path
        
    
    
    
    

    async def _evaluate_single_model(self, model, X_train, X_test, y_train, y_test, sample):
        name = model.__class__.__name__
        print(f"Evaluating Model {name} on {sample}")

        def sync_evaluation():
            model.fit(X_train, y_train)
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            model_train_logloss = log_loss(y_train, model.predict_proba(X_train))
            model_test_logloss = log_loss(y_test, model.predict_proba(X_test))
            
            train_report = self.get_classification_report(y_train, train_pred)
            test_report = self.get_classification_report(y_test, test_pred)
            cm, cm_path = self.plot_confusion_matrix(y_test, test_pred, model, target_names=target_names, sample=sample)
            roc_path = self.plot_roc(model, X_test, y_test, target_names, sample)
            
            
            return name, {
                'confusion_matrix': cm.tolist(),
                'report': {'train': train_report, 'test': test_report},
                'logloss': {'train': model_train_logloss, 'test': model_test_logloss},
                'plots': {
                          'confusion_matrix': cm_path,
                          'roc_curve': roc_path
                        }
                }

        return await asyncio.to_thread(sync_evaluation)

    async def evaluate(self, X_train, X_test, y_train, y_test, sample=None):
        tasks = [self._evaluate_single_model(model, X_train, X_test, y_train, y_test, sample) for model in self.models]
        results = await asyncio.gather(*tasks)
        return {name: result for name, result in results}
    
