from sklearn.metrics import classification_report

target_names = {'Black or African American': 0,
 'East/Southeast Asian': 1,
 'Hispanic, Latino, or Spanish': 2,
 'White': 3
 }


class ModelEvaluator:
    def __init__(self, models):
        self.models = models
        
    def get_classification_report(self, true, pred):
        return classification_report(
                true,  # y_test first, not predictions
                pred,
                digits=4,
                target_names=list(target_names.keys()),
                output_dict=True
            )

    def evaluate(self, X_train, X_test, y_train, y_test):
        result_dict = {}

        for model in self.models:
            name = model.__class__.__name__
            print(f"Evaluating Model {name}")
            model.fit(X_train, y_train)

            train_predictions = model.predict(X_train)
            test_predictions = model.predict(X_test)
            train_report = self.get_classification_report(y_train, train_predictions)
            test_report = self.get_classification_report(y_test, test_predictions)
            result_dict[name] = {'report': {
                                     'train': train_report, 
                                     'test':test_report
                                    }
                                }

        return result_dict