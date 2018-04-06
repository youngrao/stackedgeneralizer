test
import time
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

class StackedGeneralizer(object):
    def __init__(self, clfs=None, ensemble=None, n_folds=5, stratify = False, original = False):
        self.clfs = clfs
        self.ensemble = ensemble
        self.n_folds = n_folds
        self.X_trainEnsemble = None
        self.X_testEnsemble = None
        self.stratify = stratify
        self.original = original

    def get_pred(self, model, X):
        if hasattr(model, 'predict_proba'):
            pred = model.predict_proba(X)
        else:
            pred = model.predict(X)

        return pred

    def generate_base(self, X_train, y, X_test = None):
        print ('Generating Base Level Models...')

        if self.stratify:
            kf = StratifiedKFold(self.n_folds)
        else:
            kf = KFold(self.n_folds)

        for j, clf in enumerate(self.clfs):
            print('(%s) Fitting Model %d: %s' %(time.asctime(time.localtime(time.time())), j+1, clf))


            for i, (train, test) in enumerate(kf.split(X_train, y)):
                print('Fold %d' % (i + 1))

                clf.fit(X_train[train], y[train])
                X_trainEnsemble_clf = self.get_pred(clf, X_train[test])

                classes = X_trainEnsemble_clf.shape[1]

                if self.X_trainEnsemble is None:
                    self.X_trainEnsemble = np.zeros((X_train.shape[0], len(self.clfs)*classes))
                    self.X_trainEnsemble[test, j * classes:(j + 1) * classes] = X_trainEnsemble_clf
                else:
                    self.X_trainEnsemble[test, j * classes:(j + 1) * classes] = X_trainEnsemble_clf
                if X_test is not None:
                    if self.X_testEnsemble is None:
                        self.X_testEnsemble = np.zeros((X_test.shape[0], len(self.clfs)*classes))
                        self.X_testEnsemble[:,j * classes:(j + 1) * classes] += self.get_pred(clf, X_test)
                    else:
                        self.X_testEnsemble[:, j * classes:(j + 1) * classes] += self.get_pred(clf, X_test)

        self.X_testEnsemble = self.X_testEnsemble/self.n_folds

        if self.original:
            self.X_trainEnsemble = np.append(self.X_trainEnsemble, X_train, axis=1)
            self.X_testEnsemble = np.append(self.X_testEnsemble, X_test, axis=1)

    def generate_ensemble(self, X_trainEnsemble, y):
        print ('(%s) Stacking Base Level Models with %s' %(time.asctime(time.localtime(time.time())),self.ensemble))

        self.ensemble.fit(X_trainEnsemble, y)

    def fit(self, X_train, y, X_test=None):
        self.generate_base(X_train, y, X_test)
        self.generate_ensemble(self.X_trainEnsemble, y)

    def predict(self):
        return self.get_pred(self.ensemble, self.X_testEnsemble)
