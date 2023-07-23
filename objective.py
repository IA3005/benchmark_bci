from benchopt import BaseObjective, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import balanced_accuracy_score as BAS
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import KFold, LeaveOneGroupOut
    from skorch.helper import SliceDataset, to_numpy

# The benchmark objective must be named `Objective` and
# inherit from `BaseObjective` for `benchopt` to work properly.


class Objective(BaseObjective):

    # Name to select the objective in the CLI and to display the results.
    name = "BCI"

    # List of parameters for the objective. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.

    link = 'pip: git+https://github.com/Neurotechx/moabb@develop#egg=moabb'
    intall_cmd = 'conda'
    requirements = [link,
                    'scikit-learn']

    parameters = {
        'evaluation_process, subject, subject_test, session_test': [
            ('inter_session', 1, None, None),
            ('intra_subject', 1, None, None),
            ('inter_subject', None, None, None),

        ],
    }

    cv = KFold(n_splits=5, random_state=42, shuffle=True)
    # cv object from sklearn

    # The solvers will train on all the subject except subject_test.
    # It will be the same for the sessions.
    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.

    # min_benchopt_version = "1.3.2" , we don't specify the version because
    # we are working with the branch ENH_implement_cv of chris-mrn/benchopt

    def set_data(self, dataset, sfreq):
        # The keyword arguments of this function are the keys of the dictionary
        # returned by `Dataset.get_data`. This defines the benchmark's
        # API to pass data. This is customizable for each benchmark.

        data_split_subject = dataset.split('subject')

        if self.evaluation_process == 'intra_subject':

            dataset = data_split_subject[str(self.subject)]
            X = SliceDataset(dataset, idx=0)
            y = np.array(list(SliceDataset(dataset, idx=1)))-1
            # we have to susbtract 1 to the labels for compatibility reasons
            # with the deep learning solvers

            self.X = X
            self.y = y

        elif self.evaluation_process == 'inter_subject':
            # the evaluation proccess here is to leave one subject out
            #  to test on it and train on the rest of the subjects

            data_inter_subject = []
            group_subject = []
            for key in data_split_subject.items():
                id_subject = int(key[0])
                data_subject = data_split_subject[str(id_subject)]
                data_inter_subject += data_subject
                group_subject += [id_subject for i in range(len(data_subject))]

            X = SliceDataset(data_inter_subject, idx=0)
            y = np.array(list(SliceDataset(data_inter_subject, idx=1)))-1

            # you need to define the groups for the cv object here

            self.groups = group_subject

            self.cv = LeaveOneGroupOut()

            self.X = X
            self.y = y

        elif self.evaluation_process == 'inter_session':
            # the evaluation proccess here is to leave one session out
            #  to test on it and train on the rest of the sessions
            data_inter_session = []
            group_session = []
            data_subject = data_split_subject[str(self.subject)]
            data_split_session = data_subject.split('session')
            for key in data_split_session.items():
                id_session = key[0]
                data_session = data_split_session[key[0]]
                data_inter_session += data_session
                group_session += [id_session for i in range(len(data_session))]

            X = SliceDataset(data_inter_session, idx=0)
            y = np.array(list(SliceDataset(data_inter_session, idx=1)))-1

            self.groups = group_session

            self.cv = LeaveOneGroupOut()

            self.X = X
            self.y = y

        self.sfreq = sfreq

        return dict(
            X=self.X,
            y=self.y
        )

    def compute(self, model):
        # The arguments of this function are the outputs of the
        # `Solver.get_result`. This defines the benchmark's API to pass
        # solvers' result. This is customizable for each benchmark.
        if not type(model) == 'braindecode.classifier.EEGClassifier':
            self.X_train = to_numpy(self.X_train)
            self.X_test = to_numpy(self.X_test)

        # we compute here the predictions so
        # that we don't compute it for each score
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)

        score_train = accuracy_score(self.y_train, y_pred_train)
        score_test = accuracy_score(self.y_test, y_pred_test)
        bl_acc = BAS(self.y_test, y_pred_test)

        # This method can return many metrics in a dictionary. One of these
        # metrics needs to be `value` for convergence detection purposes.
        return dict(score_test=score_test,
                    value=-score_test,
                    score_train=score_train,
                    balanced_accuracy=bl_acc)

    def get_one_solution(self):
        # Return one solution. The return value should be an object compatible
        # with `self.compute`. This is mainly for testing purposes.
        return DummyClassifier().fit(self.X_train, self.y_train)

    def get_objective(self):
        # Define the information to pass to each solver to run the benchmark.
        # The output of this function are the keyword arguments
        # for `Solver.set_objective`. This defines the
        # benchmark's API for passing the objective to the solver.
        # It is customizable for each benchmark.

        '''
        If you choosed a cv that needs to have a group parameter,
        you need to pass in the inputs of get_split.
        '''

        if self.evaluation_process == 'intra_subject':

            X_train, X_test, y_train, y_test = self.get_split(
                                                        self.X,
                                                        self.y)

        else:

            X_train, X_test, y_train, y_test = self.get_split(
                                                        self.X,
                                                        self.y,
                                                        groups=self.groups)

        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test

        return dict(
            X=self.X_train,
            y=self.y_train,
            sfreq=self.sfreq,
        )
