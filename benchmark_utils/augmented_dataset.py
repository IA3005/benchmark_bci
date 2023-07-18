from benchopt import BaseSolver, safe_import_context
from abc import abstractmethod, ABC

from benchmark_utils.transformation import (
    channels_dropout,
    smooth_timemask,
)

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from sklearn.utils import resample
    from skorch.helper import to_numpy


class AugmentedBCISolver(BaseSolver, ABC):
    """Base class for solvers that use augmented data.
    This class implements some basic methods from another methods ihnerited.
    """

    @abstractmethod
    def set_objective(self, **objective_dict):
        pass

    @property
    def name(self):
        pass

    def run(self, n_iter):
        n_samples = [0.1, 0.25, 0.5, 0.7, 1, 2, 5, 7, 10, 20]

        """Run the solver to evaluate it for a given number of iterations."""
        if self.augmentation == "ChannelsDropout":
            X, y = channels_dropout(self.X, self.y, n_augmentation=n_iter)

        elif self.augmentation == "SmoothTimeMask":
            X, y = smooth_timemask(
                self.X, self.y, n_augmentation=n_iter, sfreq=self.sfreq
            )
        elif self.augmentation == "Sampler":
            X, y = resample(self.X, self.y,
                            n_samples=int(len(self.X) * n_samples[n_iter]),
                            random_state=42)
        else:
            X = to_numpy(self.X)
            y = self.y

        self.clf.fit(X, y)

    def get_next(self, n_iter):
        return n_iter + 1

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function are the arguments of `Objective.compute`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return self.clf

    def warmup_solver(self):
        pass