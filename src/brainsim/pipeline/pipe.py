from abc import ABC, abstractmethod
from pathlib import Path


# Set the names that should be imported with *-imports
__all__ = ['matches_all_patterns', 'walk', 'Pipe', 'run_pipes']


def matches_all_patterns(path, patterns):
    path = Path(path)
    for pattern in patterns:
        # Check if at least one file/directory matches glob pattern
        # Stop iteration is raised if there is no match and we return False
        try:
            next(iter(path.glob(pattern)))
        except StopIteration:
            return False
    
    # If all patterns matched, then we return True
    return True


def walk(path, patterns):
    """Iterate over all directory that matches the given glob patterns.

    Arguments
    ---------
    path : pathlib.Path or str
        Root path to start recursively iterating in
    patterns : list of str
        List of glob patterns. Each glob pattern must have at least one match.
    
    Yields
    ------
    list of pathlib.Path
        List of paths that match all glob patterns
    """
    path = Path(path)
    if matches_all_patterns(path, patterns):
        yield path
    
    for subpath in path.glob("*/"):
        yield from walk(subpath, patterns)


class Pipe(ABC):
    required_glob_patterns = []

    def __init__(self, rerun_all=False):
        self.rerun_all = rerun_all
    
    def __call__(self, directory):
        if self.check_already_ran(directory) and not self.rerun_all:
            return
        if not matches_all_patterns(directory, self.required_glob_patterns):
            raise ValueError(f"Directory {path} doesn't match all required glob patterns:\n"
                             + "\n * ".join(self.required_glob_patterns))
        self.run_pipe(directory)

    @abstractmethod
    def check_already_ran(self, directory):
        raise NotImplementedError()
    
    @abstractmethod
    def run_pipe(self, directory):
        raise NotImplementedError()


def run_pipes(root, pipes):
    for pipe in pipes:
        for directory in walk(root, pipe.required_glob_patterns):
            pipe(directory)


if __name__ == "__main__":
    patterns = [
        'surf/lh.pial',
        'surf/rh.pial',
    ]

    for path in walk(".", patterns):
        print(path)
