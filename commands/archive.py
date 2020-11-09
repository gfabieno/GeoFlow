# -*- coding: utf-8 -*-
"""Create an archive of the project in a subdirectory."""

import sys
from os import makedirs, listdir, remove, symlink, walk, chdir, getcwd
from os.path import join, realpath, exists, pardir
from subprocess import run
from copy import deepcopy

LOGS_ROOT_DIRECTORY = "logs"
PROJECT_NAME = "Deep_2D_velocity"


class ArchiveRepository:
    """Create a duplicate of the current project in a subdirectory.

    `ArchiveRepository` can be used with the `with` statement. Upon entering,
    the current repository is copied to a unique subdirectory of
    `LOGS_ROOT_DIRECTORY` and a `model` directory is also created at the same
    path. The current working directory is set as the one containing the copied
    code. Upon exiting, the previous working directory is recovered.
    """

    def __init__(self):
        (self.logs, self.model,
         self.code, self.prototype) = self.create_directory_tree()

    def __enter__(self):
        self.archive_current_state()
        self.chdir()
        self.write(self.prototype)
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            return False
        else:
            self.recover_previous_state()
            return True

    def create_directory_tree(self):
        """Create the target subdirectory tree."""
        current_branch = run(["git", "branch", "--show-current"],
                             capture_output=True, check=True, text=True)
        current_branch = current_branch.stdout.strip("\n")
        current_commit = run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, check=True, text=True)
        current_commit = current_commit.stdout.strip("\n")
        current_message = run(["git", "show", "-s", "--format=%s"],
                              capture_output=True, check=True, text=True)
        current_message = current_message.stdout.strip("\n")
        current_commit = " ".join([current_commit, current_message])
        current_commit = current_commit.replace(" ", "_")
        logs_dir = join(LOGS_ROOT_DIRECTORY, current_branch, current_commit)
        logs_dir = realpath(logs_dir)
        if exists(logs_dir):
            contents = listdir(logs_dir)
            contents = [c for c in contents if c.isnumeric()]
            current_prototype = len(contents)
        else:
            current_prototype = 0
        current_prototype = str(current_prototype)
        logs_dir = join(logs_dir, current_prototype)
        makedirs(logs_dir)

        model_dir = join(logs_dir, "model")
        makedirs(model_dir)
        code_dir = join(logs_dir, "code")
        makedirs(code_dir)

        return logs_dir, model_dir, code_dir, current_prototype

    def archive_current_state(self):
        """Copy the current project to `self.code` using `git`."""
        logs_dir, code_dir = self.logs, self.code

        stash_name = run(["git", "stash", "create", "'Uncommitted changes'"],
                         capture_output=True, check=True, text=True)
        stash_name = stash_name.stdout.strip("\n")
        archive_name = join(logs_dir, "code.tar.gz")
        # Create an archived copy of the repository.
        run(["git", "archive", "-o", archive_name, stash_name])
        # Untar the archived code to `code_dir`.
        run(["tar", "-C", code_dir, "-zxf", archive_name])
        remove(archive_name)

        symlink(realpath("Datasets"), join(code_dir, "Datasets"),
                target_is_directory=True)

    def chdir(self):
        """Select `self.code` as the current working directory."""
        self._previous_dir = getcwd()
        self._previous_state = deepcopy(sys.path)

        chdir(self.code)
        sys.path = list(filter(lambda s: PROJECT_NAME not in s, sys.path))
        subdirectories = [x[0] for x in walk(self.code)]
        subdirectories = list(set(subdirectories))
        sys.path.extend(subdirectories)

    def recover_previous_state(self):
        """Set the previous working directory as the current one."""
        chdir(self._previous_dir)
        sys.path = self._previous_state
        del self._previous_dir, self._previous_state

    def write(self, *lines):
        """Write `lines` to `command.sh` at the commit level directory."""
        command_path = join(pardir, pardir, "command.sh")
        if exists(command_path):
            mode = "a"
        else:
            mode = "w+"
        with open(command_path, mode=mode) as command_file:
            for line in lines:
                command_file.write(line)
                command_file.write("\n")
