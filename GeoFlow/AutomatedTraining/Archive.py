# -*- coding: utf-8 -*-
"""
Create an archive of the project in a subdirectory.

Duplicate the current working directory into a subdirectory at `<PROJECT_NAME>/
<current branch name>/<current commit hash and name>/<current revision>`,
create a directory for models and set the current working directory as the
duplicated repository. Uncommitted changes are preserved.
"""

import sys
from time import sleep
from importlib import import_module
from os import makedirs, listdir, remove, symlink, walk, chdir, getcwd
from os.path import join, realpath, exists, pardir
from subprocess import run, CalledProcessError
from copy import deepcopy
from weakref import proxy

PROJECT_NAME = "Deep_2D_velocity"


class ArchiveRepository:
    """
    Create a duplicate of the current project in a subdirectory.

    `ArchiveRepository` can be used with the `with` statement. Upon entering,
    the current repository is copied to a unique subdirectory of
    `self.logs_directory` and a `model` directory is also created at the same
    path. The current working directory is set as the one containing the copied
    code. Upon exiting, the previous working directory is recovered.

    Sample usage:
        with ArchiveRepository() as archive:
            ...
    """

    def __init__(self, logs_directory):
        self.logs_directory = logs_directory
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
        """
        Create the target subdirectory tree.
        """
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
        logs_dir = join(self.logs_directory, current_branch, current_commit)
        logs_dir = realpath(logs_dir)
        current_prototype = None
        while current_prototype is None:
            try:
                if exists(logs_dir):
                    contents = listdir(logs_dir)
                    contents = [c for c in contents if c.isnumeric()]
                    current_prototype = len(contents)
                else:
                    current_prototype = 0
                current_prototype = str(current_prototype)
                makedirs(join(logs_dir, current_prototype))
                logs_dir = join(logs_dir, current_prototype)
            except FileExistsError:
                current_prototype = None

        model_dir = join(logs_dir, "model")
        makedirs(model_dir)
        code_dir = join(logs_dir, "code")
        makedirs(code_dir)

        return logs_dir, model_dir, code_dir, current_prototype

    def archive_current_state(self):
        """
        Copy the current project to `self.code` using `git`.
        """
        logs_dir, code_dir = self.logs, self.code

        stash_name = None
        while stash_name is None:
            try:
                stash_name = run(["git", "stash", "create",
                                  "'Uncommitted changes'"],
                                 capture_output=True, check=True, text=True)
            except CalledProcessError:
                sleep(5)
        stash_name = stash_name.stdout.strip("\n")
        archive_name = join(logs_dir, "code.tar.gz")
        # Create an archived copy of the repository.
        run(["git", "archive", "-o", archive_name, stash_name])
        # Untar the archived code to `code_dir`.
        run(["tar", "-C", code_dir, "-zxf", archive_name])
        remove(archive_name)

        symlink(realpath("datasets"), join(code_dir, "datasets"),
                target_is_directory=True)

    def chdir(self):
        """
        Select `self.code` as the current working directory.
        """
        self._previous_dir = getcwd()
        self._previous_state = deepcopy(sys.path)

        chdir(self.code)
        sys.path = list(filter(lambda s: PROJECT_NAME not in s, sys.path))
        subdirectories = [x[0] for x in walk(self.code)]
        subdirectories = list(set(subdirectories))
        sys.path.extend(subdirectories)

    def recover_previous_state(self):
        """
        Set the previous working directory as the current one.
        """
        chdir(self._previous_dir)
        sys.path = self._previous_state
        del self._previous_dir, self._previous_state

    def write(self, *lines):
        """
        Write `lines` to `command.sh` at the commit level directory.
        """
        command_path = join(pardir, pardir, "command.sh")
        if exists(command_path):
            mode = "a"
        else:
            mode = "w+"
        with open(command_path, mode=mode) as command_file:
            for line in lines:
                command_file.write(line)
                command_file.write("\n")

    def import_main(self):
        """
        Dynamically and safely import the current main script.

        Sample usage:
            with ArchiveRepository() as archive:
                with archive.import_main() as main:
                    ...
        """
        return _ImportMain()


class _ImportMain:
    """Dynamically import the current `main`."""

    def __enter__(self):
        self.main = import_module("GeoFlow.__main__").main
        return proxy(self.main)

    def __exit__(self, exc_type, exc_value, tb):
        """Delete all references appropriately"""
        del sys.modules["__main__"]
        del self.main
