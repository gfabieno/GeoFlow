# -*- coding: utf-8 -*-
"""Create an archive of the project in a subdirectory."""

import sys
from os import makedirs, listdir, remove, symlink, walk, chdir as vanilla_chdir
from os.path import join, realpath, exists
from subprocess import run

LOGS_ROOT_DIRECTORY = "temp_logs"


def archive_current_state():
    logs_dir, model_dir, code_dir = create_directory_tree()

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

    return logs_dir, model_dir, code_dir


def create_directory_tree():
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
        current_prototype = len(listdir(logs_dir))
    else:
        current_prototype = 0
    current_prototype = str(current_prototype)
    logs_dir = join(logs_dir, current_prototype)
    makedirs(logs_dir)

    model_dir = join(logs_dir, "model")
    makedirs(model_dir)
    code_dir = join(logs_dir, "code")
    makedirs(code_dir)

    return logs_dir, model_dir, code_dir


def chdir(dir_, project_name="Deep_2D_velocity"):
    vanilla_chdir(dir_)
    sys.path = list(filter(lambda s: project_name not in s, sys.path))
    subdirectories = [x[0] for x in walk(dir_)]
    subdirectories = list(set(subdirectories))
    sys.path.extend(subdirectories)
