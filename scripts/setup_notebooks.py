"""One-time per-clone setup for notebook-friendly git config: run after `uv sync`.

Configures two things:
  - nbstripout: strips cell outputs/execution counts on `git add`/commit.
  - nbdime: diffs/merges `.ipynb` cell-by-cell instead of as raw JSON.

`.gitattributes` is already tracked with the right `filter=`/`diff=`/`merge=`
rules, so nbstripout is never invoked with `--install`: with no `--attributes`
argument that command defaults to writing straight into `.git/info/attributes`
(a per-clone, untracked file that outranks the tracked `.gitattributes`), and
it unconditionally adds a conflicting `*.ipynb diff=ipynb` line there, silently
disabling the nbdime diff driver. It's also redundant: the only thing it would
add beyond what's below is that same attributes line. `nbdime config-git
--enable` is safe to keep since it only touches the tracked `.gitattributes`
and no-ops when `diff=jupyternotebook` is already present. Both tools'
installers also register their commands either as a bare name
(`git-nbdiffdriver`, only found if the venv's bin/ happens to be on PATH) or
as an absolute path into *this* venv (breaks the moment this checkout moves
or another worktree/clone uses a different one). Repointing everything
through `uv run` makes `git diff`/`add`/`merge` work from a plain shell in
any clone or worktree of this repo. Safe to re-run.
"""

import subprocess


def run(*cmd: str) -> None:
    subprocess.run(cmd, check=True)


def git_config(key: str, value: str) -> None:
    subprocess.run(["git", "config", "--local", key, value], check=True)


def main() -> None:
    run("uv", "run", "nbdime", "config-git", "--enable")

    git_config("filter.nbstripout.clean", "uv run nbstripout")
    git_config("filter.nbstripout.smudge", "cat")
    git_config("filter.nbstripout.required", "true")
    git_config("diff.jupyternotebook.command", "uv run git-nbdiffdriver diff")
    git_config(
        "merge.jupyternotebook.driver", "uv run git-nbmergedriver merge %O %A %B %L %P"
    )
    git_config(
        "difftool.nbdime.cmd", 'uv run git-nbdifftool diff "$LOCAL" "$REMOTE" "$BASE"'
    )
    git_config(
        "mergetool.nbdime.cmd",
        'uv run git-nbmergetool merge "$BASE" "$LOCAL" "$REMOTE" "$MERGED"',
    )

    print("Notebook git tooling installed: outputs auto-strip, diffs/merges are cell-aware.")


if __name__ == "__main__":
    main()
