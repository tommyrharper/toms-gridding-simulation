"""One-time per-clone setup for notebook-friendly git config: run after `uv sync`.

Configures two things:
  - nbstripout: strips cell outputs/execution counts on `git add`/commit.
  - nbdime: diffs/merges `.ipynb` cell-by-cell instead of as raw JSON.

`.gitattributes` is already tracked with the right `filter=`/`diff=`/`merge=`
rules, so neither installer is passed `--attributes` here: nbstripout's own
`--attributes` flag would blindly re-append a conflicting `*.ipynb diff=ipynb`
line on every re-run (later lines win, silently disabling the nbdime diff
driver). Both installers also register their commands either as a bare name
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
    run("uv", "run", "nbstripout", "--install")
    run("uv", "run", "nbdime", "config-git", "--enable")

    git_config("filter.nbstripout.clean", "uv run nbstripout")
    git_config("filter.nbstripout.smudge", "cat")
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
