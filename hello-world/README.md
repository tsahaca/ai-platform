# Hello World Notebook in Dev Container

This workspace runs a simple Jupyter notebook inside a VS Code Dev Container.

## Quick open

[![Open in Dev Container](https://img.shields.io/badge/Open%20in-Dev%20Container-0A84FF?logo=visualstudiocode&logoColor=white)](command:remote-containers.reopenInContainer)

Tip: This command link works when opened in VS Code with the Dev Containers extension installed.

## Files

- `.devcontainer/devcontainer.json`: Dev container configuration.
- `hello-world.ipynb`: Notebook with a Hello World cell.

## How to run

1. Open this folder in VS Code.
2. Run **Dev Containers: Reopen in Container** from the Command Palette.
3. Wait for the container setup to complete.
4. Open `hello-world.ipynb`.
5. Select the Python kernel if prompted.
6. Run the first cell.

Expected output:

```text
Hello, world from devcontainer!
```

## Troubleshooting

- If the command is missing, install the Dev Containers extension (`ms-vscode-remote.remote-containers`) and reload VS Code.
- If no kernel appears in the notebook, run **Python: Select Interpreter** and choose Python 3 in the container.
- If packages are missing, rebuild with **Dev Containers: Rebuild Container**.
- If the notebook still fails to start, open a terminal in the container and run `python -m ipykernel install --user --name python3`.
