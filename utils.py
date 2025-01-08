import tomllib


def get_dependencies(pyproject_path="pyproject.toml"):
    """
    Read dependencies from pyproject.toml and return them as a list of strings.

    Args:
        pyproject_path (str): Path to pyproject.toml file

    Returns:
        list: List of dependency strings
    """
    with open(pyproject_path, "rb") as f:  # Note: 'rb' mode is required for tomllib
        pyproject_data = tomllib.load(f)

    # Handle different possible locations of dependencies
    dependencies = []

    # Check project.dependencies (Poetry style)
    if "project" in pyproject_data:
        project_deps = pyproject_data["project"].get("dependencies", [])
        if isinstance(project_deps, list):
            dependencies.extend(project_deps)
        elif isinstance(project_deps, dict):
            dependencies.extend(f"{pkg}{ver}" for pkg, ver in project_deps.items())

    # Check tool.poetry.dependencies (older Poetry style)
    if "tool" in pyproject_data and "poetry" in pyproject_data["tool"]:
        poetry_deps = pyproject_data["tool"]["poetry"].get("dependencies", {})
        if poetry_deps:
            dependencies.extend(
                f"{pkg}{ver}" for pkg, ver in poetry_deps.items() if pkg != "python"
            )  # Typically exclude python version constraint

    return dependencies
