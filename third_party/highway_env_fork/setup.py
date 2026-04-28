from setuptools import setup, find_packages
import os
import shutil

# The fork's top-level dir contains `envs/`, `vehicle/`, `road/`, `interval.py`, `utils.py`,
# `__init__.py`. Internal imports reference `highway_env.envs.*`, so the package must be
# installed under the name `highway_env`. We stage the files into a `highway_env/` subdir
# so setuptools picks it up as the package.

HERE = os.path.abspath(os.path.dirname(__file__))
STAGE = os.path.join(HERE, "highway_env")

if not os.path.exists(STAGE):
    os.makedirs(STAGE)
    for item in ["envs", "vehicle", "road", "interval.py", "utils.py"]:
        src = os.path.join(HERE, item)
        dst = os.path.join(STAGE, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        elif os.path.isfile(src):
            shutil.copy2(src, dst)
    # Rewrite __init__.py: fork's original uses `import envs` (bare absolute import)
    # which only worked in ad-hoc installs. Use relative import for a proper package.
    with open(os.path.join(STAGE, "__init__.py"), "w") as f:
        f.write(
            "import os\n"
            "os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'\n"
            "from . import envs  # triggers gym.register() for hetero envs\n"
            "name = 'highway_env'\n"
        )

setup(
    name="highway-env-iplan-fork",
    version="1.4.0",
    description="Author's fork of highway-env with Heterogeneous scenarios (installs as `highway_env`)",
    packages=find_packages(include=["highway_env", "highway_env.*"]),
    install_requires=[
        "gym",
        "numpy",
        "pygame",
        "matplotlib",
    ],
    python_requires=">=3.7",
)
