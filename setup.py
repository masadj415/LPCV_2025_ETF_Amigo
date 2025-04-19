from setuptools import setup, find_packages

# Function to read requirements.txt
def read_requirements():
    with open("requirements.txt", "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="LPCV_TRACK_1",
    version="1.0",
    packages=find_packages(where="src"),  # Finds packages inside `src/`
    package_dir={"": "src"},  # Maps the root package to `src/`
    install_requires=read_requirements(),  # Reads dependencies from requirements.txt
)