from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

GITHUB_REQUIREMENT = ("{name} @ git+https://github.com/{author}/{name}.git"
                      "@{version}")
REQUIREMENTS = ["numpy",
                "tensorflow-gpu==2.3.0",
                "scipy",
                "matplotlib",
                "h5py",
                "ray[tune]",
                "filelock",
                "natsort",
                GITHUB_REQUIREMENT.format(author="gfabieno",
                                          name="SeisCL",
                                          version="eef941d4e31b5fa0dc7823e491e"
                                                  "0575ad1e1f423"),
                GITHUB_REQUIREMENT.format(author="gfabieno",
                                          name="ModelGenerator",
                                          version="v0.1.1")]

setup(name="GeoFlow",
      version="0.0.1",
      author="Gabriel Fabien-Ouellet and collaborators",
      author_email="gabriel.fabien-ouellet@polymtl.ca",
      description="Dataset management interface with Keras for geophysics",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/gfabieno/GeoFlow",
      packages=find_packages(exclude=["Datasets", "logs"]),
      install_requires=REQUIREMENTS,
      classifiers=["Programming Language :: Python :: 3",
                   "License :: OSI Approved :: MIT License",
                   "Operating System :: OS Independent"],
      python_requires='>=3.6')
