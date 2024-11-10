from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="scratch_nlp",                    # Required
    version="1.0.0",                             # Required
    author="Shanmukha Sainath",                          # Optional
    author_email="venkatashanmukhasainathg@gmail.com",       # Optional
    description="Library with NLP Algorithms implemented from scratch",  # Optional
    long_description=long_description,           # Optional
    long_description_content_type="text/markdown",  # Optional (use "text/markdown" if README.md is in markdown)
    url="https://github.com/shanmukh05/scratch_nlp",   # Optional
    packages=find_packages(),                    # Automatically find and include all packages in your project
    include_package_data=True,                   # Include other files specified in MANIFEST.in
    install_requires=requirements,                        # Required dependencies
    extras_require={                             # Optional dependencies for development or testing
        "dev": [
            # "pytest>=6.0",
            "black",
        ],
    },
    classifiers=[                                # Optional classifiers
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",                     # Specify the Python versions you support
    entry_points={                               # Optional console scripts
        "console_scripts": [
            "your_command=your_module:main_function",
        ],
    },
    project_urls={                               # Additional URLs about your project
        "Bug Tracker": "https://github.com/shanmukh05/scratch_nlp/issues",
        "Documentation": "https://your_project.readthedocs.io/",
        "Source Code": "https://github.com/shanmukh05/scratch_nlp",
    },
)
