from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="ScratchNLP",                    
    version="1.0.0",                            
    author="Shanmukha Sainath",                         
    author_email="venkatashanmukhasainathg@gmail.com",     
    description="Library with NLP Algorithms implemented from scratch", 
    long_description=long_description,           
    long_description_content_type="text/markdown",  
    url="https://github.com/shanmukh05/scratch_nlp",  
    packages=find_packages(),                    
    include_package_data=True,                  
    install_requires=requirements,                       
    extras_require={                          
        "dev": [
            "black",
        ],
    },
    classifiers=[                               
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",                     
    entry_points={                              
        "console_scripts": [
            "your_command=src:main",
        ],
    },
    project_urls={                              
        "Bug Tracker": "https://github.com/shanmukh05/scratch_nlp/issues",
        "Documentation": "https://shanmukh05.github.io/scratch_nlp/",
        "Source Code": "https://github.com/shanmukh05/scratch_nlp",
    },
)
