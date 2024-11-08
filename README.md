# Retrieval Augmented Generation for Digital Humanities

## Introduction

On this project, I implement a retrieval augmented generation using Gemma and a dataset of News in Japanese and English from different newspapers in Japan between 2001 and 2021.
This project has been developed as an assignment for the course "Collaboration across STEM and Liberal Arts:AI design concept and technology that supports digital humanities" at the Institute of Sciences Tokyo.

## Project requirements

- At least 8GB of Disk space
- Python 3.11
- pip
- Kaggle API key

## Installation in Colab

Follow the steps [here](collab_install.ipynb) to install and run the project in Google Colab.

## Installation

1. Clone the repository
``` git clone https://github.com/irsotarriva/DH.git ```
2. Install the required packages
To install the required packages, run the following command:
```bash sudo pip install -r requirements.txt```
Sometimes pip will be installed as pip3, if that is the case, run the following command:
```bash sudo pip3 install -r requirements.txt```
3. Obtain your Kaggle API key, it will be required the  first time you open the program. For more information on how to get your Kaggle API key, visit [Kaggle](https://www.kaggle.com/docs/api). The Kaggle API key will be used to download the dataset and the pre-trained model.
4- Install the submodules. This will download the Gemma repository.
```git submodule update --init --recursive```
5- Install Gemma by cd into the Gemma repository and running the following command:
``` pip install -e . ```

## Usage

1. Load the python environment
``` source {DH_INSTALLATION_PATH}/py312/bin/activate ```
2. Run the main file
``` python main.py ```
3. You can look for articles by entering a keywords in Japanese or English. The system will return the most relevant articles and answer your querry based on the articles found.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Authors and Acknowledgment

This project was developed by Isai Roberto Sotarriva Alvarez as an assignment for the course "Collaboration across STEM and Liberal Arts:AI design concept and technology that supports digital humanities"
I would like to thank the Institute of Sciences Tokyo for providing me with the opportunity to develop this project.

## Contact

If you have any questions, feel free to contact me by [email](mailto:sotarriva.i.aa@m.titech.ac.jp).

## References

- [Institute of Sciences Tokyo](https://www.titech.ac.jp)
- [Kaggle](https://www.kaggle.com)
- [Gemma](https://ai.google.dev/gemma)
- [Gemma repository](https://github.com/google/gemma_pytorch?tab=readme-ov-file)
