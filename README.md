# Ethio Mart

> Ethiomart is a data science project that scrapes data from five channels and provides users with instant access to the latest product prices on Telegram. It is developed using Python and a variety of powerful libraries.

## Built With

- Major languages used: Python3
- Libraries: numpy, pandas, matplotlib.pyplot, scikit-learn
- Tools and Technlogies used: jupyter notebook, Git, GitHub, Gitflow, VS code editor.

## Demonstration and Website

[Deployment link](Soon!)

## Getting Started

You can clone my project and use it freely and then contribute to this project.

- Get the local copy, by running `git clone https://github.com/amare1990/ethioMart.git` command in the directory of your local machine.
- Go to the repo main directory, run `cd ethiomart` command
- Create python environment by running `python3 -m venv venm-name`, where `ven-name` is your python environment you create
- Activate it by running:
- `source venv-name/bin/activate` on linux os command prompt if you use linux os
- `myenv\Scripts\activate` on windows os command prompt if you use windows os.

- After that you have to install all the necessary Python libraries and tools by running `pip install -r requirements.txt`
- To run this project, run `jupyter notebook` command from the main directory of the repo

### Prerequisites

- You have to install Python (version 3.8.10 minimum), pip, git, vscode.

### Dataset

- Implemented in the `scripts/data_preprocessing.py` module in this repo
 - Fetched data from five Telegram channels, `channels = ['@ZemenExpress', '@nevacomputer', '@Shewabrand', '@modernshoppingcenter', '@aradabrand2']`
 - Fetched only text data for easy processing and for shorter download time
 - Preprocessed and clead data
 - Saved the data in the pandas DataFrame format
 - Each text message download has three columns/features, `timestamp`, `sender_id`, and `message`

### Project Requirements
- Git, GitHub setup, adding `pylint' in the GitHub workflows
- Statistical and EDA analysis on the data, ploting
- Feching data from at least from five Telegram channels
- Preprocessing and save the data in the structured format
- Labeling `message` data into `product`, `Price` and `location` and then saving in `CoNLL` format


#### GitHub Action
- Go to the main directory of this repo, create paths, `.github/workflows`. And then add `pylint` linters
- Make it to check when Pull request is created
- Run `pylint scripts/data_preprocessing.py` to check if the code follows the standard format
- Run `autopep8 --in-place --aggressive --aggressive scripts/script_name.py` to automatically fix some linters errors
- Run `pylint scripts/labeling_CoNLL.py` to check if the code follows the standard format
- Run `autopep8 --in-place --aggressive --aggressive scripts/scripts/labeling_CoNLL.py` to automatically fix some linters errors

### Data Preprocessing

The main functionality is implemented in `cdata_preprocessing.py` module. `data_preprocessor.ipynb` is the notebook you open to run and view results.
In this portion of the task, the following analysis has been conducted.

- Feching data from five channels
- Data Summary:
    Summarize statistical descriptive statistics for both numerical features and object type features too.


- Preprocessing data, removing emojis, spaces, and so on
  Run /open the jupyter notebook named `data_preprocessor.ipynb` to to view the dat

### Labeling the Message
- The main functionality is implemented in `labeling_CoNLL.py` module.`labeler_CoNLL.ipynb` is the notebook you will open to save and view the labeled data.

### Future Works
- Refining labeling of product ames, prices and locations
- Refining labeling using pretrained NLP algorithms
- Training the model
- Interpret models developed



> #### You can gain more insights by running the jupter notebook and view plots.


### More information
- You can refer to [this link]() to gain more insights about the reports of this project results.

## Authors

👤 **Amare Kassa**

- GitHub: [@githubhandle](https://github.com/amare1990)
- Twitter: [@twitterhandle](https://twitter.com/@amaremek)
- LinkedIn: [@linkedInHandle](https://www.linkedin.com/in/amaremek/)

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

Feel free to check the [issues page](https://github.com/amare1990/ethioMart/issues).

## Show your support

Give a ⭐️ if you like this project, and you are welcome to contribute to this project!

## Acknowledgments

- Hat tip to anyone whose code was referenced to.
- Thanks to the 10 academy and Kifiya financial instituion that gives me an opportunity to do this project

## 📝 License

This project is [MIT](./LICENSE) licensed.
