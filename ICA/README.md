# ICA Analysis Project

This project implements Independent Component Analysis (ICA) on topographical alignment matrices to identify patterns of parcel collaboration across different conditions. The analysis aims to uncover underlying structures in the data that may not be immediately apparent through traditional methods.

## Project Structure

```
ica-analysis-project
├── src
│   ├── ica_analysis.py       # Main script for performing ICA
│   └── utils
│       └── __init__.py       # Utility functions for data handling and visualization
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Requirements

To run this project, you need to install the following dependencies:

- numpy
- scikit-learn
- matplotlib

You can install the required packages using pip:

```
pip install -r requirements.txt
```

## Usage

1. Ensure that you have the necessary dependencies installed.
2. Place your topographical alignment matrices in an accessible directory.
3. Run the ICA analysis script:

```
python src/ica_analysis.py
```

This will load the matrices, perform ICA, and output the identified patterns of parcel collaboration.

## Contributing

Contributions to enhance the functionality of this project are welcome. Feel free to submit a pull request or open an issue for discussion.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.