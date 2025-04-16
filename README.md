# An Environment Adaptation Agent of Reinforcement Learning in Continuous Integration Test Case Prioritization
The project contains experimental code and datasets for the proposed method, a reinforcement learning-based test case prioritization technique to mitigate test case prioritization performance fluctuations.
## Directory Structure
`src`: Source code for EAA.

`dataset.zip`: The dataset used in the experiment.
## Requirements
```Python```: 3.10.9
```PyTorch```: V2.2.0
```numpy```: 1.23.5
```pandas```: 1.5.3
```scipy```: 1.10.0
```openpyxl```: 3.0.10
## Execution
Run.py is the main entry file. You can run the experiment using ```python Run.py```. The execution results will be saved in the **results** folder at the same level as the src directory. You can use ```python Statistic.py``` to analyze the execution results, which will be saved in **EAA.xlsx**.
