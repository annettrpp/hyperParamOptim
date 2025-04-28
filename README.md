# hyper parameter optimization with CG and Quasi Newton Methods


This project focuses on optimizing hyperparameters for neural networks using Python, TensorFlow, Keras, pandas, NumPy, SciPy, and scikit-learn. All development and experimentation were conducted on Google Colab.

**Datasets**
Three datasets from the UCI repository were used:

Wine Quality

Heart Disease Prediction

Bank Marketing

Each dataset was split into training (80%) and testing (20%) sets, with the training set further divided into training (70%) and validation (30%) subsets. Features and target columns were separated during preprocessing.

**Approach**
A set of utility functions was developed to automate hyperparameter sampling:

random_hyperparameters(): Randomly selects hyperparameters from predefined lists.

generate_sample(n): Generates n samples of random hyperparameters.

map_hyperparameters(): Adjusts sampled parameters to match MLPClassifier requirements.

Hyperparameter interactions were carefully considered, acknowledging that optimal parameter combinations depend on joint effects rather than individual parameters. Early stopping criteria were determined based on convergence behavior observed during descent method testing.

Initially, the optimization approach attempted to modify TensorFlow’s built-in gradient descent, but the strategy shifted to utilizing Kormos for faster experimentation. Neural networks were constructed dynamically based on randomly generated hyperparameters, and optimization was performed using BFGS and CG methods.

**Testing and Evaluation**
Testing involved:

Training models with the sampled hyperparameters.

Preprocessing data using SimpleImputer and StandardScaler.

Recording validation accuracies and training times for both BFGS and CG optimizers.

Selecting the best-performing hyperparameters based on validation accuracy.

Final model performance was evaluated across training, validation, and testing datasets using the MLPClassifier from scikit-learn. Visualizations of results were generated using bar plots comparing BFGS and CG performance.

**Results**
BFGS consistently demonstrated faster training times than CG while achieving similar validation accuracies across datasets. The project successfully highlights the impact of optimization methods on hyperparameter tuning efficiency and model performance.


**Works Cited**

Nikhil Mehta, Jonathan Lorraine, Steve Masson, Ramanathan Arunachalam, Zaid Pervaiz Bhat, James Lucas, Arun George Zachariah. "Improving hyperparameter optimization with checkpointed model weights." arXiv preprint arXiv:2406.18630 (2024).

Bernd Bischl, Martin Binder, Michel Lang, Tobias Pielok, Jakob Richter, Stefan Coors, Janek Thomas, Theresa Ullmann, Marc Becker, Anne-Laure Boulesteix, Difan Deng, Marius Lindauer. "Hyperparameter optimization: Foundations, algorithms, best practices, and open challenges." Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery 13.2 (2023): e1484.

https://docs.scipy.org/doc/scipy/index.html

https://pypi.org/project/kormos/

