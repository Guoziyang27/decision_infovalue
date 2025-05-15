# The Information Value in Human-AI Decision-making

This repository contains scripts and prompts for our paper "The Value of Information in Human-AI
Decision-making". Our decision_infovalue package consists of one main class:

- Initialization function
  - Estimate the DGP from the observed data (containing the coarsening process to avoid overfitting), and calculate the full information value and the no information value.
- Attributes
  - `dgp_data`: pd.DataFrame, that is used to estimate the Data-generating Process
  - `state`: str, Payoff-relevant state
  - `full_signals`: List[str], signals that are included in the Data-generating Process
  - `scoring_rule`: Callable, the scoring rule defining the decision making task
  - `binning_method`: str, the binning method used to bin the signals
  - `overfit_tolerance`: float, the tolerance for overfitting
  - `fit_test_ratio`: float, the ratio of the data used to fit the model and the data used to test the model
  - `all_use_data`: pd.DataFrame, the data after binning and the binning breaks
  - `all_breaks`: Dict[str, List[float]], the binning breaks for each signal
  - `full_info_value`: float, the full information value, i.e., the rational payoff when all signals are used
  - `no_info_value`: float, the no information value, i.e., the rational payoff when no signal is used
- Main methods
  - `complement_info_value`: Calculate the global complement information value of the signals (ACIV).
  - `instanse_complement_info_value`: Calculate the instance-level complement information value of the signals and its realization (ILIV).
  - `robust_analysis_on_v_shaped_scoring_rule`: Compare the informativeness between signals across a grid of V-shaped scoring rules.
  - `marginal_complement_info_value`: Same as `complement_info_value`, but calculate the marginal contribution of the signal across all possible combinations of signals.

## Experiment

- `notebooks/house-price-example/house_pricing.ipynb`: Generating of the stimuli in the experiment.
- `notebooks/house-price-example/survey_data.csv`: The generated stimuli.
- `R/analysis.Rmd`: The R scripts that used to analyze and generate the experimental figure and data in the paper.
- `R/data/exp_df_real_exp.csv`: The collected experiment data in the crowdsourcing experiment.
  - Each row contains a trial in the experiment
  - Columns:
    - `AI`: The AI model assigned to the trial
    - `explanation`: The explanation condition assigned to the trial
    - `ResponseId`, `PROLIFIC_PID`: Unique ID for each participant
    - `test_order`, `test_index`, `Order`: Unique ID for the house instance. `Order` and `test_order` are the same id in the Ames, Iowa house price dataset.
    - `test_mse`: The MSE achieved by the participant in the 24 trials
    - `Finished`: Whether participant finished all the trials
    - `displayed_reward`: The final reward given to the participant
    - `human_response`: Response in the first round, unit in $K
    - `hai_response`: Response in the second round, unit in $K
    - `Year_Built`, `Gr_Liv_Area`, `Garage_Cars`, `Fireplaces`, `Year_Remod`, `Overall_Qual`: Features of the house
    - `SalePrice`: The actual sale price
    - `pred1`, `pred2`: Predictions by AI1 and AI2

## Observation study

- `notebooks/observational-study/observational_study.ipynb`: Shows analysis of the Human-AI Interactions Dataset. We compare the performance of human+AI teams and show how human-complementary information of AI models can predict the improvement achieved by human+AI teams over human-alone decisions.

## Demostrations

The `notebooks` directory contains example Jupyter notebooks demonstrating how to use the toolkit:

- `deepfake-example/deepfake_unexploited_infovalue.ipynb`: Shows how to analyze the unexploited information value in each features by human-alone/AI/human+AI decisions in deepfake detection, where humans and AI systems collaborate to identify fake videos
- `recidivism-example/recidivism_prediction.ipynb`: Shows the SHAP explanations and ILIL-SHAP explanations for recidivsm prediction task.
- `mimic-iv-CXR-example/cxr_human_ai_complementary.ipynb`: Analyzes chest X-ray diagnosis with human-AI collaboration. We compare the human-complementary information value of different Vision models and conduct a robustness analysis over a grid of V-shaped proper scoring rules.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 