# Explainable Machine Learning Guided Modeling for Antennas
## Graduate Research Project Spring 2024


### Thesis
Antenna design processes require extensive electromagnetic (EM) simulation tasks that are resource-intensive, time-consuming, and prone to interruptions. Design equations are only available for predefined and limited antenna geometries. By applying a machine learning (ML) model to a limited set of data from EM simulations of a leaky wave antenna (LWA), performance metrics can be predicted significantly quicker than running simulations for an extensive range of geometric variations. The model can be used for inverse design techniques, where the performance requirements are provided as input, and the model generates a geometric solution that meets those requirements. Explainable ML (XAI) processes can also be applied to identify the most impactful geometric parameters, and additional simulations focusing on those parameters can be performed, creating an iterative process of intelligently generating a minimal amount of data to improve model performance. The design process was tested on the LWA, which saw an accuracy increase from 62.71\% to 73.43\% using XAI.

### Installation
1. Clone repo
2. Install Python 3.9.18 or similar
3. Install TensorFlow 2.15.0 by following [TensorFlow installation instructions](https://www.tensorflow.org/install/pip)
4. Install Python dependencies: `pip install -r requirements.txt`

### Quick-start for GUI
1. Update and run all cells in [training_leaky_wave_new.ipynb](training/training_leaky_wave_new.ipynb) to train the model
2. Move generated `antenna_model.pkl` file to main directory
3. Update and run all cells in [generate_all_predictions.ipynb](research/generate_all_predictions.ipynb) to generate the SQLite database of all predictions
4. Verify that the file [test_data/lw.db](test_data/lw.db) was created successfully
5. Run all cells in [get_dimensions_from_s11.ipynb](get_dimensions_from_s11.ipynb) to start the GUI
6. Scroll to bottom


### Process for adding a new antenna
1. Create new folder in [test_data](test_data), add CSV files to folder
2. Modify [training_leaky_wave_new.ipynb](training/training_leaky_wave_new.ipynb) or duplicate file to reference new CSV file, then run to generate `antenna_model.pkl`
3. Move generated `antenna_model.pkl` file to main directory
4. Modify [generate_all_predictions.ipynb](research/generate_all_predictions.ipynb) or duplicate file to reference new CSV file, then run to generate [test_data/lw.db](test_data/lw.db)
5. Modify [get_dimensions_from_s11.ipynb](get_dimensions_from_s11.ipynb) to reference new CSV, and change SQL query to include the geometric parameters from new antenna


### GUI Usage Instructions
1. Enter negative $S_{11}$ value 
2. Use slider to choose minimum and maximum frequency values
3. Check the box "Only Show Best" to filter the geometries to only the ones with the lowest maximum value between the chosen frequency range
4. Check the box "Only Show Simulated" to only show values that came from the original simulated dataset
5. You must wait 3-4 minutes each time the "Run" button is pressed in order for the SQL query to work. If you press "Run" multiple times, as soon as the previous query finishes, another will start.


---

### How directories are organized

| Directory Name | What does the file or directory contain? |
|---|---|
| [grp](grp) | Latex files for graduate research project proposal and report |
| [old](old) | Original files from previous students who worked on this project |
| [training](training) | Code to perform preprocessing on HFSS CSV data and train best performing model using data, saving model to `antenna_model.pkl` file |
| [test_data](test_data) | All of the CSV files from HFSS for al antennas with the geometric parameter sweeps with the corresponding frequency and s11 values.<br>Code for combining multiple CSV into one.<br>`lw.db` SQLite file |
| [requirements.txt](requirements.txt) | Required Python packages |
| [get_dimensions_from_s11.ipynb](get_dimensions_from_s11.ipynb) | Main code for the GUI, searching SQLite database and displaying graph |
| [research/antenna_predictor_gui.ipynb](research/antenna_predictor_gui.ipynb) | Original prototype for GUI |
| [research/compare_models.ipynb](research/compare_models.ipynb) | Hyperparameter tuning and comparison of regression sklearn models for patch antenna |
| [research/compare_models_lwn.ipynb](research/compare_models_lwn.ipynb) | Hyperparameter tuning and comparison of regression sklearn models for leaky wave antenna |
| [research/data_outside_range.ipynb](research/data_outside_range.ipynb) | Experimenting what happens when you predict on data outside of the training data range |
| [research/dnn_lwn_inverse.ipynb](research/dnn_lwn_inverse.ipynb) | Experimenting with making DNN with S11 and freq as input and geometric parameter as output |
| [research/dnn_lwn.ipynb](research/dnn_lwn.ipynb) | Training a DNN on leaky wave antenna data and comparing with Sklearn  |
| [research/dnn.ipynb](research/dnn.ipynb) | Training a DNN on patch antenna data and comparing with Sklearn  |
| [research/exploring_using_full_freq_range.ipynb](research/exploring_using_full_freq_range.ipynb) | Use both SQLite and pkl database to search predictions for specified geometries and plot them alongside to simulated values |
| [research/generate_all_predictions.ipynb](research/generate_all_predictions.ipynb) | Generate predictions for all geometries in generated database and save to SQLite iteratively or pkl all at once<br>Print min, max, and step values for each geometric paramter of antenna |
| [research/generate_learning_curve.ipynb](research/generate_learning_curve.ipynb) | Train sklearn model at intervals to generate learning curve |
| [research/shap_test_lwn.ipynb](research/shap_test_lwn.ipynb) | Experiment with shap for leaky wave antenna and generate beeswarm and waterfall plots |
| [research/shap_test.ipynb](research/shap_test.ipynb) | Experiment with shap for patch antenna and generate beeswarm and waterfall plots |
| [research/sql_test.ipynb](research/sql_test.ipynb) | Experimenting with interfacing with SQLite using SQL query |
| [research/unseen_rows.ipynb](research/unseen_rows.ipynb) | Generate sample of random "unseen" geometries |


### Common Issues
- If you have trouble getting Plotly to display inside VScode, try this fix https://stackoverflow.com/a/68718345/3675086
- If VScode keeps crashing or freezing when experimenting with the GUI, please limit the amount of geometries that are plotted manually or by changing your search query