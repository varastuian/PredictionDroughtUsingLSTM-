from graphviz import Digraph

dot = Digraph(comment='Time Series Model Evaluation Pipeline', format='png')
dot.attr(rankdir='TB', size='140')


# Step 1: Load raw data
dot.node('A', 'Load Raw Dataset with precipitation\n', shape='cylinder')

# Step 2: Compute SPI
dot.node('A1', 'Compute SPI\n(Gamma Distribution)', shape='parallelogram')

# Step 3: Preprocess
dot.node('B', 'For each station & SPI:\n- Drop NA\n- Create TimeSeries', shape='box')

# Step 4: Train-test split
dot.node('C', 'Train/Validation Split\n+ Scale', shape='box')

# Step 5: Model Loop
dot.node('D', 'Loop over Models:\nExtraTF, RandomRF, SVR, LSTM, WBBLSTM', shape='box3d')

# Step 6a: Wavelet Step (for WBBLSTM)
dot.node('D1', 'Wavelet Decompose\n(only WBBLSTM)', shape='parallelogram')

# Step 7: Train
dot.node('E', 'Train Model\n', shape='box')

# Step 8: Predict
dot.node('F', 'Forecast on Validation\n+ Inverse Scale', shape='box')

# Step 9: Metrics
dot.node('G', 'Compute Metrics:\nRMSE, MAE, Corr, SM\n(*Correct scale*)', shape='parallelogram')

# Step 10: Save
dot.node('H', 'Save Model & Metrics', shape='cylinder')

# Step 11: Choose Best
dot.node('I', 'Select Best Model\n(Based on Corr or RMSE)', shape='diamond')

# Connections
dot.edge('A', 'A1')
dot.edge('A1', 'B')
dot.edge('B', 'C')
dot.edge('C', 'D')
dot.edge('D', 'E')
dot.edge('D', 'D1', label='WBBLSTM only', style='dashed')
dot.edge('D1', 'E', label='use approx')
dot.edge('E', 'F')
dot.edge('F', 'G')
dot.edge('G', 'H')
dot.edge('H', 'I')

# Render to file
dot.render('model_workflow_vertical', view=True)