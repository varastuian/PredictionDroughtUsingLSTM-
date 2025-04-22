from graphviz import Digraph

dot = Digraph(comment='Time Series Model Evaluation Pipeline', format='png')
dot.attr(rankdir='LR', size='40')

# Step 1: Load data
dot.node('A', 'Load SPI Dataset')

# Step 2: Preprocess
dot.node('B', 'For each station & SPI\n- Drop NA\n- Create TimeSeries')

# Step 3: Train-test split
dot.node('C', 'Train/Validation Split\n+ Scale')

# Step 4: Model Loop
dot.node('D', 'Loop over Models\nExtraTF, RandomRF, SVR, LSTM, WBBLSTM')

# Step 5a: Wavelet Step (for WBBLSTM)
dot.node('D1', 'Wavelet Decompose\n(only WBBLSTM)', shape='box')

# Step 6: Train
dot.node('E', 'Train Model\n(if not exists)')

# Step 7: Predict
dot.node('F', 'Forecast on Validation\n+ Inverse Scale')

# Step 8: Metrics
dot.node('G', 'Compute Metrics:\nRMSE, MAE, Corr, SM\n*Correct scale!*')

# Step 9: Save
dot.node('H', 'Save Model & Metrics')

# Step 10: Choose Best
dot.node('I', 'Select Best Model\nBased on Corr or RMSE')

# Connect the nodes
dot.edges([('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E')])
dot.edge('D', 'D1', constraint='false', label='WBBLSTM')
dot.edge('D1', 'E', label='Approximation only')
dot.edge('E', 'F')
dot.edge('F', 'G')
dot.edge('G', 'H')
dot.edge('H', 'I')

# Save and render
dot.render('model_workflow', view=True)
