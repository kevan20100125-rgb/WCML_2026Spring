import DeepMIMO

# Load the default parameters
parameters = DeepMIMO.default_params()

# Set scenario name
parameters['scenario'] = 'O1_60'

# Set the main folder containing extracted scenarios
parameters['dataset_folder'] = r'C:\Users\xxx\Desktop\scenarios'

# Generate data
dataset = DeepMIMO.generate_data(parameters)