Workflow:

1. Initialize the project using DataInit. After executing DataGenerator(config, if_save=True), manually adjust the parameters in config.yaml's data_generation.reward_parameters based on the generated images. Then continue executing the following code, especially exp4_data_manager = Exp4DataManager(config, data_path) and manage_and_save_data(config, 'reward'). They both require these parameters.

2. Next, in Analysis_algorithm, execute the dynamic visualization feature:
    - Select the data: load_reward_method (str), data_type (str), algorithm.
    - After making the selections, click "Confirm" to display the corresponding dynamic graph.