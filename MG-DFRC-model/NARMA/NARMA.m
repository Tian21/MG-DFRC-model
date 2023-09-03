%% NARMA Benchmark Task
% This script is used to run NARMA benchmark on Mackey-Glass dynamical
% system with Simulink tool.

% Author: Tian Gan

clear
close all

loop = 1; % Runs 
train_err = zeros(loop,1);
test_err = zeros(loop,1); % (loop, 3 for mask / 5 for sampling)

rng(1,'twister');

for i = 1:loop
    tic
    %% Setup
    config.sequenceLength = 3000;
    memoryLength = 10;
    config.virtualnodes = 100;
    config.theta = 0.05;
    config.k = 2;
    config.tau_a = config.virtualnodes * config.theta;
    config.tau_b = config.virtualnodes * config.theta * config.k;

    config.memoryLength = '{10,5}'; %[0,0.5]

    [inputSequence, outputSequence] = generate_new_NARMA_sequence(config.sequenceLength,memoryLength);

    %% Time-multiplexing
    config.masking_type = '3'; % select between '1 = Sample and Hold','2 = Binary Weight Mask','3 = Random Weight Mask'
    [masking] = TimeMultiplexing(inputSequence,config.sequenceLength,config.virtualnodes,config);
    config.start_time = 0; % Starting time --- in order to make T = TFinal
    config.N = config.sequenceLength * config.virtualnodes; % Number of values
    timeline = config.start_time + config.theta*(0:config.N-1); % Generate time in matrix
    system_inputSequence = [timeline(:),masking(:)];

    %% Run Mackey-Glass in Simulink
    config.TFinal = config.theta * config.sequenceLength * config.virtualnodes;
    config.coupling = 0.86;
    config.Timescale = 0.05;
    config.delta_1st = 0.1; % input scaling
    config.n = 2; % Nonlinearity
    
    config.sample_time = config.theta;
    sim('SDLMackeyGlass.slx'); % ideal delay line
    state_matrix = [ans.simout(2:end)]';
    state_matrix = reshape(state_matrix,config.virtualnodes,[]);  
    
    %% Training --- ridge regression Wout = BA'(AA'-λI)^-1 / pseudo-inverse Wout =  B * pinv(A)
    [output_weights,system_train_output_sequence,target_train_state,system_test_output_sequence,...
        target_test_state,target_matrix] = train_test(state_matrix,outputSequence);

    %% Evaluation
    config.err_type = 'NRMSE';
        train_error = calculateError(system_train_output_sequence,target_train_state,config);
        test_error = calculateError(system_test_output_sequence,target_test_state,config);

        train_err(i,1) = train_error;
        test_err(i,1) = test_error;

    %% Demultiplexing
    config.plot_type = 'test set';
    [target_plot,system_plot] = demultiplexing(system_train_output_sequence,target_train_state,...
        system_test_output_sequence,target_test_state,config);
    
    %% Plot
    
    plot(target_plot(1:100),'r');
    hold on;
    plot(system_plot(1:100),'b--')
    
    xlabel('t')
    ylabel('x(t)')
    legend('target output','system output')

    toc
end