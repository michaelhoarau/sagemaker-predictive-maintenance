import boto3
import matplotlib.pyplot as plt
import mxnet as mx
import mxnet.gluon as G
import numpy as np
from matplotlib import cm
from IPython.display import clear_output
import time
import sagemaker
import pandas as pd

def generate_sequences(df, sequence_length, columns):
    array = df[columns].values
    num_elements = array.shape[0]

    for start, stop in zip(range(0, num_elements - sequence_length), range(sequence_length, num_elements)):
        yield array[start:stop, :]
        
def generate_labels(df, sequence_length, target_column):
    array = df[target_column].values
    num_elements = array.shape[0]
    return array[sequence_length:num_elements, :]

def generate_training_sequences(df, sequence_length, features, unit_list):
    sequences = []
    for unit_id in unit_list:
        # Get a subset of the data for the current engine unit:
        subset = df.loc[(unit_id)]

        # Generate all the sequences for the current unit:
        if (len(features) > 1):
            subset_sequences = list(generate_sequences(subset, sequence_length, features))
        else:
            subset_sequences = generate_labels(subset, sequence_length, features)

        # Append them to the overal sequences object:
        sequences.append(subset_sequences)

    # Convert the list of sequences into a numpy array:
    sequences = np.concatenate(sequences).astype(np.float32)
    
    return sequences

def generate_testing_sequences(df, sequence_length, features, target, unit_list):
    sequences = []
    unit_span = []
    labels = []
    
    for unit_id in unit_list:
        #print('----------------------')
        #print(unit_id)
        # Get a subset of the data for the current engine unit:
        subset = df.loc[(unit_id)]
        #print(subset.shape)

        # Generate all the sequences for the current unit:
        #print('Sequences:')
        subset_sequences = list(generate_sequences(subset, sequence_length, features))
        #print(len(subset_sequences))
        
        # Generate labels for the current unit:
        #print('Labels:')
        subset_labels = generate_labels(subset, sequence_length, target)
        #print(len(subset_labels))

        # Append them to the overall sequences object:
        if len(subset_sequences) > sequence_length:
            sequences.append(subset_sequences)
            unit_span.append(len(subset_sequences))
            labels.append(subset_labels)
            
        else:
            print('Unit {} test sequence ignored, not enough data points.'.format(unit_id))
            unit_span.append(0)

    # Convert the lists of sequences into numpy arrays:
    labels = np.concatenate(labels).astype(np.float32)        
    sequences = np.concatenate(sequences).astype(np.float32)
            
    return sequences, labels, unit_span

def plot_timestep(nb_rows, nb_cols, nb_signals, step_index, position, timestep_sequences, features, plots_per_row):
    viridis = cm.get_cmap('viridis', nb_signals)
    
    # Loop through each signal:
    axes = []
    for j in range(nb_signals):
        # Plot the current signal:
        plt.subplot(nb_rows, nb_cols, step_index * plots_per_row + j + position)
        plt.plot(timestep_sequences[j], color=viridis.colors[j], linewidth=0.5, marker='o', markevery=[-1], markerfacecolor='#CC0000', markeredgewidth=0)
        
        # Hide tick labels:
        plt.gca().axes.xaxis.set_ticklabels([])
        plt.gca().axes.yaxis.set_ticklabels([])
        
        # We set the plot title only for the first row:
        if (step_index == 0):
            plt.title(features[j], {'fontsize': 8})
        
def plot_text(fig, nb_rows, nb_cols, nb_signals, step_index, txt, position, no_axis, main_title, plots_per_row, options):
    ax = fig.add_subplot(nb_rows, nb_cols, step_index * plots_per_row + position)
    ax.text(0.5, 0.5, txt, horizontalalignment='center', verticalalignment='center', **options)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    
    if no_axis == True:
        ax.set_axis_off()
        
    if (step_index == 0):
        ax.set_title(main_title, fontsize=8)
        
def plot_unit_rul(test_sequences, test_labels, endpoint_name, labels_scaler, unit_span, unit, title):
    # Get the testing sequences and labels for this engine unit:
    nb_sequences = unit_span[unit - 1]
    start = sum(unit_span[:unit - 1])
    end = start + nb_sequences
    unit_sequences = test_sequences[start:end]
    unit_rul = test_labels[start:end]
    
    if unit_rul.shape[0] == 0:
        return None

    # Invoke the endpoint with these testing sequences:
    payload = str(bytearray(unit_sequences))
    sm_runtime_client = boto3.client('sagemaker-runtime')
    response = sm_runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=payload,
        ContentType='application/json'
    )
    results = eval(response['Body'].read().decode('utf-8'))

    # Use the scaler that was applied on the initial training labels to 
    # inverse the transformation and be back in the original referential:
    df = pd.DataFrame(labels_scaler.inverse_transform(np.array(results).reshape(-1, 1)))
    df.columns = ['predictions']
    df['ground_truth'] = pd.DataFrame(labels_scaler.inverse_transform(unit_rul.reshape(-1, 1)))

    # Plot the predicted RUL versus the ground truth for this unit
    ax = df['ground_truth'].plot(linestyle='-', linewidth=5, alpha=0.3, label='Piecewise RUL')
    df['predictions'].plot(linestyle='--', color='#CC0000', ax=ax, label='Predicted RUL')
    
    ax.set_ylim(0,155)
    ax.set_title(title)
    
    plt.legend()
    
    return ax        
        
        
        
        
        
        
        
        
        
        
        
def get_unit_predictions(unit_id, test_sequences, test_labels, unit_span, model):
    X_test = mx.nd.array(test_sequences)
    y_test = mx.nd.array(test_labels)

    test_dataset = G.data.dataset.ArrayDataset(X_test, y_test)

    unit_to_test = unit_id
    seq_start = sum(unit_span[0:unit_to_test - 1])
    seq_end = sum(unit_span[0:unit_to_test])

    test_dataset = G.data.dataset.ArrayDataset(X_test[seq_start:seq_end], y_test[seq_start:seq_end])
    
    predictions = []
    original_rul = []

    for sample in test_dataset:
        data = sample[0]
        data = data.reshape(-1, data.shape[0], data.shape[1])
        label = sample[1]

        yhat = model(data)
        predictions.append(yhat.asnumpy()[0][0])
        original_rul.append(label.asnumpy()[0])

    return original_rul, predictions

def plot_unit_predictions(unit, predictions, original_rul, label_scaler):
    rescaled_original_rul = label_scaler.inverse_transform(np.array(original_rul).reshape(-1, 1))
    rescaled_predictions = label_scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    rmse = label_scaler.inverse_transform(np.mean((np.array(original_rul) - np.array(predictions))**2).reshape(-1, 1))

    plt.plot(rescaled_original_rul, label='Groundtruth')
    plt.plot(rescaled_predictions, label='Predictions')

    plt.ylabel('Remaining useful life')
    plt.legend()
    plt.ylim(0, 160)
    plt.title('Engine unit {} - RMSE: {:.2f}'.format(unit, rmse[0][0]))
    
def get_tuner_results(tuning_job_name):
    region = boto3.Session().region_name
    sage_client = boto3.Session().client('sagemaker')

    tuning_job_result = sage_client.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=tuning_job_name)

    status = tuning_job_result['HyperParameterTuningJobStatus']
    if status != 'Completed':
        print('Reminder: the tuning job has not been completed.')

    job_count = tuning_job_result['TrainingJobStatusCounters']['Completed']
    print("{} training jobs have completed".format(job_count))

    is_minimize = (tuning_job_result['HyperParameterTuningJobConfig']['HyperParameterTuningJobObjective']['Type'] != 'Maximize')
    objective_name = tuning_job_result['HyperParameterTuningJobConfig']['HyperParameterTuningJobObjective']['MetricName']
    jobs_df = sagemaker.HyperparameterTuningJobAnalytics(tuning_job_name).dataframe()
    df = pd.DataFrame()
    if len(jobs_df) > 0:
        df = jobs_df[jobs_df['FinalObjectiveValue'] > -float('inf')]
        if len(df) > 0:
            df = df.sort_values(by='FinalObjectiveValue', ascending=is_minimize)
            print("Number of training jobs with valid objective: %d" % len(df))
            print({"lowest":min(df['FinalObjectiveValue']),"highest": max(df['FinalObjectiveValue'])})
            pd.set_option('display.max_colwidth', None)  # Don't truncate TrainingJobName        
        else:
            print("No training jobs have reported valid results yet.")

    return df