import boto3
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import clear_output
import time
import tabulate
import sagemaker

def get_training_metrics(training_job_name, train_metric_name, val_metric_name):
    """
    This function uses Amazon CloudWatch to extract the 
    training and validation metrics for a given training job.
    
    :param: training_job_name - string containing the ID of the training job 
                                to fetch the metrics for
    :param: train_metric_name - string to search for to extract the training metric
    :param: val_metric_name - string to search for to extract the validation metric
    :return: dataframe containing the metrics with for each step
    :rtype: dataframe
    """
    # Gets a Cloudwatch logs client:
    cwlogs_client = boto3.client('logs')

    # Each training has its log stored in a given log stream under the TrainingJobs logs group:
    response = cwlogs_client.describe_log_streams(
        logGroupName='/aws/sagemaker/TrainingJobs',
        logStreamNamePrefix=training_job_name,
    )
    training_log_stream_name = response['logStreams'][0]['logStreamName']

    # We can then get the events from this particular log stream:
    response = cwlogs_client.get_log_events(
        logGroupName='/aws/sagemaker/TrainingJobs',
        logStreamName=training_log_stream_name,
        startFromHead=True
    )

    # Process the event logs to get the detailed train and validation metrics:
    train_metric_list = []
    val_metric_list = []
    for event in response['events']:
        search_train_metric = re.search('{}([-+]?[0-9]*\.?[0-9]+)'.format(train_metric_name), event['message'])
        if (search_train_metric):
            current_train_metric = float(search_train_metric.group(1))
            train_metric_list.append(current_train_metric)

        search_val_metric = re.search('{}([-+]?[0-9]*\.?[0-9]+)'.format(val_metric_name), event['message'])
        if (search_val_metric):
            current_val_metric = float(search_val_metric.group(1))
            val_metric_list.append(current_val_metric)

    metrics_df = pd.concat([pd.Series(train_metric_list), pd.Series(val_metric_list)], axis='columns')
    metrics_df.columns = [train_metric_name, val_metric_name]

    return metrics_df

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig

def get_tuner_status(xgb_tuner_job_name):
    """
    This functions get the status of a hyperparameter job and prints a tabulated output.
    
    :param: xgb_tuner_job_name - a string containing the label of the hyperparameter tuning job to query.
    
    :return: status of the job (Completed or not)
    :rtype: string
    """
    sagemaker_client = boto3.Session().client('sagemaker')
    tuning_job_result = sagemaker_client.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=xgb_tuner_job_name)
    status = tuning_job_result['HyperParameterTuningJobStatus']

    # While the tuning job is in progress...
    jobs_headers = [
        'TrainingJobName', 'FinalObjectiveValue', 'eta', 
        'gamma', 'max_depth', 'subsample', 'colsample_bytree'
    ]
    while status == 'InProgress':
        # Clear the notebook output cell to update its content:
        clear_output()
        print('Tuning job in progress (status: {})'.format(status))

        # Check if at least one job started:
        job_count = tuning_job_result['TrainingJobStatusCounters']['Completed']
        if job_count > 0:
            print("{} training jobs are complete:".format(job_count))

            # Get the details about the different jobs launched by the hyperparameter tuning process
            jobs_df = sagemaker.HyperparameterTuningJobAnalytics(xgb_tuner_job_name).dataframe()

            # Sorts the dataframe and keep only the fields we are interested into:
            jobs_df = jobs_df.sort_values(by='TrainingStartTime', ascending=False)[jobs_headers]
            jobs_df = jobs_df[jobs_df['FinalObjectiveValue'] > -float('inf')]
            print(tabulate.tabulate(jobs_df, tablefmt='psql', headers=jobs_headers))

        else:
            print('No job completed yet.')

        time.sleep(30)

        tuning_job_result = sagemaker_client.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=xgb_tuner_job_name)
        status = tuning_job_result['HyperParameterTuningJobStatus']
        
    return status