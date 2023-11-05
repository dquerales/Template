from datetime import datetime

def logger(message = str):
    sttime = datetime.now().strftime('[%Y-%m-%d] %H:%M:%S - ')
    with open('../src/changelog.log', 'a') as logfile:
        logfile.write(f'{sttime}{message}\n')

def log_data_status(dataframe):
    df = dataframe.copy()
    logger(f'This dataframe has {df.shape[0]} rows and {df.shape[1]} columns.')
    