import requests
from pathlib import Path

# Function to download data from original GitHub
def download_seteval_data(task_name):
    # See if data/SentEval exist
    base = Path(__file__).resolve()
    senteval_route = base.parent.parent.parent / 'data' / 'SentEval'
    
    senteval_route.mkdir(parents=True, exist_ok=True) # Create directories if they don't exist
    
    # See if task data exist
    senteval_route_task = senteval_route / f'{task_name}.txt'
    
    if senteval_route_task.exists():
        print(f'The file {task_name}.txt already exists.')
        return
    
    url = f'https://raw.githubusercontent.com/facebookresearch/SentEval/main/data/probing/{task_name}.txt'
    
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            
            with open(senteval_route_task, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192): # Chunks of 8KB
                    if chunk: f.write(chunk) # Get all data
                    
        print(f'The file {task_name}.txt was downloaded successful.')
    
    # Exception handling
    except requests.exceptions.HTTPError as e:
        print(f'Task \'{task_name}\' does not exist in SentEval.')

    except requests.exceptions.RequestException as e:
        print(f'Network error while downloading \'{task_name}\': {e}')
        