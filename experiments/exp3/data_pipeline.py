import requests
from pathlib import Path

# Function to dowload dataset from dropbox
def download_sva_dataset():
    url = 'https://www.dropbox.com/scl/fi/fph8xsz683bne62rpxcjs/agr_50_mostcommon_10K.tsv.gz?e=3&rlkey=t7tgb87xd3073qzv7fghu7vm6&dl=1'
    file_name = 'agr_50_mostcommon_10K.tsv.gz'
    
    # Get route
    base = Path(__file__).resolve()
    route = base.parent.parent.parent / 'data' / 'SVADataset'
    
    route.mkdir(parents=True, exist_ok=True)
    
    route_file = route / file_name
    
    if route_file.exists():
        print(f'The file {file_name} already exists.')
        return
    
    try:
        request = requests.get(url, stream=True)
        request.raise_for_status() # Check if the data was download
        
        with open(route_file, 'wb') as file:
            # Write file in chunks from 8 KB
            for chunk in request.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    
        print(f'The file {file_name} was downloaded successful.')
        
    # Downloading error exception
    except requests.exceptions.RequestException as e:
        print(f'Network error while downloading \'{file_name}\': {e}')