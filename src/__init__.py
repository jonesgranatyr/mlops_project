import logging
import dagshub
from dotenv import load_dotenv

load_dotenv()

# Initialize DagsHub with credentials
dagshub.init(
    repo_owner='dennymarcels',
    repo_name='mlops_project'
)

# Configure the logging strategy
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler()
    ]
)
