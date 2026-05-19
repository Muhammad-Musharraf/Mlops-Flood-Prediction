import logging
import os

LOGS_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOGS_DIR, "running_logs.log")

logging.basicConfig(
    format="[ %(asctime)s ] %(filename)s:%(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, mode='a'), 
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("MLOpsWorkflow")
def log_info(message):
    logger.info(message)
    
def log_warning(message):
    logger.warning(message)

def log_error(message):
    logger.error(message)

def log_debug(message):
    logger.debug(message)

if __name__ == "__main__":
    log_info("Logger initialized in logs/running_logs.log")