import logging
from datetime import datetime
from pkg_resources import resource_filename

# Setup file logging
log_datetime = datetime.now().isoformat()
log_path = resource_filename(__name__, "../../logs")
file_log_handler = logging.FileHandler(f"{log_path}/{log_datetime}_bert_model.log")
logger = logging.getLogger(__name__)
stderr_log_handler = logging.StreamHandler()

# nice output format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_log_handler.setFormatter(formatter)
stderr_log_handler.setFormatter(formatter)

logger.addHandler(stderr_log_handler)
logger.addHandler(file_log_handler)
logger.setLevel("INFO")
