import logging, colorlog
from logging.handlers import RotatingFileHandler

LOG_DIR = "logs"
CHILD_DIR = "p10"
GCS_PARENT_DIR = "files"
LOCAL_PARENT_DIR = "D:/MIMIC-CXR-JPG"

LOG_FILE = f"{LOG_DIR}/{CHILD_DIR}.log"
BILLING_PROJECT_ID = "summer-branch-251417"
BUCKET_NAME = "mimic-cxr-jpg-2.1.0.physionet.org"
GCS_FOLDER = f"{GCS_PARENT_DIR}/{CHILD_DIR}/"  # os.path.join(GCS_PARENT_DIR, CHILD_DIR)
LOCAL_DESTINATION = (
    f"{LOCAL_PARENT_DIR}/{CHILD_DIR}/"  # os.path.join(LOCAL_PARENT_DIR, CHILD_DIR)
)

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
COLOR_END = "\033[0m"

print(GCS_FOLDER)


def setup_logging():
    # add a custom 'SUCCESS' log level
    SUCCESS_LEVEL_NUM = 25
    logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")

    def success(self, message, *args, **kws):
        if self.isEnabledFor(SUCCESS_LEVEL_NUM):
            self._log(SUCCESS_LEVEL_NUM, message, args, **kws)

    logging.Logger.success = success

    logger = logging.getLogger("MIMIC-CXR-JPG-DOWNLOADER")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # avoid adding handlers if they already exist
    if logger.hasHandlers():
        logger.handlers.clear()

    # formatter for the console with colors
    console_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "blue",
            "SUCCESS": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )

    # formatter for the file (plain text)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # console handler
    console_handler = colorlog.StreamHandler()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # this will create up to 5 backup files, each 10MB in size.
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=10485760,  # 10 * 1024 * 1024 bytes = 10MB
        backupCount=5,
        mode="a",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger


logger = setup_logging()
