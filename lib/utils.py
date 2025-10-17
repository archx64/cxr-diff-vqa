import logging, colorlog
from logging.handlers import RotatingFileHandler
from torch.utils.data import DataLoader
import torch


RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
COLOR_END = "\033[0m"


def collate(batch):
    # simple default collator (images are same size from transforms)
    out = {k: [] for k in batch[0].keys()}
    for b in batch:
        for k,v in b.items():
            out[k].append(v)
    out["img_cur"] = torch.stack(out["img_cur"])
    out["img_ref"] = torch.stack(out["img_ref"])
    out["answer_id"] = torch.stack(out["answer_id"])
    return out

def make_loader(ds, bs, shuffle):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=0, pin_memory=True, collate_fn=collate)


def setup_logging(log_file: str):
    # add a custom 'SUCCESS' log level
    SUCCESS_LEVEL_NUM = 25
    logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")

    def success(self, message, *args, **kws):
        if self.isEnabledFor(SUCCESS_LEVEL_NUM):
            self._log(SUCCESS_LEVEL_NUM, message, args, **kws)

    logging.Logger.success = success

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.propagate = True

    # avoid adding handlers if they already exist
    if logger.hasHandlers():
        logger.handlers.clear()

    # formatter for the console with colors
    console_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        log_colors={
            "DEBUG": "white",
            "INFO": "cyan",
            "SUCCESS": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_white, bg_red",
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
        log_file,
        maxBytes=10485760,  # 10 * 1024 * 1024 bytes = 10MB
        backupCount=5,
        mode="a",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger
