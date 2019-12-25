from logging import Formatter, handlers, StreamHandler, getLogger

#定数宣言
LOG_SET_LEVEL = 'DEBUG'         #対象とするログレベル（ファイルにも使用する）
STREAM_LOG_SET_LEVEL = 'INFO'   #標準出力で表示するログレベル

def setLogger(logger):
    logger.setLevel(LOG_SET_LEVEL)
    formatter = Formatter("[%(asctime)s] [%(name)s] [%(lineno)d] [%(levelname)s] %(message)s")

    # stdout
    handler = StreamHandler()
    handler.setLevel(STREAM_LOG_SET_LEVEL)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # file
    handler = handlers.RotatingFileHandler(filename = 'logfile/logger.log')
    handler.setLevel(LOG_SET_LEVEL)
    handler.setFormatter(formatter)
    logger.addHandler(handler)