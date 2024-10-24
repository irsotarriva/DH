''' Logging configuration for any python project.'''
import json
import logging
import logging.config
import logging.handlers
import pathlib

log : logging.Logger = logging.getLogger(__name__)

class ColouredFormatter(logging.Formatter):
    '''
    @summary: A custom logging formatter that outputs log records with color
    '''
    RED = '\x1b[31;20m'
    GREEN = '\x1b[32;20m'
    YELLOW = '\x1b[33;20m'
    BLUE = '\x1b[34;20m'
    MAGENTA = '\x1b[35;20m'
    CYAN = '\x1b[36;20m'
    WHITE = '\x1b[37;20m'
    BOLD_RED = '\x1b[31;1m'
    RESET = '\033[0m'
    format_str = '[%(asctime)s] [%(levelname)s] [%(module)s - %(funcName)s:%(lineno)d] %(message)s'

    COLORS = {
        logging.DEBUG: CYAN + format_str + RESET,
        logging.INFO: WHITE + format_str + RESET,
        logging.WARNING: YELLOW + format_str + RESET,
        logging.ERROR: RED + format_str + RESET,
        logging.CRITICAL: BOLD_RED + format_str + RESET
    }


    def __init__(self, *, fmt_keys: dict[str, str]| None = None):
        '''
        @summary: Initialize the formatter
        @precondition: None
        @postcondition: The formatter is initialized
        @param fmt_keys: A dictionary of keys to format strings
        @type fmt_keys: dict[str, str]
        @return: None
        @rtype: None
        '''
        super().__init__(None)
        self.fmt_keys = fmt_keys if fmt_keys is not None else {}

    #@override
    def format(self, record: logging.LogRecord) -> str:
        '''
        @summary: Format a log record with color
        @precondition: None
        @postcondition: The log record is formatted with color
        @param record: The log record to format
        @type record: logging.LogRecord
        @return: The log record as a colored string
        @rtype: str
        '''
        color = self.COLORS.get(record.levelno, self.WHITE + self.format_str + self.RESET)
        formatter = logging.Formatter(color)
        return formatter.format(record)

class JsonFormatter(logging.Formatter):
    '''
    @summary: A custom logging formatter that outputs log records as JSON
    '''
    def __init__(self, *, fmt_keys: dict[str, str]| None = None):
        '''
        @summary: Initialize the formatter
        @precondition: None
        @postcondition: The formatter is initialized
        @param fmt_keys: A dictionary of keys to format strings
        @type fmt_keys: dict[str, str]
        @return: None
        @rtype: None
        '''
        super().__init__(None)
        self.fmt_keys = fmt_keys if fmt_keys is not None else {}

    #@override
    def format(self, record: logging.LogRecord) -> str:
        '''
        @summary: Format a log record as a JSON string
        @precondition: None
        @postcondition: The log record is formatted as a JSON string
        @param record: The log record to format
        @type record: logging.LogRecord
        @return: The log record as a JSON string
        @rtype: str
        '''
        record_dict: dict[str, str] = record.__dict__.copy()
        for key, fmt in self.fmt_keys.items():
            record_dict[key] = fmt.format(record_dict[key])
        return json.dumps(record_dict)

def setup_logging():
    '''
    @summary: Setup the logging configuration
    @precondition: log_config.json must be in the same directory as this file
    @postcondition: The logging configuration is setup
    @return: None
    @rtype: None
    @raise FileNotFoundError: If log_config.json is not found
    '''
    config: dict[str, str] = {}
    with open(pathlib.Path(__file__).parent / 'log_config.json', encoding='utf-8') as f:
        if not f:
            raise FileNotFoundError('log_config.json not found')
        config = json.load(f)
    logging.config.dictConfig(config)
