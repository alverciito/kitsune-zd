# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import time
import logging as log

# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
log.basicConfig(level=log.INFO)
logging = log.getLogger(__name__)
logging.info(f'[+] Logging initialized at f{timestamp}.')
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
HASTE_VALUE = 1
BATCH_SIZE = 1
EXECUTION_BS = 512
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
