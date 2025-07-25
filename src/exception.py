import sys
import logging
from src.logger import logging
def error_message_details(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    if exc_tb is None:
        error_message=f"Error message: {str(error)}"
    else:
        file_name=exc_tb.tb_frame.f_code.co_filename
        error_message="Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name,exc_tb.tb_lineno,str(error))
    return error_messagel
   
class CustomException(Exception):
        def __init__(self,error_message,error_details:sys):
            super().__init__(error_message)
            self.error_message=error_message_details(error_message,error_details)
            
            
        def __str__(self):
            return self.error_message

