import sys


class NetworkSecurityException(Exception):
    """
    Custom exception class for the Network Security project.

    This exception adds extra debugging information to the standard Python
    exception, such as the file name and line number where the error occurred.
    """

    def __init__(self, error_message, error_details: sys):
        """
        Initialize the custom exception.

        Args:
            error_message: The original error message or exception object.
            error_details: Usually the sys module, used to extract exception details.
        """

        # Store the original error message
        self.error_message = error_message

        # Extract exception information:
        # exc_type  -> type of exception
        # exc_value -> exception message/value
        # exc_tb    -> traceback object
        _, _, exc_tb = error_details.exc_info()

        # Get the line number where the exception occurred
        self.lineno = exc_tb.tb_lineno

        # Get the file name where the exception occurred
        self.file_name = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        """
        Return a formatted error message when the exception is printed.
        """

        return "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
            self.file_name,
            self.lineno,
            str(self.error_message)
        )