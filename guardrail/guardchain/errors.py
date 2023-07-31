class OutputParserException(Exception):
    """Exception raised by output parsers to signify a parsing error.

    This exception is specifically used for handling parsing errors encountered while processing
    the output from a tool. It allows differentiation between parsing errors and other code or
    execution errors that may occur within the output parser. When an OutputParserException is
    raised, it can be caught and handled in a way to address the parsing error, while other errors
    will be raised as usual.
    """

    pass


class ToolRunningError(Exception):
    """Exception raised when a tool fails to run.

    This exception is raised to indicate that an error occurred while attempting to run a tool. It
    can be caught and handled by the application to address the specific failure in running the tool.
    The `message` attribute of this exception contains information about the reason for the failure.
    """

    def __init__(self, message):
        self.message = message
