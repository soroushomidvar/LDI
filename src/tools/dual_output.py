import sys


class dual_output:
    def __init__(self, file_path):
        self.file_path = file_path
        self.console = sys.stdout  # Save the original standard output

    def write(self, message):
        # Write to the console
        self.console.write(message)
        # Write to the file
        with open(self.file_path, 'a') as file:
            file.write(message)

    def flush(self):
        self.console.flush()
