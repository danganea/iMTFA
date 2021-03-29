class PrintNTimes():
    """
    Object that prints a message up to N times
     when it's "print" function is called or it is called
    """

    def __init__(self, N: int):
        """
        Initializes the internal counter of the class and the
        number of times to print the message as N.
        :type N: int
        """
        self.counter = 0
        self.N = N

    def print(self, msg):
        """
        Print the contents of the message if the counter is below N
        """
        if self.counter < self.N:
            print(msg)
            self.counter += 1

    def __call__(self, msg):
        self.print(msg)


class PrintOnce(PrintNTimes):
    def __init__(self):
        super().__init__(1)


def print_file(file_name):
    with open(file_name, 'r') as file:
        print(file.read())
