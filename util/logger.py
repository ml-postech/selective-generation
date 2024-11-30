import os, sys

class Logger(object):
    def __init__(self, fn):
        self.stdout = sys.stdout
        self.file = open(fn, 'w')

    def write(self, msg):
        self.stdout.write(msg)
        self.file.write(msg)

    def flush(self):
        self.stdout.flush()
        self.file.flush()
        
