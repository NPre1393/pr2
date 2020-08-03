class Algorithm:
    def __init__(self, args):
        self.args = args
        self.train_epochs = self.args["train_epochs"]
        self.learning_rate = self.args["learning_rate"]
        self.batch_size = self.args["batch_size"]

    def __repr__(self):
        repr = 'Algorithm Variables\n'\
            'Train epochs = {}\n'\
            'Learning rate = {}\n'\
            'Batch size = {}\n'\
            .format(self.train_epochs, self.learning_rate, self.batch_size)

        return repr

    def run(self):
        return None

    def parse_params(self, args):
        return None