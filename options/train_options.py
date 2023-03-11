from argparse import ArgumentParser

class TrainOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):

		self.parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training')
		self.parser.add_argument('--workers', default=4, type=int, help='Number of train dataloader workers')

		self.parser.add_argument('--learning_rate', default=0.5, type=float, help='Optimizer learning rate')

		self.parser.add_argument('--l2_lambda', default=1.0, type=float, help='l2 loss')
		self.parser.add_argument('--cos_lambda', default=1.0, type=float, help='cos loss')

		self.parser.add_argument('--checkpoint_path', default='checkpoints', type=str, help='Path to StyleCLIPModel model checkpoint')
		self.parser.add_argument('--classname', type=str, default='ffhq', help="which specific domain for training")
		self.parser.add_argument('--print_interval', default=1000, type=int, help='Interval for printing loss values during training')
		self.parser.add_argument('--val_interval', default=5000, type=int, help='Validation interval')
		self.parser.add_argument('--save_interval', default=10000, type=int, help='Model checkpoint interval')

	def parse(self):
		opts = self.parser.parse_args()
		return opts