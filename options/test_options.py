from argparse import ArgumentParser

class TestOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		# arguments for inference script
		
		self.parser.add_argument('--batch_size', default=1, type=int, help='Batch size for inference')
		self.parser.add_argument('--workers', default=4, type=int, help='Number of test dataloader workers')
		
		self.parser.add_argument('--stylegan_weights', default='models/pretrained_models/stylegan2-ffhq-config-f.pt', type=str, help='Path to StyleGAN model weights')
		self.parser.add_argument('--stylegan_size', default=1024, type=int)
		
		self.parser.add_argument("--threshold", type=int, default=0.03)
		self.parser.add_argument("--checkpoint_path", type=str, default='checkpoints/net_face.pth')
		self.parser.add_argument("--save_dir", type=str, default='output')
		self.parser.add_argument("--num_all", type=int, default=20)
		
		self.parser.add_argument("--target", type=str, required=True, help='Specify the target attributes to be edited')

	def parse(self):
		opts = self.parser.parse_args()
		return opts