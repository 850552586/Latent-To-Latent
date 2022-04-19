from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
from utils.latent_utils import get_latent
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.utils.data

class Img_Latent_Dataset(Dataset):

	def __init__(self, root, transform=None):
		self.paths = sorted(data_utils.make_dataset(root))
		self.transform = transform

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, index):
		from_path = self.paths[index]
		from_im = Image.open(from_path).convert('RGB')
		if self.transform:
			from_im = self.transform(from_im)
		return from_im

def load_data(root,batch_size):
	inference_transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
	dataset = Img_Latent_Dataset(root,transform=inference_transform)
	train_size = int(0.5 * len(dataset))
	eval_size = int(0.1 * len(dataset))
	train_dataset, eval_dataset,_ = torch.utils.data.random_split(dataset, [train_size, eval_size,int(0.4 * len(dataset))])
	train_dataloader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=int(0),
                            drop_last=False)
	eval_dataloader = DataLoader(eval_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=int(0),
                            drop_last=False)
	return train_dataloader,eval_dataloader