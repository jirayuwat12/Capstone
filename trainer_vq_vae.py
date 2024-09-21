from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import pandas as pd

from T2M_GPT_lightning.models.vqvae import HumanVQVAEWrapper
from T2M_GPT_lightning.dataset.toy_vq_vae_dataset import ToyDataset
from T2M_GPT.options import option_vq

from lightning.pytorch import Trainer

# Get the arguments
args_parser = option_vq.get_args_parser()
args_parser.add_argument("--is_toy", action="store_true", help="whether to run the model on toy data")

# Parse the arguments
args = args_parser.parse_args()
args.nb_joints = 150
args.recon_loss = "l1_smooth"
args.window_size = 64

# Initialize the model
model = HumanVQVAEWrapper(args)

# Initialize the dataset
if args.is_toy:
    train_dataset = ToyDataset(skels_data_path="./data/toy_data/train.skels", frame_size=100, window_size=args.window_size)
    test_dataset = ToyDataset(skels_data_path="./data/toy_data/train.skels", frame_size=100, window_size=args.window_size)
else:
    assert False, "Not implemented yet"

# Initialize the dataloaders
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)

# Initialize the trainer
trainer = Trainer(log_every_n_steps=10, max_epochs=2000)

# Train the model
trainer.fit(model, train_loader, test_loader)

# Save the model
trainer.save_checkpoint("vq_vae_model_loss=.pth")

# Get logging path
logging_path = model.logger.log_dir
# Read csv file
df = pd.read_csv(logging_path + "/metrics.csv")
# Plot the loss
plt.title("Train Loss")
plt.plot(df.loc[df["train_loss"].notnull(), "train_loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.show()