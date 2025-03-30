import matplotlib.pyplot as plt
import json

# this is a simple script for visualizing the training stats, we open our json stats file, then just plot

with open("gan/models/230325-172325/training_stats.json", 'r') as f:
    training_stats = json.load(f)

# 2 rows, 1 col per subplot, ax1 and ax2 are the subplots, the overall size is 10x8.
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(training_stats["epoch"], training_stats["g_loss"])
ax1.set_title('Generator Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')

ax2.plot(training_stats["epoch"], training_stats["d_loss"])
ax2.set_title('Discriminator Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')

# sets the padding so there is no overlap between the text of the bottom one and the top one.
plt.tight_layout()

plt.show()