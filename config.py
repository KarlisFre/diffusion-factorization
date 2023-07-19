from diffusion.categorical_diffusion import CategoricalDiffusion
from diffusion.relaxed_categorical import RelaxedCategoricalDiffusion

n_classes = 2

# switch these if you want to use Discrete Catwegorical distribution. Currenly using Relaxed
#dist = CategoricalDiffusion()
dist = RelaxedCategoricalDiffusion(n_classes)

label_smoothing = 0.01
batch_size = 128
dropout_rate = 0.1
hidden_maps = 48 * 8
in_maps = 13