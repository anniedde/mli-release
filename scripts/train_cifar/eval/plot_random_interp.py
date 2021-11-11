import numpy as np
import json
import matplotlib.pyplot as plt

alpha_steps = np.linspace(0, 1, 20, endpoint=True)
with open("/usr/xtmp/CSPlus/VOLDNN/Annie/mli-release/random_interpolation_test/1577604-alpha20.json", "r") as read_file:
    interp_losses = json.load(read_file) 

interp_losses = interp_losses['loss']

fig = plt.figure()

plt.plot(alpha_steps, np.array(interp_losses), label='Interpolation Loss')

fpath = "/usr/xtmp/CSPlus/VOLDNN/Annie/mli-release/random_interpolation_test/plot_alpha-20"

plt.title("Redo vgg19, lr 0.01, no BN, SGD, init-final, outputScaled")
plt.tight_layout()
plt.savefig(fpath + ".png", dpi=300, bbox_inches = 'tight')
plt.clf()
plt.close()