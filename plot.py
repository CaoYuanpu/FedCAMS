import pandas as pd
import matplotlib.pyplot as plt
import pickle

lw = 3
step = 60
fs = 20
mfs = 20
sfs = 15


central_mlp = pickle.load(open('./saved/cifar10_mlp_fedavg_llr[0.01]_bs[1.0]_central.pkl', 'rb'))[1]
central_mlp_lora = pickle.load(open('./saved/cifar10_mlp_lora_fedavg_llr[0.1]_bs[1.0]_central.pkl', 'rb'))[1]
central_mlp_lora_reset10 = pickle.load(open('./saved/cifar10_mlp_lora_fedavg_llr[0.1]_bs[1.0]_central_reset10.pkl', 'rb'))[1]
central_mlp_lora_reset1 = pickle.load(open('./saved/cifar10_mlp_lora_fedavg_llr[0.1]_bs[1.0]_central_reset1.pkl', 'rb'))[1]

fig = plt.figure(figsize=(10, 8))

ax_acc = plt.subplot()
ax_acc.plot(central_mlp, linewidth=lw, label=r"base")
ax_acc.plot(central_mlp_lora, linewidth=lw, label=r"LoRA")
ax_acc.plot(central_mlp_lora_reset1, linewidth=lw, label=r"LoRA reset per 1 epoch")
ax_acc.plot(central_mlp_lora_reset10, linewidth=lw, label=r"LoRA reset per 10 epochs")

ax_acc.set_xlabel('# Epochs', fontsize=fs)
ax_acc.set_ylabel('Test Accuracy', fontsize=fs)

# ax_acc.set_ylim([0., 0.5])
ax_acc.legend()
ax_acc.legend(fontsize=sfs)
# plt.title('FashionMNIST: 0.1 fedprox')

plt.grid()
plt.xticks(size = sfs)
plt.yticks(size = sfs)
# plt.show()

plt.savefig('output/centralized.png', bbox_inches='tight', dpi=600)