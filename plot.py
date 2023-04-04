import pandas as pd
import matplotlib.pyplot as plt
import pickle

lw = 3
step = 60
fs = 30
mfs = 25
sfs = 20



central_mlp = pickle.load(open('./saved/cifar10_mlp_fedavg_llr[0.01]_bs[1.0]_central.pkl', 'rb'))[1]
central_mlp_lora = pickle.load(open('./saved/cifar10_mlp_lora_fedavg_llr[0.1]_bs[1.0]_central_lora.pkl', 'rb'))[1]
central_mlp_lora_reset = pickle.load(open('./saved/cifar10_mlp_lora_fedavg_llr[0.1]_bs[1.0]_central_reset.pkl', 'rb'))[1]
central_mlp_lora_reset1 = pickle.load(open('./saved/cifar10_mlp_lora_fedavg_llr[0.1]_bs[1.0]_central_reset1.pkl', 'rb'))[1]
central_mlp_lora_reset5 = pickle.load(open('./saved/cifar10_mlp_lora_fedavg_llr[0.1]_bs[1.0]_central_reset5.pkl', 'rb'))[1]

fig = plt.figure(figsize=(10, 8))

ax_acc = plt.subplot()
ax_acc.plot(central_mlp, linewidth=lw, label=r"mlp")
ax_acc.plot(central_mlp_lora, linewidth=lw, label=r"$mlp_{loRA}$")
ax_acc.plot(central_mlp_lora_reset, linewidth=lw, label=r"$mlp_{loRA}$-10")
ax_acc.plot(central_mlp_lora_reset5, linewidth=lw, label=r"$mlp_{loRA}$-5")
ax_acc.plot(central_mlp_lora_reset1, linewidth=lw, label=r"$mlp_{loRA}$-1")

ax_acc.set_xlabel('# Communication Rounds', fontsize=fs)
ax_acc.set_ylabel('Test Accuracy', fontsize=fs)
# ax_acc.set_ylim([0., 0.5])
ax_acc.legend()
ax_acc.legend(fontsize=sfs)
# plt.title('FashionMNIST: 0.1 fedprox')
plt.grid()
plt.xticks(size = sfs)
plt.yticks(size = sfs)
plt.show()
# plt.savefig('output/norm_cifar10_alpha0.04_w10h8.png', bbox_inches='tight', dpi=600)