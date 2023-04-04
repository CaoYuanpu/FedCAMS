import pickle

res = open('./results/cifar10_mlp_fedavg_llr[1.0]_glr[1.0]_eps[0.0]_le[3]_bs[20]_iid[1]_mi[0.001]_frac[0.1].pkl', 'rb')

res = pickle.load(res)

for i, acc in enumerate(res[2]):
    print(i+1, acc)

