import mlatom as ml
import matplotlib.pyplot as plt
import numpy as np
import os
from joblib import Parallel, delayed

traj_path = '/export/home/yifanhou/project/ADL/xtbstar_b3lyp_d4/md/calculations'

if not os.path.exists('plot_roaming'):
    os.mkdir('plot_roaming')

def bond_length(molecule):
    atomic_orders = [(58, 70), (59, 73)]
    distances = []
    for atoms_1 in atomic_orders:
        ii, jj = atoms_1
        dist_1 = np.sqrt(np.sum(np.square(molecule[ii].xyz_coordinates - molecule[jj].xyz_coordinates)))
        distances.append(dist_1)
    return min(distances), max(distances), distances

def check_traj(forward_traj, backward_traj):
    forward_label = backward_label = 'ts'
    forward_time = backward_time = -1
    forward_distances_list = []
    backward_distances_list = []

    for imol in range(len(forward_traj)):
        min_dist, max_dist, distances = bond_length(forward_traj[imol])
        forward_distances_list.append(distances)
        if min_dist > 5.0:
            forward_label = 'reactant'
            break
        if max_dist < 1.6:
            forward_label = 'product'
            break
        if (min_dist > 2.60 or max_dist < 1.86) and forward_time < 0:
            forward_time = imol * 0.5

    for imol in range(len(backward_traj)):
        min_dist, max_dist, distances = bond_length(backward_traj[imol])
        backward_distances_list.append(distances)
        if min_dist > 5.0:
            backward_label = 'reactant'
            break
        if max_dist < 1.6:
            backward_label = 'product'
            break
        if (min_dist > 2.60 or max_dist < 1.86) and backward_time < 0:
            backward_time = imol * 0.5

    gap = None
    if (forward_label == 'product' and backward_label == 'reactant') or (forward_label == 'reactant' and backward_label == 'product'):
        distances_list = forward_distances_list if forward_label == 'product' else backward_distances_list
        bond1_formed = bond2_formed = False

        for istep in range(len(distances_list)):
            distances = distances_list[istep]
            if min(distances) < 1.6 and not bond1_formed:
                gap = istep * 0.5
                bond1_formed = True
            if max(distances) < 1.6 and not bond2_formed:
                gap = istep * 0.5 - gap
                bond2_formed = True
                break

    return forward_label, backward_label, forward_time + backward_time, gap

def get_bond_length(mol):
    d1 = [mol.internuclear_distance(ii, 70) for ii in range(60)]
    d2 = [mol.internuclear_distance(ii, 73) for ii in range(60)]
    return min(d1), min(d2)

def read_traj(itraj):
    if not os.path.exists(f'{traj_path}/forward_temp298_{itraj}.json') or not os.path.exists(f'{traj_path}/backward_temp298_{itraj}.json'):
        return None, None

    try:
        if not os.path.exists(f"plot_roaming/list_d1_{itraj}.npy"):
            forward_moldb = ml.data.molecular_database.load(f'{traj_path}/forward_temp298_{itraj}.json',format='json')
            backward_moldb = ml.data.molecular_database.load(f'{traj_path}/backward_temp298_{itraj}.json',format='json')

            d1_list = []
            d2_list = []
            for mol in forward_moldb:
                d1, d2 = get_bond_length(mol)
                d1_list.append(d1)
                d2_list.append(d2)

            d1_list = d1_list[::-1]
            d2_list = d2_list[::-1]

            for mol in backward_moldb[1:]:
                d1, d2 = get_bond_length(mol)
                d1_list.append(d1)
                d2_list.append(d2)

            np.save(f'plot_roaming/list_d1_{itraj}.npy', d1_list)
            np.save(f'plot_roaming/list_d2_{itraj}.npy', d2_list)
        else:
            d1_list = np.load(f'plot_roaming/list_d1_{itraj}.npy')
            d2_list = np.load(f'plot_roaming/list_d2_{itraj}.npy')

        return d1_list, d2_list
    except Exception as e:
        print(f"Error processing trajectory {itraj}: {e}")
        return None, None

# Main script
icount = 0
iroaming1 = 0
iroaming2 = 0
iroaming3 = 0
iroaming4 = 0
iroaming5 = 0
iroaming6 = 0
iroaming7 = 0
iroaming8 = 0
iroaming9 = 0
iroaming10 = 0
roaming_traj = []
fig, ax = plt.subplots()

with open('analyze_traj.log', 'r') as f:
    lines = f.readlines()

line_index = 0
for ii in range(1, 1001):
    d1_list, d2_list = read_traj(ii)
    if d1_list is None or d2_list is None:
        continue

    line = [each.strip() for each in lines[line_index].strip().split(',')]
    line_index += 1

    if (line[0] == 'reactant' and line[1] == 'product') or (line[0] == 'product' and line[1] == 'reactant'):
        icount += 1
        traj1 = d1_list if line[0] == 'reactant' else d1_list[::-1]
        traj2 = d2_list if line[0] == 'reactant' else d2_list[::-1]

        time_list = np.linspace(-1500, 1500, 6001)
        ax.plot(time_list, traj1, linewidth=0.5)
        ax.plot(time_list, traj2, linewidth=0.5)
        
        if traj1[0] < 5.0 or traj2[0] < 5.0:
            iroaming1 += 1
            roaming_traj.append(ii)
        if traj1[0] < 4.5 and traj2[0] < 4.5:
            iroaming2 += 1
            roaming_traj.append(ii)
        if traj1[0] < 4.75 and traj2[0] < 4.75:
            iroaming3 += 1
            roaming_traj.append(ii)
        if traj1[0] < 5.0 and traj2[0] < 5.0:
            iroaming4 += 1
            roaming_traj.append(ii)
        if traj1[0] < 5.25 and traj2[0] < 5.25:
            iroaming5 += 1
            roaming_traj.append(ii)
        if traj1[0] < 5.50 and traj2[0] < 5.50:
            iroaming6 += 1
            roaming_traj.append(ii)
        if traj1[0] < 5.75 and traj2[0] < 5.75:
            iroaming7 += 1
            roaming_traj.append(ii)
        if traj1[0] < 6.0 and traj2[0] < 6.0:
            iroaming8 += 1
            roaming_traj.append(ii)
        if traj1[0] < 6.25 and traj2[0] < 6.25:
            iroaming9 += 1
            roaming_traj.append(ii)
        if traj1[0] < 6.5 and traj2[0] < 6.5:
            iroaming10 += 1
            roaming_traj.append(ii)

print(f"roaming/reactive(traj1[0] < 5.00  or traj2[0] < 5.00): {iroaming1/icount}({iroaming1}/{icount})")
print(f"roaming/reactive(traj1[0] < 4.50 and traj2[0] < 4.50): {iroaming2/icount}({iroaming2}/{icount})")
print(f"roaming/reactive(traj1[0] < 4.75 and traj2[0] < 4.75): {iroaming3/icount}({iroaming3}/{icount})")
print(f"roaming/reactive(traj1[0] < 5.00 and traj2[0] < 5.00): {iroaming4/icount}({iroaming4}/{icount})")
print(f"roaming/reactive(traj1[0] < 5.25 and traj2[0] < 5.25): {iroaming5/icount}({iroaming5}/{icount})")
print(f"roaming/reactive(traj1[0] < 5.50 and traj2[0] < 5.50): {iroaming6/icount}({iroaming6}/{icount})")
print(f"roaming/reactive(traj1[0] < 5.75 and traj2[0] < 5.75): {iroaming7/icount}({iroaming7}/{icount})")
print(f"roaming/reactive(traj1[0] < 6.00 and traj2[0] < 6.00): {iroaming8/icount}({iroaming8}/{icount})")
print(f"roaming/reactive(traj1[0] < 6.25 and traj2[0] < 6.25): {iroaming9/icount}({iroaming9}/{icount})")
print(f"roaming/reactive(traj1[0] < 6.50 and traj2[0] < 6.50): {iroaming10/icount}({iroaming10}/{icount})")

np.save('plot_roaming/roaming_traj.npy', roaming_traj)
ax.plot([-1500, 1500], [2.227, 2.227], color='C0', linestyle=':')
plt.xlabel('Time (fs)')
plt.ylabel('d1 or d2 (Angstrom)')
plt.ylim(1, 20)
plt.xlim(-1500, 1500)
plt.savefig('plot_roaming/roaming.png', dpi=300)