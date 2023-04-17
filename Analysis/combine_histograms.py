import numpy as np

"""Combining the incremental histograms into one large histogram"""

def combine_hists(hist1, hist2, name):
    """Combines the data in two histograms"""

    # Do not want method to function in place
    hist1 = np.copy(hist1)
    hist2 = np.copy(hist2)

    # Check if values of hist2 is found in hist1
    for i in range(len(hist2[0])):
        print(f"{name}: {i}/{len(hist2[0])-1}")
        value = hist2[0][i]
        if value in hist1[0]: # Check if value in hist2 found in hist1
            idx = np.where(hist1[0] == value)[0][0] # Get index in hist1 where the value is found
            hist1[1][idx] += hist2[1][i] # Update hist1
        else:

            hist1 = np.array([hist1[0].tolist() + [value], hist1[1].tolist() + [hist2[1][i]]])

    return hist1

# Combined set
element_type = np.load(f"Histograms/Combined/element_type.npy", allow_pickle=True)
num_heavy_atom_neighbors = np.load(f"Histograms/Combined/num_heavy_atom_neighbors.npy", allow_pickle=True)
num_h_neighbors = np.load(f"Histograms/Combined/num_h_neighbors.npy", allow_pickle=True)
formal_charge = np.load(f"Histograms/Combined/formal_charge.npy", allow_pickle=True)
in_a_ring = np.load(f"Histograms/Combined/in_a_ring.npy", allow_pickle=True)
is_aromatic = np.load(f"Histograms/Combined/is_aromatic.npy", allow_pickle=True)

bond_type = np.load(f"Histograms/Combined/bond_type.npy", allow_pickle=True)

num_heavy_atoms = np.load(f"Histograms/Combined/num_heavy_atoms.npy", allow_pickle=True)
total_num_atoms = np.load(f"Histograms/Combined/total_num_atoms.npy", allow_pickle=True)
molecular_weight = np.load(f"Histograms/Combined/molecular_weight.npy", allow_pickle=True)

sas = np.load(f"Histograms/Combined/sas.npy", allow_pickle=True)
qed = np.load(f"Histograms/Combined/qed.npy", allow_pickle=True)

# Uncombined set
# 2
n = 2
element_type_2 = np.load(f"Histograms/{n}/element_type_{n}.npy", allow_pickle=True)
num_heavy_atom_neighbors_2 = np.load(f"Histograms/{n}/num_heavy_atom_neighbors_{n}.npy", allow_pickle=True)
num_h_neighbors_2 = np.load(f"Histograms/{n}/num_h_neighbors_{n}.npy", allow_pickle=True)
formal_charge_2 = np.load(f"Histograms/{n}/formal_charge_{n}.npy", allow_pickle=True)
in_a_ring_2 = np.load(f"Histograms/{n}/in_a_ring_{n}.npy", allow_pickle=True)
is_aromatic_2 = np.load(f"Histograms/{n}/is_aromatic_{n}.npy", allow_pickle=True)

bond_type_2 = np.load(f"Histograms/{n}/bond_type_{n}.npy", allow_pickle=True)

num_heavy_atoms_2 = np.load(f"Histograms/{n}/num_heavy_atoms_{n}.npy", allow_pickle=True)
total_num_atoms_2 = np.load(f"Histograms/{n}/total_num_atoms_{n}.npy", allow_pickle=True)
molecular_weight_2 = np.load(f"Histograms/{n}/molecular_weight_{n}.npy", allow_pickle=True)

sas_2 = np.load(f"Histograms/{n}/sas_{n}.npy", allow_pickle=True)
qed_2 = np.load(f"Histograms/{n}/qed_{n}.npy", allow_pickle=True)

# Combining histograms
element_type = combine_hists(element_type, element_type_2, "element_type")
num_heavy_atom_neighbors = combine_hists(num_heavy_atom_neighbors, num_heavy_atom_neighbors_2, "num_heavy_atom_neighbors")
num_h_neighbors = combine_hists(num_h_neighbors, num_h_neighbors_2, "num_h_neighbors")
formal_charge = combine_hists(formal_charge, formal_charge_2, "formal_charge")
in_a_ring = combine_hists(in_a_ring, in_a_ring_2, "in_a_ring")
is_aromatic = combine_hists(is_aromatic, is_aromatic_2, "is_aromatic")

bond_type = combine_hists(bond_type, bond_type_2, "bond_type")

num_heavy_atoms = combine_hists(num_heavy_atoms, num_heavy_atoms_2, "num_heavy_atoms")
total_num_atoms = combine_hists(total_num_atoms, total_num_atoms_2, "total_num_atoms")
molecular_weight = combine_hists(molecular_weight, molecular_weight_2, "molecular_weight")

sas = combine_hists(sas, sas_2, "sas")
qed = combine_hists(qed, qed_2, "qed")

# Saving
np.save(f"Histograms/Combined/element_type", element_type, allow_pickle=True)
np.save(f"Histograms/Combined/num_heavy_atom_neighbors", num_heavy_atom_neighbors, allow_pickle=True)
np.save(f"Histograms/Combined/num_h_neighbors", num_h_neighbors, allow_pickle=True)
np.save(f"Histograms/Combined/formal_charge", formal_charge, allow_pickle=True)
np.save(f"Histograms/Combined/in_a_ring", in_a_ring, allow_pickle=True)
np.save(f"Histograms/Combined/is_aromatic", is_aromatic, allow_pickle=True)

np.save(f"Histograms/Combined/bond_type", bond_type, allow_pickle=True)

np.save(f"Histograms/Combined/num_heavy_atoms", num_heavy_atoms, allow_pickle=True)
np.save(f"Histograms/Combined/total_num_atoms", total_num_atoms, allow_pickle=True)
np.save(f"Histograms/Combined/molecular_weight", molecular_weight, allow_pickle=True)

np.save(f"Histograms/Combined/sas", sas, allow_pickle=True)
np.save(f"Histograms/Combined/qed", qed, allow_pickle=True)