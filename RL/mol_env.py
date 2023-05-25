import rdkit.Chem as Chem
from rdkit.Chem import RWMol
from rdkit.Chem import Draw

import graph_embedding as GE

# QED
import rdkit.Chem.QED as QED

# SAS
from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

import matplotlib.pyplot as plt
import copy

class Mol_Env():
    def __init__(self, max_mol_size, max_steps):
        self.state = RWMol()
        self.mol_size = self.state.GetNumHeavyAtoms()  # The number of heavy atoms in the current molecule

        self.atom_bank = [Chem.Atom(symbol) for symbol in ['C', 'O', 'N', 'S', 'F', 'Cl', 'P', 'Br', 'I', 'B']]
        self.bond_bank = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]

        self.max_valences = {'C': 4, 'O': 2, 'N': 3, 'S': 6, 'F': 1, 'Cl': 1, 'P': 5, 'Br': 1, 'I': 1, 'B': 3}

        self.max_mol_size = max_mol_size

        self.timestep = 0
        self.max_steps = max_steps

        pass

    def _reward(self):
        """
        Calculates reward based on QED and SAS of a molecule
        """

        qed = QED.weights_mean(self.state) # From 0 to 1 where 1 is the most drug-like
        sas = sascorer.calculateScore(self.state) # From 1 to 10 where 1 is easiest to synthesize

        reward = (qed + (1-(sas-1)/9))*0.5 # QED and SAS are weighed equally
        # print(f'QED: {qed}, SAS: {sas}')

        return reward

    def _update_state(self, atom1, atom2, bond):
        """Returns the new state given the atom indices and a bond"""

        new_state = copy.copy(self.state)

        if atom2 >= self.mol_size:  # Adding a new atom to the molecule
            idx = new_state.AddAtom(Chem.Atom(self.atom_bank[atom2 - self.mol_size]))
            new_state.AddBond(atom1, idx, order=self.bond_bank[bond])

        else:  # Creating a bond between pre-existing atoms
            new_state.AddBond(atom1, atom2, order=self.bond_bank[bond])

        return new_state

    def _validate_action(self, atom1, atom2, bond):
        """
        Analyzes whether a proposed action is chemically valid
        :return: True if the action is invalid and False if action is valid
        """

        if atom1 == atom2: # Can not form a bond with the same atom
            return True
        # Can not form a bond between atoms which already have a bond
        elif atom2 < self.mol_size and self.state.GetBondBetweenAtoms(atom1, atom2) is not None:
            return True
        else:
            # Check valences of proposed addition

            # Calculating valence for the first atom which will need to be checked regardless of whether a new atom
            # is being added or a bond is being formed between pre-existing atoms
            if atom2 >= self.mol_size: # Adding a new atom
                atom_1 = self.state.GetAtomWithIdx(atom1)
                atom_valence_1 = sum([int(bond.GetBondType()) for bond in atom_1.GetBonds()])
                proposed_valence_1 = atom_valence_1 + bond + 1
                max_valence_1 = self.max_valences.get(atom_1.GetSymbol())

                atom_2 = Chem.Atom(self.atom_bank[atom2 - self.mol_size])
                proposed_valence_2 = bond + 1
                max_valence_2 = self.max_valences.get(atom_2.GetSymbol())

                if proposed_valence_1 > max_valence_1 or proposed_valence_2 > max_valence_2:
                    return True

            else: # Making a bond between pre-existing atoms
                atom_1 = self.state.GetAtomWithIdx(atom1)
                atom_valence_1 = sum([int(bond.GetBondType()) for bond in atom_1.GetBonds()])
                proposed_valence_1 = atom_valence_1 + bond + 1
                max_valence_1 = self.max_valences.get(atom_1.GetSymbol())

                atom_2 = self.state.GetAtomWithIdx(atom2)
                atom_valence_2 = sum([int(bond.GetBondType()) for bond in atom_2.GetBonds()])
                proposed_valence_2 = atom_valence_2 + bond + 1
                max_valence_2 = self.max_valences.get(atom_2.GetSymbol())

                # If either valence is violated, action is invalid
                if proposed_valence_1 > max_valence_1 or proposed_valence_2 > max_valence_2:
                    return True

            ring_info = self.state.GetRingInfo()
            if len(ring_info.AtomRings()) != 0:
                # Cannot make bond between atoms in the same ring
                if atom2 < self.mol_size:
                    return ring_info.AreAtomsInSameRing(atom1, atom2)

            # All rings must be between 3 and 7 atoms in size for the new state
            proposed_state = self._update_state(atom1, atom2, bond)
            proposed_ring_info = proposed_state.GetRingInfo()
            if len(proposed_ring_info.AtomRings()) != 0:
                proposed_ring_sizes = [len(ring) for ring in proposed_ring_info.AtomRings()]
                if min(proposed_ring_sizes) < 3 or max(proposed_ring_sizes) > 7:
                    return True

        return False

    def reset(self, smiles = 'C'):
        """
        Resets the environment to an initial state with a single random atom
        :return: initial state, info (invalid, timestep)
        """
        # atom = Chem.Atom(random.choice(self.atom_bank)) # Resets to a random atom in atom_bank

        # Resets to a molecule of choice (defaults to carbon atom)
        if smiles == 'C':
            atom = Chem.Atom(smiles)
            self.state.AddAtom(atom)
        else:
            # mol = Chem.MolFromSmiles('Cn1cnc2n(C)c(=O)n(C)c(=O)c12') # Caffeine molecule
            mol = Chem.MolFromSmiles(smiles)
            self.state = RWMol(mol)
        self.mol_size = self.state.GetNumHeavyAtoms()

        info = (self.timestep, False)

        return self.state, info

    def step(self, terminate, atom1, atom2, bond):
        """
        The step function of the Mol_Env class
        :param terminate: either 0 (continue) or 1 (terminate)
        :param atom1: index of first atom from 0 to N where N is the total number of atoms
        :param atom2: index of second atom from 0 to N where atom1 < atom2
        :param bond: a number from 0 to 2 indicating the 3 possible bond types to add (aromatic bonds calculated later)
        :return: new state, reward, terminate/truncated, info
        """

        self.timestep += 1
        invalid = self._validate_action(atom1, atom2, bond) # True if the proposed action is invalid
        info = (self.timestep, invalid)

        truncated = False
        terminated = False

        reward = self._reward()

        if terminate == 1: # Model decides to stop adding on to molecule
            terminated = True
        elif self.state.GetNumHeavyAtoms() == self.max_mol_size or self.timestep == self.max_steps: # The size of the molecule hits the limit
            truncated = True
        else: # No truncating or terminating
            if invalid: # If the action is invalid, return the current state again with -0.5 as reward
                reward = -0.5
            else: # If the action is valid, update RWMol accordingly
                self.state = self._update_state(atom1, atom2, bond)

        self.mol_size = self.state.GetNumHeavyAtoms()

        self.state.UpdatePropertyCache()
        Chem.SanitizeMol(self.state, catchErrors = True)

        return self.state, reward, terminated, truncated, info

        def visualize(self, invalid=False, timestep=True):
            """
            Visualize the current state
            """

            if not invalid:
                # self.state = Chem.MolFromSmiles("Cn1cnc2n(C)c(=O)n(C)c(=O)c12") # Caffeine drawing
                state_copy = copy.copy(self.state)
                for atom in state_copy.GetAtoms():
                    atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))
                img = Draw.MolToImage(state_copy)
                plt.imshow(img)
                plt.axis('off')
                if timestep:
                    plt.text(x=0, y=1, s=f"Timestep: {self.timestep}", size='large', transform=plt.gca().transAxes)
                plt.show()
            else:
                pass

# env = Mol_Env(50)
# print("Reset")
# env.reset('CCCCC')
# env.visualize()
# print()
#
# print("Add carbon")
# print(env.step(0, 0, 5, 0))
# print(Chem.MolToSmiles(env.state))
# env.visualize()
# print()
#
# print("Create erroneous bond")
# print(env.step(0, 4, 5, 0))
# print(Chem.MolToSmiles(env.state))
# env.visualize()
#
# print(len(env.state.GetRingInfo().AtomRings()))