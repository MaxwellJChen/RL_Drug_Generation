# RDKit
import rdkit.Chem as Chem
from rdkit.Chem import RWMol
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors # MW
import numpy as np
import rdkit.Chem.QED as QED # QED
from rdkit.Chem import RDConfig # SAS
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

import copy

# Visualize
import matplotlib.pyplot as plt

class vectorized_mol_env():
    """
    Vectorized environment for SURGE training.
    """
    def __init__(self, num_envs = 4, max_mol_size = 100, max_steps = 100):
        self.num_envs = num_envs
        self.states = []

        self.atom_bank = [Chem.Atom(symbol) for symbol in ['C', 'O', 'N', 'S', 'F', 'Cl', 'P', 'Br', 'I', 'B']]
        # No aromatic bonds. All molecules are kekulized.
        self.bond_bank = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]

        # Max valences of the atoms in the atom bank
        self.max_valences = {'C': 4, 'O': 2, 'N': 3, 'S': 6, 'F': 1, 'Cl': 1, 'P': 5, 'Br': 1, 'I': 1, 'B': 3}

        # Represent general guidelines on range of acceptable molecular weights
        self.mw_std = 75 # Not an actual standard deviation but acts similarly to a standard deviation for mw score
        self.mw_mean = 300
        self.max_mw = 1000
        self.mol_mws = []

        self.mol_sizes = []
        self.max_mol_size = max_mol_size

        self.timestep = 1
        self.max_steps = max_steps

    def reset(self, *args):
        """
        Provides the initial observation and resets all states to carbon atoms if not otherwise specified.
        """
        smiles = args
        if len(smiles) == 0: # When no smiles are specified, default to a carbon atom
            self.states = [RWMol() for _ in range(self.num_envs)]
            for i in range(self.num_envs):
                self.states[i].AddAtom(Chem.Atom('C'))
                self.states[i].UpdatePropertyCache()
        else:
            smiles = args[0]
            self.states = [RWMol(Chem.MolFromSmiles(smiles[i])) for i in range(self.num_envs)]
            for i in range(self.num_envs):
                Chem.Kekulize(self.states[i])
                self.states[i].UpdatePropertyCache()

        self.mol_sizes = [state.GetNumHeavyAtoms() for state in self.states]
        self.mol_mws = [Descriptors.MolWt(state) for state in self.states]
        self.timestep = 1

        return self.states

    def _score(self, i):
        """
        Calculates the final chemical score of a molecule. All intermediate scores are from -2 to 2.
        """
        qed = 4 * (QED.weights_mean(self.states[i]) - 0.5) # From -2 to 2 where 2 is the most drug-like

        sas = sascorer.calculateScore(self.states[i]) # From 1 to 10 where 1 is the easiest to synthesize
        sas = 4/9 * (sas - 5.5) # Scaled from -2 to 2
        sas = 1 - sas # Transformed so that more accessible molecules have a higher score

        # Molecular weight score linearly decreases as weight deviates from mean and is between -2 and 2
        mw = Descriptors.MolWt(self.states[i])
        mw = max(-2, -np.abs(mw - self.mw_mean)/self.mw_std + 2)

        return qed + sas + mw

    def _update(self, nmol, nfull, bond, i):
        """
        Returns an updated version of the state of a specific environment given the action.
        """
        new_state = copy.copy(self.states[i])
        if nfull[i] >= self.mol_sizes[i]:
            idx = new_state.AddAtom(Chem.Atom(self.atom_bank[nfull[i] - self.states[i].GetNumHeavyAtoms()]))
            new_state.AddBond(nmol[i], idx, order = self.bond_bank[bond[i]])
        else:
            new_state.AddBond(nmol[i], nfull[i], order = self.bond_bank[bond[i]])

        Chem.SanitizeMol(new_state, catchErrors = True)
        new_state.UpdatePropertyCache()
        Chem.Kekulize(new_state)

        return new_state

    def _validate(self, terminate, nmol, nfull, bond):
        """
        Analyzes whether a proposed action is chemically valid.
        :return a list of True (valid action) or False (invalid action) for each state-action pair called valids

        Checks for invalidity:
            1. Action is not termination
            2. Cannot form a bond between the same atom
            3. Cannot form a bond between atoms that already have a bond
            4. Valence of the bond cannot exceed the remaining valence of the selected atoms
            5. Cannot make a bond between atoms in the same ring
            6. Cannot make a bond that would result in a ring of length greater than 7
        """

        valids = [True for _ in range(self.num_envs)]

        for i in range(self.num_envs):
            if terminate[i] == 1: # 1. Action is not termination
                continue
            elif nmol[i] == nfull[i]: # 2. Cannot form a bond between the same atom
                valids[i] = False
                continue
            # 3. Cannot form a bond between atoms that already have a bond
            elif nfull[i] < self.mol_sizes[i] and self.states[i].GetBondBetweenAtoms(nmol[i], nfull[i]) is not None:
                valids[i] = False
                continue
            else:
                # 4. Valence of the bond cannot exceed the remaining valence of the selected atoms
                atom_mol = self.states[i].GetAtomWithIdx(nmol[i])
                atom_valence_mol = sum([int(bond.GetBondType()) for bond in atom_mol.GetBonds()]) - atom_mol.GetFormalCharge()
                proposed_valence_mol = atom_valence_mol + bond[i] + 1
                max_valence_mol = self.max_valences[atom_mol.GetSymbol()]

                if nfull[i] >= self.mol_sizes[i]: # nfull is in the atom bank
                    atom_full = self.atom_bank[nfull[i] - self.mol_sizes[i]]
                    atom_valence_full = 0
                    max_valence_full = self.max_valences[atom_full.GetSymbol()]
                else: # nfull is a preexisting atom
                    atom_full = self.states[i].GetAtomWithIdx(nfull[i])
                    atom_valence_full = sum([int(bond.GetBondType()) for bond in atom_full.GetBonds()]) - atom_full.GetFormalCharge()
                    max_valence_full = self.max_valences[atom_full.GetSymbol()]
                proposed_valence_full = atom_valence_full + bond[i] + 1

                if proposed_valence_mol > max_valence_mol or proposed_valence_full > max_valence_full:
                    valids[i] = False
                    continue

            # 5. Cannot make bond between atoms in the same ring
            ring_info = self.states[i].GetRingInfo()
            if len(ring_info.AtomRings()) != 0:
                if nfull[i] < self.mol_sizes[i] and ring_info.AreAtomsInSameRing(nmol[i], nfull[i]):
                    valids[i] = False
                    continue

            # 6. Cannot make a bond that would result in a ring of more than 7 atoms
            proposed_state = self._update(nmol, nfull, bond, i)
            proposed_ring_info = proposed_state.GetRingInfo()
            if len(proposed_ring_info.AtomRings()) != 0:
                proposed_ring_sizes = [len(ring) for ring in proposed_ring_info.AtomRings()]
                if max(proposed_ring_sizes) > 7:
                    valids[i] = False

        return valids

    def step(self, terminate, nmol, nfull, bond):
        """
        Given an action, updates each of the states accordingly.
        If one of the states terminates, it automatically resets to a new initial state.
        Returns the updated states, rewards for each action, action validities, and the timestep.
        """

        self.timestep += 1
        valids = self._validate(terminate, nmol, nfull, bond)
        rewards = [0 for _ in range(self.num_envs)]

        for i in range(self.num_envs):

            # Reset if model decides to terminate, molecules hit maximum size or weight, or max number of timesteps is reached
            if terminate[i] == 1 or self.mol_sizes[i] >= self.max_mol_size or self.mol_mws[i] >= self.max_mw or self.timestep >= self.max_steps:


                if self.states[i].GetNumHeavyAtoms() == 1:
                    # Return lowest chemical score possible if state is a single carbon atom
                    rewards[i] = -6
                else:
                    rewards[i] = self._score(i)

                # Reset environment to single carbon atom
                self.states[i] = RWMol()
                self.states[i].AddAtom(Chem.Atom('C'))
                self.states[i].UpdatePropertyCache()

            # Score the step based on action validity and molecular weight
            elif valids[i]:
                self.states[i] = self._update(nmol, nfull, bond, i)
                rewards[i] = 0.5 + max(0, (-np.abs(self.mol_mws[i] - self.mw_mean)/self.mw_std + 1))/10
            else: # Invalid action
                rewards[i] = -0.5 + max(0, (-np.abs(self.mol_mws[i] - self.mw_mean)/self.mw_std + 1))/10

        self.mol_sizes = [state.GetNumHeavyAtoms() for state in self.states]
        self.mol_mws = [Descriptors.MolWt(state) for state in self.states]

        return self.states, rewards, valids, self.timestep

    def visualize(self):
        """
        Returns a matplotlib plot of each of the states.
        """
        for i in range(self.num_envs):
            state_copy = copy.copy(self.states[i])
            for atom in state_copy.GetAtoms():
                atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))
            img = Draw.MolToImage(state_copy)
            plt.imshow(img)
            plt.axis('off')
            plt.text(x = 0, y = 1, s = f'Env: {i + 1}\nTimestep: {self.timestep}',
                     size = 'large', transform = plt.gca().transAxes)
            plt.show()

class single_mol_env():
    """
    Molecular environment with a single state at a time.
    """

    def __init__(self, max_mol_size = 100, max_steps = 100):
        self.state = RWMol()

        self.atom_bank = [Chem.Atom(symbol) for symbol in ['C', 'O', 'N', 'S', 'F', 'Cl', 'P', 'Br', 'I', 'B']]
        self.bond_bank = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE] # No aromatic bonds. All molecules are kekulized.

        self.max_valences = {'C': 4, 'O': 2, 'N': 3, 'S': 6, 'F': 1, 'Cl': 1, 'P': 5, 'Br': 1, 'I': 1,
                             'B': 3} # Max valences of the atoms in the atom bank

        self.mol_size = 0
        self.max_mol_size = max_mol_size

        self.timestep = 1
        self.max_steps = max_steps

    def reset(self, smiles = 'C'):
        if smiles == 'C':
            self.state.AddAtom(Chem.Atom(smiles))
        else:
            self.state = Chem.MolFromSmiles(smiles)
            self.state = RWMol(self.state)
            Chem.Kekulize(self.state)

        self.mol_size = self.state.GetNumHeavyAtoms()

        return self.state

    def _score(self):
        qed = QED.weights_mean(self.state)
        sas = sascorer.calculateScore(self.state)
        sas = (sas - 1)/9
        sas = 1 - sas

        return 0.5 * qed + 0.5 * sas + self.state.GetNumHeavyAtoms()/self.max_mol_size

    def _update(self, nmol, nfull, bond):
        """
        Applies a theoretical update to the current state without actually changing the state.
        """

        new_state = copy.copy(self.state)
        if nfull >= self.mol_size:
            idx = new_state.AddAtom(Chem.Atom(self.atom_bank[nfull - self.state.GetNumHeavyAtoms()]))
            new_state.AddBond(nmol, idx, order = self.bond_bank[bond])
        else:
            new_state.AddBond(nmol, nfull, order = self.bond_bank[bond])

        Chem.SanitizeMol(new_state, catchErrors = True)
        new_state.UpdatePropertyCache()
        Chem.Kekulize(new_state)

        return new_state

    def _validate(self, terminate, nmol, nfull, bond):
        """
        Analyzes whether a proposed action is chemically valid.
        :return a list of True (valid action) or False (invalid action)

        Checks for invalidity:
            1. Action is not termination
            2. Cannot form a bond between the same atom
            3. Cannot form a bond between atoms that already have a bond
            4. Valence of the bond cannot exceed the remaining valence of the selected atoms
            5. Cannot make a bond between atoms in the same ring
            6. Cannot make a bond that would result in a ring of length greater than 7
        """

        if terminate == 1: #1. Action is not termination
            return True
        elif nmol == nfull: #2. Cannot form a bond between the same atom
            return False
        elif nfull < self.mol_size and self.state.GetBondBetweenAtoms(nmol, nfull) is not None: #3. Cannot form a bond between atoms that already have a bond
            return False
        else:
            #4. Valence of bond cannot exceed the remaining valence of the selected atoms
            atom_mol = self.state.GetAtomWithIdx(nmol)
            atom_valence_mol = sum([int(bond.GetBondType()) for bond in atom_mol.GetBonds()]) - atom_mol.GetFormalCharge()
            proposed_valence_mol = atom_valence_mol + bond + 1
            max_valence_mol = self.max_valences[atom_mol.GetSymbol()]

            if nfull >= self.mol_size: # nfull is in the atom bank
                atom_full = self.atom_bank[nfull - self.mol_size]
                atom_valence_full = 0
                max_valence_full = self.max_valences[atom_full.GetSymbol()]
            else:   #nfull is a preexisting atom
                atom_full = self.state.GetAtomWithIdx(nfull)
                atom_valence_full = sum([int(bond.GetBondType()) for bond in atom_full.GetBonds()]) - atom_full.GetFormalCharge()
                max_valence_full = self.max_valences[atom_full.GetSymbol()]
            proposed_valence_full = atom_valence_full + bond + 1

            if proposed_valence_mol > max_valence_mol or proposed_valence_full > max_valence_full:
                return False

        # 5. Cannot make bond between atoms in the same ring
        ring_info = self.state.GetRingInfo()
        if len(ring_info.AtomRings()) != 0:
            if nfull < self.mol_size and ring_info.AreAtomsInSameRing(nmol, nfull):
                return False

        # 6. Cannot make a bond that would result in a ring of more than 7 atoms
        proposed_state = self._update(nmol, nfull, bond)
        proposed_ring_info = proposed_state.GetRingInfo()
        if len(proposed_ring_info.AtomRings()) != 0:
            proposed_ring_sizes = [len(ring) for ring in proposed_ring_info.AtomRings()]
            if max(proposed_ring_sizes) > 7:
                return False

        return True

    def step(self, terminate, nmol, nfull, bond):
        """
        Given an action, updates the state accordingly.
        If the state terminates, the environment automatically resets to a new initial state.
        Returns the updated state, reward for the action, action validity, and the timestep.
        """

        self.timestep += 1
        valid = self._validate(terminate, nmol, nfull, bond)
        reward = 0

        # Reset if model decides to terminate, the molecule hits the maximum size, or the max number of timesteps is reached
        if terminate == 1 or self.mol_size == self.max_mol_size or self.timestep == self.max_steps:
            if self.state.GetNumHeavyAtoms() != 1:
                reward = self._score()
            else:
                reward = self.state.GetNumHeavyAtoms()/self.max_mol_size
            self.state = RWMol() # Resetting state after termination
            self.state.AddAtom(Chem.Atom('C'))
        elif valid: # Valid action
            self.state = self._update(nmol, nfull, bond)
            reward = self._score()
        else: # Invalid action
            reward = 0

        self.mol_size = self.state.GetNumHeavyAtoms()

        return self.state, reward, valid, self.timestep

    def visualize(self):
        """
        Returns a matplotlib plot of the state.
        """
        state_copy = copy.copy(self.state)
        for atom in state_copy.GetAtoms():
            atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))
        img = Draw.MolToImage(state_copy)
        plt.imshow(img)
        plt.axis('off')
        plt.text(x = 0, y = 1, s = f'Timestep: {self.timestep}', size = 'large', transform = plt.gca().transAxes)
        plt.show()

if __name__ == '__main__':
    mol_env = single_mol_env()