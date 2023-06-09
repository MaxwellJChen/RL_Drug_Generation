# RDKit
import rdkit.Chem as Chem
from rdkit.Chem import RWMol
from rdkit.Chem import Draw
import rdkit.Chem.QED as QED # QED
from rdkit.Chem import RDConfig # SAS
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

import copy

# Visualize
import matplotlib.pyplot as plt


class single_mol_env():
    """
    Unvectorized environment
    """

    def __init__(self, max_mol_size, max_steps):
        self.state = RWMol()
        self.mol_size = self.state.GetNumHeavyAtoms()  # The number of heavy atoms in the current molecule

        self.atom_bank = [Chem.Atom(symbol) for symbol in ['C', 'O', 'N', 'S', 'F', 'Cl', 'P', 'Br', 'I', 'B']]
        self.bond_bank = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]

        self.max_valences = {'C': 4, 'O': 2, 'N': 3, 'S': 6, 'F': 1, 'Cl': 1, 'P': 5, 'Br': 1, 'I': 1, 'B': 3}
        self.has_max_valence = {} # Records the atoms that are bonded to their maximum number of heavy neighbors
        self.atom_valences = {}

        self.max_mol_size = max_mol_size

        self.timestep = 0
        self.max_steps = max_steps

    def _reward(self):
        """
        Calculates reward based on QED and SAS of a molecule
        """

        qed = QED.weights_mean(self.state) # From 0 to 1 where 1 is the most drug-like
        sas = sascorer.calculateScore(self.state) # From 1 to 10 where 1 is easiest to synthesize

        reward = qed*0.75 + (1-(sas-1)/9)*0.25 # QED and SAS are weighed equally

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
        Resets the environment to an initial state
        :return: initial state, info (timestep, invalid)
        """
        # atom = Chem.Atom(random.choice(self.atom_bank)) # Resets to a random atom in atom_bank

        # Resets to a molecule of choice (defaults to carbon atom)
        self.state = RWMol()
        if smiles == 'C':
            atom = Chem.Atom(smiles)
            self.state.AddAtom(atom)
        else:
            # mol = Chem.MolFromSmiles('Cn1cnc2n(C)c(=O)n(C)c(=O)c12') # Caffeine molecule
            mol = Chem.MolFromSmiles(smiles)
            self.state = RWMol(mol)
        self.mol_size = self.state.GetNumHeavyAtoms()
        self.timestep = 0

        self.atom_valences = {} # Reset atom valence
        self.has_max_valence = {}
        atoms = self.state.GetAtoms()
        for i, atom in enumerate(self.state.GetAtoms()):
            self.atom_valences[i] = sum([int(bond.GetBondType()) for bond in atom.GetBonds()]) # Recalculate atom valences
            if self.max_valences[atoms[i].GetSymbol()] == self.atom_valences[i]:
                self.has_max_valence[i] = True
            else:
                self.has_max_valence[i] = False

        info = (self.timestep, False)

        return self.state, info

    def step(self, terminate, atom1, atom2, bond):
        """
        The step function of the Mol_Env class
        :param terminate: either 0 (continue) or 1 (terminate)
        :param atom1: index of first atom from 0 to n where n is the number of atoms in the molecule
        :param atom2: index of second atom from 0 to n + 10 (atoms in molecule + atoms in node bank) where atom1 < atom2
        :param bond: a number from 0 to 2 indicating the 3 possible bond types to add (aromatic bonds calculated later)
        :return: new state, reward, terminate/truncated, info
        """

        self.timestep += 1
        invalid = False # True if the proposed action is invalid
        info = (self.timestep, invalid)

        truncated = False
        terminated = False

        reward = 0 # If the action is invalid, return the current state again with -0.5 as reward

        if terminate == 1: # Model decides to stop adding on to molecule
            terminated = True
            if self.mol_size != 1:
                reward = self._reward() + self.mol_size/50
        elif self.state.GetNumHeavyAtoms() == self.max_mol_size or self.timestep == self.max_steps: # The size of the molecule hits the limit
            truncated = True
            if self.mol_size != 1:
                reward = self._reward() + self.mol_size/50
        else: # No truncating or terminating
            invalid = self._validate_action(atom1, atom2, bond)  # True if the proposed action is invalid
            if not invalid: # If the action is valid, update RWMol accordingly
                self.state = self._update_state(atom1, atom2, bond)
                self.state.UpdatePropertyCache()
                Chem.SanitizeMol(self.state, catchErrors=True)

                self.atom_valences[atom1] += bond + 1  # Updating the atom valences
                if atom2 >= self.mol_size:  # Adding a new atom
                    self.atom_valences[self.mol_size] = bond + 1
                    self.has_max_valence[self.mol_size] = False
                else:
                    self.atom_valences[atom2] += bond + 1

                atoms = self.state.GetAtoms()
                for valence in self.atom_valences:
                    # Get the max valence
                    atom_max_valence = self.max_valences[atoms[valence].GetSymbol()]
                    if atom_max_valence == self.atom_valences[valence]:
                        self.has_max_valence[valence] = True

                self.mol_size = self.state.GetNumHeavyAtoms()
                reward = self._reward() + self.mol_size/50

        return self.state, reward, terminated, truncated, info

    def visualize(self, return_image=False):
        """
        Visualize the current state.
        """

        # Generate image
        state_copy = copy.copy(self.state)
        for atom in state_copy.GetAtoms():
            atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))
        img = Draw.MolToImage(state_copy)

        # Return a single frame as a gif
        if return_image:
            return img

        # Display image with matplotlib
        else:
            plt.imshow(img)
            plt.axis('off')
            plt.text(x=0, y=1, s=f"Timestep: {self.timestep}", size='large', transform=plt.gca().transAxes)
            plt.show()

class vectorized_mol_env():
    """
    Vectorized environment for PPO.
    Returns batches of observations and steps according to a user-specified number of environments.
    Maintains a done vector representing whether a certain environment is finished.
    Accepts a batch of actions.
    Simultaneously updates N independent environments.
    """

    def __init__(self, max_mol_size, max_steps, num_envs):
        self.num_envs = num_envs

        self.states = [RWMol() for i in range(self.num_envs)] # List of states
        self.mol_sizes = [state.GetNumHeavyAtoms() for state in self.states] # List of mol_sizes

        self.atom_bank = [Chem.Atom(symbol) for symbol in ['C', 'O', 'N', 'S', 'F', 'Cl', 'P', 'Br', 'I', 'B']]
        self.bond_bank = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]

        self.max_valences = {'C': 4, 'O': 2, 'N': 3, 'S': 2, 'F': 1, 'Cl': 1, 'P': 3, 'Br': 1, 'I': 1, 'B': 3}

        self.max_mol_size = max_mol_size

        self.timesteps = [0 for i in range(self.num_envs)]
        self.max_steps = max_steps

    def _reward(self, i):
        """
        Calculates reward for a specific state based on QED and SAS
        """

        qed = QED.weights_mean(self.states[i])  # From 0 to 1 where 1 is the most drug-like
        sas = sascorer.calculateScore(self.states[i])  # From 1 to 10 where 1 is easiest to synthesize

        reward = qed * 0.75 + (1 - (sas - 1) / 9) * 0.25  # QED and SAS are weighed equally

        return reward

    def _update_state(self, atom1, atom2, bond, i):
        """Returns the new state given the atom indices and a bond"""

        new_state = copy.copy(self.states[i])

        if atom2 >= self.mol_sizes[i]:  # Adding a new atom to the molecule
            atom = self.atom_bank[atom2 - self.mol_sizes[i]]
            new_state.AddAtom(atom)
            new_state.AddBond(atom1, new_state.GetNumHeavyAtoms() - 1, order=self.bond_bank[bond])

        else:  # Creating a bond between pre-existing atoms
            new_state.AddBond(atom1, atom2, order=self.bond_bank[bond])

        return new_state

    def _validate_action(self, atom1: int, atom2: int, bond: int, i):
        """
        Analyzes whether a proposed action is chemically valid
        Operates on one action at a time on a state specified by idx
        :return: True if the action is invalid and False if action is valid
        """
        if atom1 == atom2:  # Can not form a bond with the same atom
            return True
        # Can not form a bond between atoms which already have a bond
        elif atom2 < self.mol_sizes[i] and self.states[i].GetBondBetweenAtoms(atom1, atom2) is not None:
            return True
        else:
            # Check valences of proposed addition
            # Calculating valence for the first atom which will need to be checked regardless of whether a new atom
            # is being added or a bond is being formed between pre-existing atoms
            if atom2 >= self.mol_sizes[i]:  # Adding a new atom
                atom_1 = self.states[i].GetAtomWithIdx(atom1)
                atom_valence_1 = sum([int(bond.GetBondType()) for bond in atom_1.GetBonds()])
                proposed_valence_1 = atom_valence_1 + bond + 1
                max_valence_1 = self.max_valences.get(atom_1.GetSymbol())

                atom_2 = Chem.Atom(self.atom_bank[atom2 - self.mol_sizes[i]])
                proposed_valence_2 = bond + 1
                max_valence_2 = self.max_valences.get(atom_2.GetSymbol())

                if proposed_valence_1 > max_valence_1 or proposed_valence_2 > max_valence_2:
                    return True

            else:  # Making a bond between pre-existing atoms
                atom_1 = self.states[i].GetAtomWithIdx(atom1)
                atom_valence_1 = sum([int(bond.GetBondType()) for bond in atom_1.GetBonds()])
                proposed_valence_1 = atom_valence_1 + bond + 1
                max_valence_1 = self.max_valences.get(atom_1.GetSymbol())

                atom_2 = self.states[i].GetAtomWithIdx(atom2)
                atom_valence_2 = sum([int(bond.GetBondType()) for bond in atom_2.GetBonds()])
                proposed_valence_2 = atom_valence_2 + bond + 1
                max_valence_2 = self.max_valences.get(atom_2.GetSymbol())

                # If either valence is violated, action is invalid
                if proposed_valence_1 > max_valence_1 or proposed_valence_2 > max_valence_2:
                    return True

            ring_info = self.states[i].GetRingInfo()
            if len(ring_info.AtomRings()) != 0:
                # Cannot make bond between atoms in the same ring
                if atom2 < self.mol_sizes[i]:
                    return ring_info.AreAtomsInSameRing(atom1, atom2)

            # All rings must be between 3 and 7 atoms in size for the new state
            proposed_state = self._update_state(atom1, atom2, bond, i)
            proposed_ring_info = proposed_state.GetRingInfo()
            if len(proposed_ring_info.AtomRings()) != 0:
                proposed_ring_sizes = [len(ring) for ring in proposed_ring_info.AtomRings()]
                if min(proposed_ring_sizes) < 3 or max(proposed_ring_sizes) > 7:
                    return True
        return False

    def reset(self, smiles='C'):
        """
        Resets all environments to an initial state
        :return: initial state, info (timestep, mol_size, info)
        """
        # Resets to a molecule of choice (defaults to carbon atom)
        for i in range(len(self.states)):
            self.states[i] = RWMol()
            if smiles == 'C':
                atom = Chem.Atom(smiles)
                self.states[i].AddAtom(atom)
            else:
                mol = Chem.MolFromSmiles(smiles)
                self.states[i] = RWMol(mol)

            self.mol_sizes[i] = self.states[i].GetNumHeavyAtoms()

            self.timesteps[i] = 0

        info = [(0, 1, False) for _ in range(self.num_envs)]

        return self.states, info

    def _reset_single(self, i, smiles = 'C'):
        """
        Resets a specific environment
        """

        self.states[i] = RWMol()
        if smiles == 'C':
            atom = Chem.Atom(smiles)
            self.states[i].AddAtom(atom)
        else:
            # mol = Chem.MolFromSmiles('Cn1cnc2n(C)c(=O)n(C)c(=O)c12') # Caffeine molecule
            mol = Chem.MolFromSmiles(smiles)
            self.states[i] = RWMol(mol)
        self.mol_sizes[i] = self.states[i].GetNumHeavyAtoms()
        self.timesteps[i] = 0

    def step(self, terminate: list, atom1: list, atom2: list, bond: list, debug = False):
        """
        The step function of the vectorized environment. Given lists of each parameter, outputs a list of new states.
        :param terminate: list of either 0 (continue) or 1 (terminate)
        :param atom1: list of indices of first atom from 0 to n where n is the number of atoms in the molecule
        :param atom2: list of indices of second atom from 0 to n + 10 (atoms in molecule + atoms in node bank) where atom1 < atom2
        :param bond: list of numbers from 0 to 2 indicating the 3 possible bond types to add (aromatic bonds calculated later)
        :return: list of new states, rewards, done, info
        """

        rewards = []
        infos = []

        dones = [0 for i in range(self.num_envs)]

        for i in range(self.num_envs):
            self.timesteps[i] += 1
            invalid = self._validate_action(atom1[i], atom2[i], bond[i], i)  # True if the proposed action is invalid
            truncated = False
            terminated = False

            reward = -0.5  # If the action is invalid, return the current state again with -0.5 as reward

            if terminate[i] == 1:  # Model decides to stop adding on to molecule
                terminated = True
            elif self.states[i].GetNumHeavyAtoms() == self.max_mol_size or self.timesteps[i] == self.max_steps:  # The size of the molecule hits the limit
                truncated = True
            else:  # No truncating or terminating
                if not invalid:  # If the action is valid, update RWMol accordingly
                    self.states[i] = self._update_state(atom1[i], atom2[i], bond[i], i)
                    self.states[i].UpdatePropertyCache()
                    Chem.SanitizeMol(self.states[i], catchErrors=True)

                    self.mol_sizes[i] = self.states[i].GetNumHeavyAtoms()
                    reward = self._reward(i)

            if terminated or truncated: # Reset this specific state
                self._reset_single(i)
                dones[i] += 1
                reward = 0

            info = (self.timesteps[i], self.mol_sizes[i], invalid)

            rewards.append(reward)
            infos.append(info)

        if debug:
            print(self.states)
            for i, state in enumerate(self.states):
                print(f'mol {i + 1}: {[atom.GetSymbol() for atom in state.GetAtoms()]}')
            print(f'mol_sizes: {self.mol_sizes}')
            print(dones)
            print(infos)
            print('-----------------------------------------------------------------')
            print()

        return self.states, rewards, dones, infos

    def visualize(self, idx, return_image=False):
        """
        Visualize a specific current state.
        """

        # Generate image
        state_copy = copy.copy(self.states[idx])
        for atom in state_copy.GetAtoms():
            atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))
        img = Draw.MolToImage(state_copy)

        # Return a single frame as a gif
        if return_image:
            return img

        # Display image with matplotlib
        else:
            plt.imshow(img)
            plt.axis('off')
            plt.text(x=0, y=1, s=f"Timestep: {self.timestep}", size='large', transform=plt.gca().transAxes)
            plt.show()

if __name__ == '__main__':
    from Reinforcement_Learning.PPO.ppo_policy import ppo_policy
    from graph_embedding import batch_from_smiles, batch_from_states
    import warnings
    import torch

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    smiles = 'Clc1ccc([C@]23CNC[C@H]2C3)cc1Cl'
    sample_batch = batch_from_smiles([smiles])

    SURGE = ppo_policy(sample_batch.num_node_features)
    mol_env = single_mol_env(max_mol_size = 50, max_steps = 50)
    obs, info = mol_env.reset()
    mol_env.visualize()

    all_obs = []

    # ['C', 'O', 'N', 'S', 'F', 'Cl', 'P', 'Br', 'I', 'B']

    for _ in range(20):
        print(mol_env.timestep)
        mol_env.visualize()
        with torch.no_grad():
            t_act, t_log_prob, n1_act, n2_act, nmol_act, nmol_log_prob, nfull_act, nfull_log_prob, b_act, b_log_prob = SURGE.act(batch_from_states([obs]), mol_env)
        print(f'{t_act} {n1_act} {n2_act} {b_act}')
        next_obs, reward, terminated, truncated, info = mol_env.step(t_act[0], n1_act[0], n2_act[0], b_act[0])
        if info[1]:
            print('Invalid action.')
        else:
            print('Valid.')
        if terminated or truncated:
            next_obs, info = mol_env.reset()

        all_obs.append(obs)
        obs = next_obs

        print()