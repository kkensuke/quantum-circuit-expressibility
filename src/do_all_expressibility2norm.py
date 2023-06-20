import matplotlib.pyplot as plt
import numpy as np

from .expressibility import (Expressibility2norm,
                             analytical_haar_frame_potential)


class do_all:
    def __init__(self, circuit_types, nqubits_list, nlayers_list, nsamples):
        self.circuit_types = circuit_types
        self.nqubits_list = nqubits_list
        self.nlayers_list = nlayers_list
        self.nsamples = nsamples

    def frame_potential_samples_each_circuit_type_(self):
        self.frame_potential_samples_each_circuit_type = []
        for circuit_type in self.circuit_types:
            frame_potential_samples_each_nqubits = []
            for nqubits in self.nqubits_list:
                frame_potential_samples_each_nlayers = []
                for nlayers in self.nlayers_list:
                    exp = Expressibility2norm(circuit_type, nqubits, nlayers, self.nsamples)
                    frame_potential_samples_each_nlayers.append(exp.circuit_frame_potential())
                frame_potential_samples_each_nqubits.append(frame_potential_samples_each_nlayers)
            self.frame_potential_samples_each_circuit_type.append(frame_potential_samples_each_nqubits)
        return self.frame_potential_samples_each_circuit_type

    def frame_potential_analytical_samples_each_nqubits_(self):
        self.frame_potential_analytical_samples_each_nqubits = []
        for nqubits in self.nqubits_list:
            frame_potential_analytical_samples_each_nlayers = []
            for nlayers in self.nlayers_list:
                frame_potential_analytical_samples_each_nlayers.append(analytical_haar_frame_potential(nqubits))
            self.frame_potential_analytical_samples_each_nqubits.append(frame_potential_analytical_samples_each_nlayers)

        return self.frame_potential_analytical_samples_each_nqubits

    def plot_all(self, circuit_type):
        if circuit_type in ("TPA", "HEA", "ALT"):
            pass
        else:
            raise ValueError

        l = dict(zip(self.circuit_types, [0, 1, 2]))
        frame_potential_samples_each_nqubits = self.frame_potential_samples_each_circuit_type[l[circuit_type]]

        # plot frame potential vs nlayers
        for i in range(len(self.nqubits_list)):
            plt.semilogy(
                x=self.nlayers_list,
                y=frame_potential_samples_each_nqubits[i],
                label=f"circuit frame potential, nqubits={self.nqubits_list[i]}",
                marker="o",
            )

        for i in range(len(self.nqubits_list) - 1):
            plt.semilogy(
                x=self.nlayers_list,
                y=self.frame_potential_analytical_samples_each_nqubits[i],
                linestyle="dashed",
                color="black",
            )
        plt.semilogy(
            x=self.nlayers_list,
            y=self.frame_potential_analytical_samples_each_nqubits[-1],
            label="analytical haar frame potential",
            linestyle="dashed",
            color="black",
        )

        plt.xlabel("nlayers")
        plt.ylabel("frame potential")
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize=10)
        plt.show()

        # plot expressibility vs nlayers
        expressibility_each_nqubits = []
        for i in range(len(self.nqubits_list)):
            expressibility_each_nlayers = np.sqrt(
                np.array(frame_potential_samples_each_nqubits[i])
                - np.array(self.frame_potential_analytical_samples_each_nqubits[i])
            )
            expressibility_each_nqubits.append(expressibility_each_nlayers)
            plt.semilogy(
                x=self.nlayers_list,
                y=expressibility_each_nlayers,
                label=f"circuit expressibility, nqubits={self.nqubits_list[i]}",
                marker="o",
            )

        plt.xlabel("nlayers")
        plt.ylabel("expressibility")
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize=10)
        plt.show()
