import matplotlib.pyplot as plt
import numpy as np

from .expressibility_pennylane import analytical_haar_frame_potential, Expressibility2norm_pennylane


class do_all_pennylane:
    def __init__(self, circuit_types: list[str], nqubits_list: list[int], nlayers_list: list[int], nsamples: int):
        self.circuit_types = circuit_types
        self.nqubits_list = nqubits_list
        self.nlayers_list = nlayers_list
        self.nsamples = nsamples

    def frame_potential_samples_each_circuit_type_(self) -> tuple[list[list[list[float]]], list[list[list[float]]]]:
        self.frame_potential_samples_each_circuit_type_mean = []
        self.frame_potential_samples_each_circuit_type_std = []
        for circuit_type in self.circuit_types:
            frame_potential_samples_each_nqubits_mean = []
            frame_potential_samples_each_nqubits_std = []
            for nqubits in self.nqubits_list:
                frame_potential_samples_each_nlayers_mean = []
                frame_potential_samples_each_nlayers_std = []
                for nlayers in self.nlayers_list:
                    exp = Expressibility2norm_pennylane(circuit_type, nqubits, nlayers, self.nsamples)
                    mean, std = exp.circuit_frame_potential()
                    frame_potential_samples_each_nlayers_mean.append(mean)
                    frame_potential_samples_each_nlayers_std.append(std)
                frame_potential_samples_each_nqubits_mean.append(frame_potential_samples_each_nlayers_mean)
                frame_potential_samples_each_nqubits_std.append(frame_potential_samples_each_nlayers_std)
            self.frame_potential_samples_each_circuit_type_mean.append(frame_potential_samples_each_nqubits_mean)
            self.frame_potential_samples_each_circuit_type_std.append(frame_potential_samples_each_nqubits_std)
        return self.frame_potential_samples_each_circuit_type_mean, self.frame_potential_samples_each_circuit_type_std

    def frame_potential_analytical_samples_each_nqubits_(self) -> list[list[float]]:
        self.frame_potential_analytical_samples_each_nqubits = []
        for nqubits in self.nqubits_list:
            frame_potential_analytical_samples_each_nlayers = []
            for nlayers in self.nlayers_list:
                frame_potential_analytical_samples_each_nlayers.append(analytical_haar_frame_potential(nqubits))
            self.frame_potential_analytical_samples_each_nqubits.append(frame_potential_analytical_samples_each_nlayers)

        return self.frame_potential_analytical_samples_each_nqubits

    def plot_all(self, circuit_type: str) -> None:
        if circuit_type in ("TPA", "HEA", "ALT"):
            pass
        else:
            raise ValueError

        l = dict(zip(self.circuit_types, [0, 1, 2]))
        frame_potential_samples_each_nqubits_mean = self.frame_potential_samples_each_circuit_type_mean[l[circuit_type]]

        # plot frame potential vs nlayers
        for i in range(len(self.nqubits_list)):
            plt.semilogy(
                self.nlayers_list,
                frame_potential_samples_each_nqubits_mean[i],
                label=f"nqubits={self.nqubits_list[i]}",
                marker="o",
                linestyle="--",
            )

        for i in range(len(self.nqubits_list) - 1):
            plt.semilogy(
                self.nlayers_list,
                self.frame_potential_analytical_samples_each_nqubits[i],
                linestyle="--",
                color="black",
            )
        plt.semilogy(
            self.nlayers_list,
            self.frame_potential_analytical_samples_each_nqubits[-1],
            label="Haar",
            linestyle="--",
            color="black",
        )

        plt.xlabel("nlayers")
        plt.ylabel("frame potential")
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize=18)
        plt.show()

        # plot expressibility vs nlayers
        expressibility_each_nqubits = []
        for i in range(1,len(self.nqubits_list)):
            expressibility_each_nlayers = np.array(frame_potential_samples_each_nqubits_mean[i]) - np.array(self.frame_potential_analytical_samples_each_nqubits[i])
            expressibility_each_nqubits.append(expressibility_each_nlayers)
            plt.semilogy(
                self.nlayers_list,
                expressibility_each_nlayers,
                label=f"n={self.nqubits_list[i]}",
                marker="o",
                linestyle="--",
            )
            # plt.errorbar(
            #     self.nlayers_list,
            #     expressibility_each_nlayers,
            #     label=f"nqubits={self.nqubits_list[i]}",
            #     yerr=self.frame_potential_samples_each_circuit_type_std[l[circuit_type]][i],
            #     fmt='o',
            #     capsize=3
            # )
        
        plt.ylim(10**(-4.5), 0.4)
        plt.yscale('log')

        plt.xlabel("L", fontsize=18)
        plt.ylabel("$\epsilon^2$", fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        # plt.legend(bbox_to_anchor=(0.99, 0.99), loc="upper right", borderaxespad=0, fontsize=10)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize=18)
        if circuit_type == "TPA":
            plt.savefig("tpa-circuit-exp.pdf", bbox_inches="tight")
        elif circuit_type == "ALT":
            plt.savefig("alt-circuit-exp.pdf", bbox_inches="tight")
        elif circuit_type == "HEA":
            plt.savefig("hea-circuit-exp.pdf", bbox_inches="tight")
        plt.show()
