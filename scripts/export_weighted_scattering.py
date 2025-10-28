"""Utility helpers for exporting weighted AMSET scattering rates."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

try:
    from amset.plot.rates import RatesPlotter
except ModuleNotFoundError as exc:  # pragma: no cover - convenience guard for local runs
    raise ModuleNotFoundError(
        "The amset package is required to export scattering rates."
    ) from exc


_K_BOLTZMANN = 8.617_333_262_145e-5  # eV/K


def _validate_indices(plotter: RatesPlotter, temperature_idx: int, doping_idx: int) -> None:
    if not (0 <= temperature_idx < len(plotter.temperatures)):
        raise IndexError(
            "temperature_idx out of range: "
            f"0 <= idx < {len(plotter.temperatures)} expected, "
            f"got {temperature_idx}."
        )

    if not (0 <= doping_idx < plotter.fermi_levels.shape[0]):
        raise IndexError(
            "doping_idx out of range: "
            f"0 <= idx < {plotter.fermi_levels.shape[0]} expected, "
            f"got {doping_idx}."
        )


def _resolve_scattering_index(
    plotter: RatesPlotter, scattering_label: str
) -> tuple[int, str]:
    labels = np.array(plotter.scattering_labels, dtype=str)
    matches = np.where(labels == scattering_label)[0]

    if matches.size == 0:
        available = ", ".join(labels)
        raise ValueError(
            f"Scattering mechanism '{scattering_label}' not found. "
            f"Available mechanisms: {available}"
        )

    return int(matches[0]), str(labels[matches[0]])


def _normalisation_energy(plotter: RatesPlotter, doping_idx: int, temperature_idx: int) -> float:
    if plotter.is_metal:
        # Metals only expose a single Fermi level so we use the actual value at
        # the requested temperature.
        return float(plotter.fermi_levels[doping_idx, temperature_idx])

    conduction_band_max = max(
        np.max(plotter.energies[spin][: plotter.vb_idx[spin] + 1])
        for spin in plotter.spins
    )
    return float(conduction_band_max)


def fermi_dirac_derivative(energies: np.ndarray, fermi_level: float, temperature: float) -> np.ndarray:
    """Derivative of the Fermi-Dirac distribution with respect to energy.

    A numerically-stable implementation that avoids overflowing the exponential
    for energies far from the Fermi level.
    """

    beta = 1.0 / (_K_BOLTZMANN * temperature)
    reduced_energy = (energies - fermi_level) * beta

    # f(E) = 1 / (1 + exp((E - Ef)/kT))
    # Use logaddexp for improved numerical stability.
    occupation = np.exp(-np.logaddexp(0.0, reduced_energy))
    return -(occupation * (1.0 - occupation)) * beta


def export_weighted_scattering_rate(
    filename: str | Path,
    *,
    temperature_idx: int,
    doping_idx: int,
    scattering_label: Literal["ADP", "IMP", "POP", "PIE", "PZA", "N" ] = "ADP",
    output_file: str | Path = "E-S-weighted.dat",
    normalize_energy: bool = True,
) -> Path:
    """Save the derivative-weighted scattering rate for a specific mechanism.

    Parameters
    ----------
    filename:
        Path to the AMSET HDF5 file (``mesh_*.h5``).
    temperature_idx:
        Index into :pyattr:`RatesPlotter.temperatures` specifying the selected
        temperature.
    doping_idx:
        Index into the doping axis used by :pyattr:`RatesPlotter.fermi_levels`.
    scattering_label:
        Label of the scattering mechanism to export. Defaults to ``"ADP"``.
    output_file:
        Target file path for the ``energy`` vs. ``weighted scattering rate``
        table.
    normalize_energy:
        Whether the exported energies should be referenced to the band edge
        (semiconductors) or Fermi level (metals), matching AMSET's plotting
        behaviour.
    """

    plotter = RatesPlotter(str(filename))
    _validate_indices(plotter, temperature_idx, doping_idx)
    scattering_idx, canonical_label = _resolve_scattering_index(plotter, scattering_label)

    energies = plotter.plot_energies.ravel().astype(float)
    raw_rates = plotter.plot_rates[scattering_idx, doping_idx, temperature_idx].ravel().astype(float)

    norm_shift = 0.0
    if normalize_energy:
        norm_shift = _normalisation_energy(plotter, doping_idx, temperature_idx)

    energies -= norm_shift
    fermi_level = float(plotter.fermi_levels[doping_idx, temperature_idx]) - norm_shift
    min_fd, max_fd = (np.asarray(plotter.fd_cutoffs, dtype=float) - norm_shift)

    energy_mask = (energies > min_fd) & (energies < max_fd)
    energies = energies[energy_mask]
    raw_rates = raw_rates[energy_mask]

    weights = np.abs(fermi_dirac_derivative(energies, fermi_level, float(plotter.temperatures[temperature_idx])))
    weighted_rates = raw_rates * weights

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        f"# Energy (eV)    Weighted {canonical_label} Scattering Rate (s^-1)\n"
        "# Energies are referenced to the band edge/Fermi level following AMSET's plots."
    )

    data = np.column_stack((energies, weighted_rates))
    np.savetxt(output_path, data, header=header, comments="")

    return output_path


if __name__ == "__main__":  # pragma: no cover - convenience CLI shim
    import argparse

    parser = argparse.ArgumentParser(description="Export weighted AMSET scattering rates.")
    parser.add_argument("filename", type=Path, help="AMSET mesh HDF5 file")
    parser.add_argument("output", type=Path, nargs="?", default=Path("E-S-weighted.dat"))
    parser.add_argument("--temperature-idx", type=int, default=12, help="Index of the target temperature")
    parser.add_argument("--doping-idx", type=int, default=4, help="Index of the target doping level")
    parser.add_argument(
        "--mechanism",
        type=str,
        default="ADP",
        help="Scattering mechanism label to export (e.g. ADP, POP, IMP)",
    )
    parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Disable the energy normalization applied by AMSET plots.",
    )

    args = parser.parse_args()
    export_weighted_scattering_rate(
        args.filename,
        temperature_idx=args.temperature_idx,
        doping_idx=args.doping_idx,
        scattering_label=args.mechanism,
        output_file=args.output,
        normalize_energy=args.normalize,
    )
