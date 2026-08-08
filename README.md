# BOLEH

**BOLEH** is a Python package for solving flavour-covariant Boltzmann equations in baryogenesis scenarios.

The package currently provides implementations for:

* Standard Model flavour-covariant evolution.
* Lepton-covariant evolution.
* Quark-covariant evolution.

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/smcubidesperez/BOLEH.git
cd BOLEH
pip install -e .
```

## Structure

The main package is organized as:

```text
src/BOLEH/
├── SMCovariant/
├── Leptoncovariant/
└── Quarkcovariant/
```

The `Examples/` directory contains Jupyter notebooks and example results for different baryogenesis scenarios.

## Implementation

BOLEH formulates the Boltzmann equations in terms of flavour-covariant charge matrices and solves the resulting stiff system numerically.

New-physics interactions can be incorporated through source and washout contributions to the existing Boltzmann equations.

## Examples

Examples currently include:

* Type-I seesaw.
* Type-II seesaw.
* Cloistered baryogenesis.