# DSPG documentation index

This directory extends the root [`README.md`](../README.md) (install and quick start) with a structured description of **module roles, call relationships, and `results/` / figure outputs** for reproducing experiments and downstream development.

| Document | Contents |
|----------|----------|
| [**Architecture & workflows**](architecture.md) | Repository layout, partial equilibrium (PE) vs general equilibrium (GE), data flow |
| [**Python modules**](python-modules.md) | Each `dspg/*.py`: purpose, dependencies, typical CLI entry points |
| [**Notebooks & artifacts**](notebooks-and-artifacts.md) | `dspg/notebooks/`, `results/`, `figures_tables/` |
| [**Quick replication (figures & tables)**](instruction-for-quick-replication.md) | Which scripts and experiments regenerate `figures_tables/` outputs |

**Suggested reading order:** for a **fast path to reproduce paper figures**, start with [**instruction-for-quick-replication.md**](instruction-for-quick-replication.md). For deeper context, read [**architecture.md**](architecture.md) and [**python-modules.md**](python-modules.md) as needed.
