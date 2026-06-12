# Hz QPO Search — Fermi GBM

Leahy PDS and cross-spectrum QPO search for **GRB 220910A** (NaI-b), using [Stingray](https://stingray.science/). 
Follows Caballero-García et al. (2024), Table 2 (Fermi/nb).

## Run

```bash
git clone <your-repo-url>
cd <repo-name>

pip install numpy scipy astropy matplotlib pandas ipywidgets jupyter stingray
jupyter notebook hz_qpo_search.ipynb
```

**Kernel → Restart & Run All.** 

## Files

- `hz_qpo_search.ipynb` — main notebook (single-detector PDS + cross-spectrum)
- `test-data/` — barycentred TTE files

