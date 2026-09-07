# IS4001 G1 — Smart Building Analytics (CU-BEMS)

Final project for IS4001, using the CU-BEMS Smart Building Energy & Indoor Environmental Data dataset (7 floors, Jul 2018–Dec 2019, 1-minute sampling — AC power, lighting, plug loads, temperature, humidity, ambient light).

Kaggle source: https://www.kaggle.com/datasets/claytonmiller/cubems-smart-building-energy-and-iaq-data/data

---

## Getting set up

Hopefully everyone still remembers their basics from AF 🙂
Here's the full rundown for this just in case, so nobody's stuck.

### 1. Install VS Code (if you don't have it)

Download from [code.visualstudio.com](https://code.visualstudio.com). Once installed, add these extensions (Extensions panel on the left, or `Cmd/Ctrl+Shift+X`):

- **Python**
- **Jupyter**

### 2. Clone the repo

Open a terminal (VS Code has one built in: ``Ctrl+` ``) and run:

```bash
git clone https://github.com/<org-or-user>/is4001-g1-smart-building.git
cd is4001-g1-smart-building
```

Then open the folder in VS Code: `File → Open Folder` → select `is4001-g1-smart-building`.

### 3. Set up your Python environment

Still in the terminal, inside the repo folder:

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

When you open the notebook in `workspace/`, make sure VS Code is using this `venv` as the kernel (top-right corner of the notebook — click it and select the `venv` interpreter).

---

## Repo structure

```
is4001-g1-smart-building/
├── README.md
├── requirements.txt
├── data/                     # CU-BEMS CSVs
└── workspace/                # notebooks — explore here
    └── cubems_eda_starter.ipynb
```

---

## Floor assignments

Each person owns exploration for their floor (both 2018 and 2019 files):

| Floor   | Owner    |
| ------- | -------- |
| Floor 1 | Leyna    |
| Floor 2 | Shree    |
| Floor 3 | Mik      |
| Floor 4 | Kenneth  |
| Floor 5 | Terrence |
| Floor 6 | Williams |
| Floor 7 | Sana     |

---

## How we're working on this

- **`workspace/` is your sandbox** — open the starter notebook (or make your own copy) and go explore. Try things, break things, that's the point right now.
- **Don't commit/push yet.** Keep your exploration local for now while everyone's poking around independently — no need to worry about conflicts or cluttering the repo with half-finished notebooks.
- **Once we've actually got findings worth keeping**, we'll regroup, compare notes, and push the good stuff into the repo together (probably via a proper branch + PR at that point, or we'll just decide as a group).

If you get stuck on setup, shout in the group chat before burning too much time on it.
