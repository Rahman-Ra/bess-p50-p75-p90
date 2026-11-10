# Deploying the Streamlit UI (public link)

The app is a normal Streamlit app. The entry point is:

```text
bess_app_v4_fixed.py
```

The repository must contain at least:

```text
bess_app_v4_fixed.py
requirements.txt
.streamlit/config.toml
sample_data/battery_input_annual_p50.csv
sample_data/battery_input_hourly_p50.csv
```

`quantile_scenarios.py`, `run_scenario_experiment.py`, `run_experiment.py` and
the docs are not needed at runtime but are included so the method can be
reproduced.

## Option A — Streamlit Community Cloud (free, recommended)

Streamlit Community Cloud gives a public `https://<app>.streamlit.app` URL that
anyone can open. It deploys from a GitHub repository and branch.

### 1. Public GitHub repository (already created)

The package is already published at:

```text
https://github.com/Rahman-Ra/bess-p50-p75-p90
```

The default branch is `rahman` (not `main`), and it contains
`bess_app_v4_fixed.py` at the repository root.

### 2. Push updates to the `rahman` branch

From this package folder:

```powershell
git add -A
git commit -m "describe the change"
git push origin rahman
```

The branch name is `rahman`, not `main`.

If git asks for credentials, use a GitHub Personal Access Token as the password
(Settings → Developer settings → Personal access tokens → Fine-grained tokens →
repository access to the new repo, Contents: Read and write).

### 3. Deploy

1. Go to <https://share.streamlit.io> and sign in with GitHub (the account that
   owns the repository).
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Repository: `Rahman-Ra/bess-p50-p75-p90`
4. Branch: `rahman`
5. Main file path: `bess_app_v4_fixed.py`
6. Optionally set the app URL subdomain, e.g. `bess-p50-p75-p90`.
7. Click **Deploy**.

Signing in is the only manual step: Streamlit Cloud requires the repository
owner's GitHub session, so it cannot be completed from a script or token.

The first build takes a few minutes. The resulting URL looks like:

```text
https://bess-p50-p75-p90.streamlit.app
```

Share that URL with your sir. Anyone can open it; no login is required.

### 4. After deployment

- Upload `sample_data/battery_input_annual_p50.csv` (download it from the repo
  first) in the **Upload BESSTOOL_INPUT** box.
- Choose P50 / P75 / P90 in the **Generation scenario** selector.
- Click **Run BESS dispatch**.

Streamlit Community Cloud apps sleep after a period of inactivity. The first
visit after sleeping takes about 30–60 seconds to wake. This is normal on the
free tier.

## Option B — Hugging Face Spaces (free alternative)

1. Create a new Space → SDK: **Streamlit**.
2. Upload the files from this package to the Space repository.
3. Set the entry point to `bess_app_v4_fixed.py` (Space settings → App file).
4. The Space gives a public `https://<user>-<space>.hf.space` URL.

Hugging Face Spaces does not require a Streamlit Cloud account.

## Option C — Temporary tunnel from this computer

If you only need a link for a short demo and cannot push to GitHub:

```powershell
# terminal 1
python -m streamlit run bess_app_v4_fixed.py --server.port 8501

# terminal 2 (if cloudflared is installed)
cloudflared tunnel --url http://localhost:8501
```

`cloudflared` prints a temporary `https://<random>.trycloudflare.com` URL. The
link dies when the tunnel or the app stops, and the computer must stay on, so
this is only suitable for a live demo.

## Troubleshooting

**`Port 8501 is already in use`** — run on another port:

```powershell
python -m streamlit run bess_app_v4_fixed.py --server.port 8502
```

**The scenario selector does not appear** — the uploaded file has no
`solar_p75_MWh` / `wind_p75_MWh` columns. Upload one of the files in
`sample_data/`; both already contain all three scenarios.

**`ModuleNotFoundError`** — install the pinned dependencies:

```powershell
pip install -r requirements.txt
```

**The app builds but crashes on upload** — confirm the file has the columns
`Year, Day, Month, Hour, solar_MWh, wind_MWh`. The app also accepts the scenario
columns `solar_p50_MWh`, `solar_p75_MWh`, `solar_p90_MWh` and the matching wind
columns.

**Streamlit Cloud cannot find the app** — the main file path must be exactly
`bess_app_v4_fixed.py`, and the selected branch must contain that file at the
repository root.
