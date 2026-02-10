No. Render deploys **once** when you push code to GitHub, then the app runs continuously as a web server.

**Here's the flow:**

1. **You push to GitHub** → Render detects the change
2. **Render deploys** (installs dependencies, runs your app)
3. **App runs 24/7** on Render's servers and handles user requests
4. **No redeploy on requests** — your app stays running, serving traffic

So users can access your app repeatedly without any redeploy happening.

**Important caveat for your app:**

Your app loads data from CSV files at startup:
```python
poverty_data = pd.read_csv(data_dir / 'PovStatsData.csv')
poverty = pd.read_csv(data_dir / 'poverty.csv', low_memory=False)
```

Make sure those CSV files are **committed to GitHub**, so they're included when Render clones your repo. If they're in .gitignore, Render won't have them and the app will crash.

Check:
```bash
git ls-files | grep .csv
```

If your CSVs aren't showing up, remove them from .gitignore (if they're there) and commit them.