# 🚀 Deployment Guide - Market Wars Simulator

This guide covers multiple deployment options for your Market Wars Simulator application.

---

## 📋 Pre-Deployment Checklist

Before deploying, ensure:
- ✅ `.gitignore` excludes `.venv/`, `__pycache__/`, and large data files
- ✅ All hardcoded paths are removed (use relative paths)
- ✅ `requirements.txt` is up to date
- ✅ `.streamlit/config.toml` is configured
- ✅ Code is pushed to GitHub

---

## Option 1: Streamlit Community Cloud ⭐ (RECOMMENDED)

**Best for:** Quick deployment, free hosting, automatic updates

### Prerequisites
- GitHub account
- Your repository is public or you have Streamlit Cloud Pro

### Steps:

1. **Go to Streamlit Cloud**
   - Visit: https://share.streamlit.io/
   - Sign in with GitHub

2. **Deploy New App**
   - Click "New app"
   - Select your repository: `nitesh0007-edith/market-wars-simulator`
   - Main file path: `app.py`
   - App URL: Choose a custom name (e.g., `market-wars-simulator`)

3. **Configure (if needed)**
   - Python version: 3.9+ (auto-detected)
   - Click "Deploy!"

4. **Your app will be live at:**
   ```
   https://[your-app-name].streamlit.app
   ```

### Monitoring
- View logs in Streamlit Cloud dashboard
- Auto-redeploys on git push
- Free tier: 1GB RAM, shared resources

---

## Option 2: Render.com (Free Tier Available)

**Best for:** More control, custom domains, database support

### Steps:

1. **Create Web Service on Render**
   - Go to https://render.com
   - New → Web Service
   - Connect your GitHub repo

2. **Configure Build Settings**
   ```yaml
   Build Command: pip install -r requirements.txt
   Start Command: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
   ```

3. **Environment Variables** (Optional)
   ```
   PYTHON_VERSION=3.9.0
   ```

4. **Deploy**
   - Click "Create Web Service"
   - Free tier: 512MB RAM, auto-sleep after 15min inactivity

### Custom Domain
- Add custom domain in Render dashboard
- Free SSL included

---

## Option 3: Heroku

**Best for:** Established platform, add-ons ecosystem

### Prerequisites
- Heroku account
- Heroku CLI installed

### Files Needed:

**1. Create `Procfile`:**
```bash
web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

**2. Create `setup.sh`:**
```bash
mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

**3. Update `Procfile`:**
```bash
web: sh setup.sh && streamlit run app.py
```

### Deploy Commands:
```bash
# Login to Heroku
heroku login

# Create app
heroku create market-wars-simulator

# Push to Heroku
git push heroku main

# Open app
heroku open
```

### Cost:
- Free tier (deprecated) - Eco Dyno: $5/month
- Basic: $7/month

---

## Option 4: Railway.app

**Best for:** Simple deployment, generous free tier

### Steps:

1. **Go to Railway**
   - Visit https://railway.app
   - Sign up with GitHub

2. **New Project**
   - "New Project" → "Deploy from GitHub repo"
   - Select `market-wars-simulator`

3. **Configure**
   - Railway auto-detects Python
   - Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

4. **Deploy**
   - Automatic deployment on every git push
   - Free tier: $5 credit/month (approx. 500 hours)

---

## Option 5: Google Cloud Platform (Cloud Run)

**Best for:** Scalability, enterprise needs

### Steps:

1. **Create `Dockerfile`:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD streamlit run app.py --server.port 8080 --server.address 0.0.0.0
```

2. **Deploy to Cloud Run:**
```bash
# Install gcloud CLI
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login

# Build and deploy
gcloud run deploy market-wars-simulator \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

3. **Cost:**
   - Free tier: 2 million requests/month
   - Pay-per-use after that

---

## Option 6: AWS (Elastic Beanstalk or EC2)

**Best for:** Full control, AWS ecosystem integration

### Using Elastic Beanstalk:

1. **Install EB CLI:**
```bash
pip install awsebcli
```

2. **Initialize:**
```bash
eb init -p python-3.9 market-wars-simulator
```

3. **Create environment:**
```bash
eb create market-wars-env
```

4. **Deploy:**
```bash
eb deploy
```

---

## Option 7: DigitalOcean App Platform

**Best for:** Developer-friendly, predictable pricing

### Steps:

1. **Create App**
   - Go to https://cloud.digitalocean.com/apps
   - Create App → GitHub → Select repo

2. **Configure**
   ```
   Run Command: streamlit run app.py --server.port 8080 --server.address 0.0.0.0
   HTTP Port: 8080
   ```

3. **Deploy**
   - Basic tier: $5/month
   - Includes 512MB RAM

---

## 🎯 Quick Start: Streamlit Cloud (5 Minutes)

**Fastest way to get online:**

```bash
# 1. Ensure code is clean
git status

# 2. Commit any changes
git add .
git commit -m "Prepare for deployment"

# 3. Push to GitHub
git push origin main

# 4. Go to https://share.streamlit.io and deploy!
```

---

## 🔧 Troubleshooting

### Common Issues:

**1. App crashes on startup**
- Check Python version compatibility (use 3.9+)
- Verify all imports are in requirements.txt
- Check logs for missing dependencies

**2. Module not found errors**
- Ensure relative imports use `from . import`
- Check PYTHONPATH configuration
- Verify directory structure is correct

**3. Memory errors**
- Reduce dataset size for free tiers
- Use lazy loading for large data
- Consider paid tier with more RAM

**4. Slow performance**
- Cache expensive computations with `@st.cache_data`
- Minimize GA population/generations on free tier
- Use session state wisely

---

## 📊 Platform Comparison

| Platform | Free Tier | RAM | Custom Domain | Auto-Deploy | Ease |
|----------|-----------|-----|---------------|-------------|------|
| **Streamlit Cloud** | ✅ | 1GB | ❌ | ✅ | ⭐⭐⭐⭐⭐ |
| **Render** | ✅ | 512MB | ✅ | ✅ | ⭐⭐⭐⭐ |
| **Railway** | $5 credit | 512MB | ✅ | ✅ | ⭐⭐⭐⭐ |
| **Heroku** | ❌ | 512MB | ✅ | ✅ | ⭐⭐⭐ |
| **GCP Cloud Run** | ✅ | 2GB | ✅ | Manual | ⭐⭐⭐ |
| **AWS EB** | 1 year | 1GB | ✅ | Manual | ⭐⭐ |
| **DigitalOcean** | ❌ | 512MB | ✅ | ✅ | ⭐⭐⭐ |

---

## 🎨 Post-Deployment Optimizations

### Performance Tips:

1. **Cache Expensive Operations:**
```python
@st.cache_data
def run_simulation(params):
    # Your expensive simulation code
    return results
```

2. **Reduce GA Parameters on Free Tier:**
```python
# For deployment, use smaller values
if os.getenv("DEPLOYMENT") == "streamlit":
    pop_size = 8  # instead of 12
    gens = 5      # instead of 10
```

3. **Add Loading Indicators:**
```python
with st.spinner("Running simulation..."):
    results = run_simulation()
```

---

## 📞 Support

- **Streamlit Docs:** https://docs.streamlit.io/streamlit-community-cloud
- **Community Forum:** https://discuss.streamlit.io
- **GitHub Issues:** https://github.com/nitesh0007-edith/market-wars-simulator/issues

---

## ✅ Deployment Verification

After deployment, test:
- [ ] App loads without errors
- [ ] Market simulation runs successfully
- [ ] GA optimization completes
- [ ] Charts render properly
- [ ] Download buttons work
- [ ] Navigation between pages works
- [ ] Performance is acceptable

---

**Next Steps:** Choose a platform and deploy! Start with Streamlit Cloud for the easiest experience.
