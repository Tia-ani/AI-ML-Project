# 🚀 Deployment Guide

## Quick Start - Choose Your Platform

### ✅ Option 1: Streamlit Community Cloud (RECOMMENDED - Easiest & Free)

**Steps:**

1. **Push to GitHub** (if not already done):
   ```bash
   git add .
   git commit -m "Add deployment files"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to https://share.streamlit.io
   - Click "New app"
   - Sign in with GitHub
   - Select your repository: `Tia-ani/AI-ML-Project`
   - Main file path: `app.py`
   - Click "Advanced settings"
   - Add secrets (if using Google API):
     ```toml
     GOOGLE_API_KEY = "your_api_key_here"
     ```
   - Click "Deploy"

3. **Done!** Your app will be live at: `https://your-app-name.streamlit.app`

---

### Option 2: Render (Free Tier)

**Steps:**

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Add deployment files"
   git push origin main
   ```

2. **Deploy on Render**:
   - Go to https://render.com
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Render will auto-detect `render.yaml`
   - Add environment variable `GOOGLE_API_KEY` in dashboard
   - Click "Create Web Service"

3. **Done!** Your app will be live at: `https://your-app-name.onrender.com`

---

### Option 3: Heroku

**Steps:**

1. **Install Heroku CLI**:
   ```bash
   brew install heroku/brew/heroku  # macOS
   ```

2. **Login and create app**:
   ```bash
   heroku login
   heroku create your-churn-app-name
   ```

3. **Set environment variables**:
   ```bash
   heroku config:set GOOGLE_API_KEY=your_api_key_here
   ```

4. **Deploy**:
   ```bash
   git push heroku main
   ```

5. **Open app**:
   ```bash
   heroku open
   ```

---

### Option 4: Docker (Local or Cloud)

**Local Testing:**
```bash
docker build -t churn-app .
docker run -p 8501:8501 -e GOOGLE_API_KEY=your_key churn-app
```

**Deploy to Cloud:**
- **Google Cloud Run**: `gcloud run deploy --source .`
- **AWS ECS**: Use AWS Console or CLI
- **Azure Container Instances**: Use Azure Portal

---

### Option 5: AWS EC2

**Steps:**

1. **Launch EC2 instance** (Ubuntu 22.04, t2.medium recommended)

2. **SSH into instance**:
   ```bash
   ssh -i your-key.pem ubuntu@your-ec2-ip
   ```

3. **Setup**:
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-venv git -y
   git clone https://github.com/Tia-ani/AI-ML-Project.git
   cd AI-ML-Project
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

4. **Create .env file**:
   ```bash
   echo "GOOGLE_API_KEY=your_key_here" > .env
   ```

5. **Run with nohup**:
   ```bash
   nohup streamlit run app.py --server.port 8501 --server.address 0.0.0.0 > app.log 2>&1 &
   ```

6. **Configure Security Group**: Allow inbound traffic on port 8501

7. **Access**: `http://your-ec2-ip:8501`

---

## 📋 Pre-Deployment Checklist

- [ ] All files committed to Git
- [ ] `.env` file NOT committed (in .gitignore)
- [ ] API keys ready for environment variables
- [ ] `artifacts/` and `data/` folders included in repo
- [ ] Tested locally with `streamlit run app.py`

---

## 🔐 Environment Variables Needed

Add these to your hosting platform:

```
GOOGLE_API_KEY=your_google_api_key_here
```

---

## 🐛 Troubleshooting

**Issue**: App crashes on startup
- Check logs for missing dependencies
- Ensure all artifact files are present

**Issue**: Model not loading
- Verify `artifacts/model.pkl` exists in deployment
- Check file paths are relative, not absolute

**Issue**: Out of memory
- Upgrade to a plan with more RAM (minimum 1GB recommended)
- Consider lazy loading models

---

## 📊 Monitoring

After deployment, monitor:
- Response times
- Memory usage
- Error logs
- User traffic

---

## 🎯 Recommended: Start with Streamlit Cloud

It's the fastest way to get your app live with zero configuration!
