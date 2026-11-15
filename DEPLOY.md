# Putting THEIA online under your own domain

THEIA is a Streamlit website (`streamlit_app.py`). Running it on your own computer first:

```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Then open http://localhost:8501.

## What the server needs

- **About 1.5 GB of RAM** (2 GB recommended). The model, the global data and PyTorch stay loaded
  for fast answers. The smallest 512 MB plans will run out of memory.
- **About 2 GB of disk** for the Docker image.
- Internet access for visitors' browsers to stream the NASA Trek and Sentinel-2 detail layers. The
  core globe texture, model and data are all served by your own server.

> Streamlit Community Cloud (streamlit.io) is free but only gives `yourapp.streamlit.app`
> addresses. It can't use your own domain, so use one of the options below.

---

## Option A (recommended): a small cloud server + Docker + automatic HTTPS

About 5 USD a month (for example Hetzner CX22, DigitalOcean 2 GB, or AWS Lightsail 2 GB). You
control everything, and HTTPS is automatic.

1. **Buy the domain** from any registrar (Namecheap, Cloudflare, Porkbun, GoDaddy...).
2. **Create the server:** Ubuntu 24.04, 2 GB RAM. Note its public IP address.
3. **Point the domain at it.** In your registrar's DNS settings, add an `A` record:
   - Host: `@` (or `theia` for `theia.yourdomain.com`)
   - Value: the server's IP address

   It usually takes a few minutes to start working.
4. **Install Docker on the server** (log in with `ssh root@SERVER_IP`):
   ```
   curl -fsSL https://get.docker.com | sh
   ```
5. **Copy THEIA to the server.** Either `git clone` your repository, or upload the zip and unzip it:
   ```
   scp THEIA.zip root@SERVER_IP:~
   ```
6. **Set your domain.** Edit `Caddyfile` and replace `theia.example.com` with your domain.
7. **Start it:**
   ```
   cd THEIA
   docker compose up -d --build
   ```
   The first build takes about 5 minutes. Caddy then gets an HTTPS certificate on its own.
8. Open `https://yourdomain.com`. Done.

**Updating later:** copy the new files to the server, then run `docker compose up -d --build` again.

---

## Option B: a platform that builds the Dockerfile for you

These are easier (no server to manage) but usually cost more for 2 GB of RAM.

| Platform | How | Custom domain |
|---|---|---|
| **Railway** | New Project, Deploy from GitHub repo (it finds the `Dockerfile`) | Settings, Networking, Custom Domain, then add the CNAME it shows at your registrar |
| **Render** | New Web Service, pick the repo, runtime Docker, instance with at least 2 GB | Settings, Custom Domains |
| **Fly.io** | `fly launch` (uses the Dockerfile), then `fly scale memory 2048` | `fly certs add yourdomain.com` |
| **Google Cloud Run** | `gcloud run deploy --source . --memory 2Gi` | Domain mappings, or a load balancer |

All of them set a `$PORT` variable; the Dockerfile already uses it.

---

## Notes

- **Google Photorealistic 3D Tiles are not used on the website.** An API key put in a public web
  page can be copied by anyone. The web version uses THEIA's own MODIS globe plus Sentinel-2
  from Microsoft Planetary Computer, which need no key.
- **The repository must contain `assets/` and `static/`** (the trained model, data and textures),
  otherwise the app starts empty.
- **Traffic:** each search takes well under a second on the server. Live training on the "Watch the
  AI think" page uses the CPU for about a minute per click.
- **Licences:** keep the data credits visible (the app shows them under the globe and on the
  *Data & method* page).
