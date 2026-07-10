# Vera Challenge Bot — Setup Guide (Windows, beginner-friendly)

This folder contains your working bot. Follow these steps in order. Copy-paste
the commands exactly. If anything errors, copy the FULL error text back to me.

---

## STEP 0 — Check if Python is installed

1. Press `Win` key, type `cmd`, hit Enter (opens Command Prompt).
2. Type this and press Enter:
   ```
   python --version
   ```
3. **If you see something like `Python 3.11.5`** → great, skip to Step 1.
4. **If you see an error** ("not recognized") → you need to install Python:
   - Go to https://www.python.org/downloads/
   - Download the latest Windows installer and run it.
   - **IMPORTANT:** on the first installer screen, check the box that says
     **"Add python.exe to PATH"** before clicking Install.
   - After it finishes, close and reopen Command Prompt, then re-run
     `python --version` to confirm it works.

---

## STEP 1 — Open this folder in Command Prompt

1. Unzip this folder somewhere easy to find, e.g. `C:\Users\YourName\vera-bot`.
2. In Command Prompt, navigate there:
   ```
   cd C:\Users\YourName\vera-bot
   ```
   (Replace with wherever you actually unzipped it — you can also just type
   `cd ` with a trailing space, then drag the folder into the Command Prompt
   window, which auto-fills the path.)

---

## STEP 2 — Install the required packages

```
pip install -r requirements.txt
```

This downloads FastAPI, Uvicorn, and Requests. Takes ~30 seconds.

---

## STEP 3 — Get a free Groq API key, then set it

Groq gives a genuinely free API tier (no credit card) that's plenty for this
challenge. Get your key first:

1. Go to **https://console.groq.com**
2. Sign up with email or Google (no card needed)
3. Click **API Keys** in the left sidebar → **Create API Key**
4. Copy the key (starts with `gsk_...`)

Then, still in the same Command Prompt window, run (replace with your real key):

```
set GROQ_API_KEY=gsk_YOUR_ACTUAL_KEY_HERE
```

**Note:** `set` only lasts for this Command Prompt window. If you close it,
you'll need to re-run this line next time.

---

## STEP 4 — Run the bot locally

```
python -m uvicorn bot:app --host 0.0.0.0 --port 8080
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8080
```

Leave this window open — it IS your running server. Don't close it.

---

## STEP 5 — Test it's alive

Open a **second** Command Prompt window (keep the first one running the
server), and run:

```
curl http://localhost:8080/v1/healthz
curl http://localhost:8080/v1/metadata
```

You should get JSON back, e.g. `{"status":"ok",...}`.

---

## STEP 6 — Run the judge simulator against your bot

This simulates the real challenge judge talking to your bot end-to-end. It's
configured by editing a few lines at the TOP of `judge_simulator.py` (not
command-line flags) — open it in Notepad and edit lines ~24-33:

```python
BOT_URL = "http://localhost:8080"   # leave as-is, matches your local server
LLM_PROVIDER = "groq"                # change from "openai" to "groq"
LLM_API_KEY = "gsk_YOUR_ACTUAL_KEY_HERE"   # paste your real Groq key here
LLM_MODEL = "llama-3.3-70b-versatile" # IMPORTANT: don't leave blank! The
                                      # script's blank default is an older
                                      # model name that may be retired
```

Save the file, then in the second Command Prompt window (server still
running in the first window) run:

```
python judge_simulator.py
```

**Note:** this file already ships with a `dataset` folder next to it
(the original seed data), which the simulator reads automatically — you
don't need to run `generate_dataset.py` for this step. (The expanded
dataset is more useful for stress-testing your bot with more variety later
if you want, by pointing your own test scripts at it — but the official
simulator uses the seed set as shipped.)

Watch the output. You should see conversations being generated where Vera
sends real, specific, well-written messages (not the generic fallback text).
**If you instead see fallback-sounding messages** like "quick update on your
account", that means the Groq call is failing — check:
- Is `GROQ_API_KEY` actually set in *that* Command Prompt window? (Step 3
  must be run in the same window you're using, or use the `set` command
  again in this second window too.)
- Paste me anything printed in the FIRST window (the one running the
  server) — errors get logged there.

---

## STEP 7 — Push this code to GitHub

Render deploys from a GitHub repo, so:

1. Go to https://github.com and create a free account if you don't have one.
2. Click the `+` in the top right → **New repository**. Name it e.g.
   `vera-bot`. Keep it **Public** or **Private** (either works for Render's
   free tier). Don't add a README (you already have one).
3. Back in Command Prompt, in your bot folder:
   ```
   git init
   git add .
   git commit -m "Vera challenge bot"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/vera-bot.git
   git push -u origin main
   ```
   (If `git` isn't recognized, install it from https://git-scm.com/download/win
   first, then reopen Command Prompt and retry.)

**Do NOT commit your real API key.** This repo has no key in it — you'll
enter the key directly into Render's dashboard in Step 8, which is the safe
way to do it.

---

## STEP 8 — Deploy on Render (free tier)

1. Go to https://render.com and sign up (you can sign up with your GitHub
   account — makes the next step easier).
2. Click **New +** → **Web Service**.
3. Connect your GitHub account if asked, then select the `vera-bot` repo.
4. Fill in:
   - **Name:** anything, e.g. `vera-bot`
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn bot:app --host 0.0.0.0 --port $PORT`
5. Scroll to **Environment Variables**, click **Add Environment Variable**:
   - Key: `GROQ_API_KEY`
   - Value: your real key
6. Click **Create Web Service**. Wait a couple of minutes for the first
   deploy — you'll see build logs streaming.
7. Once it says **Live**, you'll get a public URL like
   `https://vera-bot-xxxx.onrender.com`.
8. Test it from your Command Prompt:
   ```
   curl https://vera-bot-xxxx.onrender.com/v1/healthz
   ```

---

## STEP 9 — Submit

Go back to the challenge portal (Step 3 of 5 in the Vera journey) and submit
your public base URL, e.g. `https://vera-bot-xxxx.onrender.com`.

**Free-tier note:** Render's free web services "sleep" after ~15 minutes of
no traffic, and the first request after sleeping takes ~30-50 seconds to
wake up. If the judge harness has a strict timeout, mention this or consider
a paid "always-on" tier if it becomes an issue — otherwise it's usually fine
for evaluation.

---

## Files in this folder

- `bot.py` — the actual bot server (all 5 endpoints)
- `requirements.txt` — Python packages needed
- `judge_simulator.py` — provided by magicpin, simulates the judge locally
  (edit the config lines at the top before running — see Step 6)
- `dataset/` — the original seed dataset (categories/merchants/customers/triggers),
  used automatically by `judge_simulator.py`. Run `python dataset/generate_dataset.py`
  if you also want a larger expanded dataset for extra testing.

## If something breaks

Copy-paste the exact error text back to me and tell me which Step you were
on. I'll help you fix it.
