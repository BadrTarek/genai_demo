# ETIC GenAI frontend

React + TypeScript app for FastAPI registration, sign-in, chat, and document
upload.

## Run locally

Start FastAPI from the repository root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn etic_genai.api.main:app --reload
```

Then, from this folder:

```powershell
npm install
npm run dev
```

Open the URL printed by Vite. The development server proxies `/api` requests to
`http://127.0.0.1:8000`, mapping them to the API's `/auth/register`,
`/auth/sign-in`, `/chat`, and `/documents/upload` endpoints. If FastAPI uses another origin, set
`API_PROXY_TARGET` in a local `.env` file (see `.env.example`).

For a deployment, set `VITE_API_BASE_URL` to the FastAPI origin and configure
that API to allow requests from the frontend's origin. The sign-in bearer token
is kept in `sessionStorage` for the current browser tab/session and read from
there for each authenticated request (`Authorization: Bearer <token>`). After
signing in, the app redirects to `/home/chat`; the workspace menu switches
between chat and document upload. Chat requests continue using the conversation
ID returned by the API, and uploads are private by default.
