# ETIC GenAI frontend

React + TypeScript app for FastAPI registration, sign-in, chat, document upload,
and administrator evaluations.

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
`/auth/sign-in`, `/chat`, `/documents/upload`, and `/admin/evaluations/run`
endpoints. If FastAPI uses another origin, set `API_PROXY_TARGET` in a local
`.env` file (see `.env.example`).

For a deployment, set `VITE_API_BASE_URL` to the FastAPI origin and configure
that API to allow requests from the frontend's origin. The sign-in bearer token
is kept in `sessionStorage` for the current browser tab/session. After signing
in, the app redirects to `/home/chat`; the workspace menu switches between chat
and document upload. Administrator accounts also see an **Evaluations** page
to run the dataset and view the experiment response. The API enforces
administrator access even if a regular user calls the endpoint directly. Chat
requests continue using the conversation ID returned by the API, and uploads
are private by default.
