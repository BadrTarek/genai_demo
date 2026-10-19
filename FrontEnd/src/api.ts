const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "/api").replace(
  /\/+$/,
  "",
);
const ACCESS_TOKEN_KEY = "etic_access_token";
const USER_KEY = "etic_user";

export interface User {
  user_id: string;
  email: string;
  is_admin: boolean;
}

export interface SignInResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  user: User;
}

export interface RetrievedSource {
  document_title: string;
  source: string;
  section: string | null;
  score: number;
}

export interface ChatResponse {
  conversation_id: string;
  response: string;
  sources: RetrievedSource[];
}

export interface UploadResponse {
  document_id: string;
  document_title: string;
  chunk_count: number;
}

export interface EvaluationResponse {
  status: "completed";
  experiment_name: string;
  experiment_id: string;
  experiment_url: string | null;
}

interface ApiErrorBody {
  detail?: string | Array<{ msg?: string; loc?: Array<string | number> }>;
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export function saveSession(session: SignInResponse): void {
  sessionStorage.setItem(ACCESS_TOKEN_KEY, session.access_token);
  sessionStorage.setItem(USER_KEY, JSON.stringify(session.user));
}

export function getSavedSession(): SignInResponse | null {
  const accessToken = sessionStorage.getItem(ACCESS_TOKEN_KEY);
  const serializedUser = sessionStorage.getItem(USER_KEY);
  if (!accessToken || !serializedUser) return null;

  try {
    const user = JSON.parse(serializedUser) as User;
    if (typeof user.user_id !== "string" || typeof user.email !== "string") {
      throw new Error("Saved user data is invalid.");
    }
    return {
      access_token: accessToken,
      token_type: "bearer",
      expires_in: 0,
      user: { ...user, is_admin: user.is_admin === true },
    };
  } catch {
    clearSavedSession();
    return null;
  }
}

export function clearSavedSession(): void {
  sessionStorage.removeItem(ACCESS_TOKEN_KEY);
  sessionStorage.removeItem(USER_KEY);
}

function getSavedAccessToken(): string {
  const accessToken = sessionStorage.getItem(ACCESS_TOKEN_KEY);
  if (!accessToken) {
    throw new ApiError("Your session has expired. Please log in again.", 401);
  }
  return accessToken;
}

async function readResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as ApiErrorBody | null;
    const detail = body?.detail;
    const message = Array.isArray(detail)
      ? detail.map((item) => item.msg || "Invalid input").join(" ")
      : detail || `Request failed (${response.status}).`;
    throw new ApiError(message, response.status);
  }

  return (await response.json()) as T;
}

async function post<T>(
  path: string,
  payload: object,
  requiresAuthentication = false,
): Promise<T> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (requiresAuthentication) {
    headers.Authorization = `Bearer ${getSavedAccessToken()}`;
  }

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
    });
  } catch {
    throw new ApiError(
      "Could not reach the API. Check that the FastAPI server is running.",
      0,
    );
  }

  return readResponse<T>(response);
}

export function register(email: string, password: string): Promise<User> {
  return post<User>("/auth/register", { email, password });
}

export function signIn(
  email: string,
  password: string,
): Promise<SignInResponse> {
  return post<SignInResponse>("/auth/sign-in", { email, password });
}

export function sendChatMessage(
  message: string,
  conversationId?: string,
): Promise<ChatResponse> {
  return post<ChatResponse>(
    "/chat",
    { message, ...(conversationId ? { conversation_id: conversationId } : {}) },
    true,
  );
}

export async function uploadDocument(
  file: File,
  visibility: "private" | "public",
): Promise<UploadResponse> {
  const accessToken = getSavedAccessToken();
  const formData = new FormData();
  formData.append("file", file);
  formData.append("visibility", visibility);

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/documents/upload`, {
      method: "POST",
      headers: { Authorization: `Bearer ${accessToken}` },
      body: formData,
    });
  } catch {
    throw new ApiError(
      "Could not reach the API. Check that the FastAPI server is running.",
      0,
    );
  }

  return readResponse<UploadResponse>(response);
}

export function runEvaluation(): Promise<EvaluationResponse> {
  return post<EvaluationResponse>("/admin/evaluations/run", {}, true);
}
