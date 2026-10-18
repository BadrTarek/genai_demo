import { useRef, useState, type FormEvent } from "react";
import { Link, NavLink, Navigate, Route, Routes, useLocation } from "react-router-dom";
import {
  ApiError,
  runEvaluation,
  sendChatMessage,
  uploadDocument,
  type ChatResponse,
  type EvaluationResponse,
  type SignInResponse,
  type UploadResponse,
} from "./api";

const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
const SUPPORTED_EXTENSIONS = [".pdf", ".md", ".markdown", ".txt", ".csv"];

interface HomePageProps {
  user: SignInResponse["user"];
  onSignOut: () => void;
}

export default function HomePage({ user, onSignOut }: HomePageProps) {
  const location = useLocation();
  const currentPage = location.pathname.includes("/documents")
    ? "documents"
    : location.pathname.includes("/evaluations")
      ? "evaluations"
      : "chat";

  return (
    <main className="workspace-shell">
      <aside className="workspace-sidebar">
        <Link className="workspace-brand" to="/home/chat">
          <span className="workspace-brand-mark" aria-hidden="true">e</span>
          <span>etic<span>/genai</span></span>
        </Link>
        <div className="workspace-nav-label">WORKSPACE</div>
        <nav className={`workspace-nav${user.is_admin ? " admin-nav" : ""}`} aria-label="Workspace">
          <NavLink
            className={({ isActive }) => `workspace-nav-item${isActive ? " selected" : ""}`}
            to="/home/chat"
          >
            <span className="nav-icon chat-icon" aria-hidden="true">◌</span>
            Chat
          </NavLink>
          <NavLink
            className={({ isActive }) => `workspace-nav-item${isActive ? " selected" : ""}`}
            to="/home/documents"
          >
            <span className="nav-icon document-icon" aria-hidden="true">▤</span>
            Document upload
          </NavLink>
          {user.is_admin && (
            <NavLink
              className={({ isActive }) => `workspace-nav-item${isActive ? " selected" : ""}`}
              to="/home/evaluations"
            >
              <span className="nav-icon" aria-hidden="true">◷</span>
              Evaluations
            </NavLink>
          )}
        </nav>
        <div className="sidebar-bottom">
          <div className="sidebar-tip">
            <span className="tip-spark" aria-hidden="true">✳</span>
            <strong>Your workspace</strong>
            <p>Upload your documents, then ask questions to explore what’s inside.</p>
          </div>
          <div className="workspace-profile">
            <span className="profile-avatar" aria-hidden="true">
              {user.email.slice(0, 1).toUpperCase()}
            </span>
            <span className="profile-email" title={user.email}>
              {user.email}
            </span>
            <button
              className="sign-out-button"
              onClick={onSignOut}
              aria-label="Sign out"
              title="Sign out"
              type="button"
            >
              ↗
            </button>
          </div>
        </div>
      </aside>

      <section className="workspace-main">
        <header className="workspace-topbar">
          <div>
            <span className="workspace-breadcrumb">WORKSPACE <span>/</span></span>
            <h1>
              {currentPage === "documents"
                ? "Document upload"
                : currentPage === "evaluations"
                  ? "Evaluations"
                  : "Chat"}
            </h1>
          </div>
          <div className="topbar-status">
            <span className="status-dot" />
            Assistant ready
          </div>
        </header>
        <div className="workspace-content">
          <Routes>
            <Route path="chat" element={<ChatPanel />} />
            <Route
              path="documents"
              element={<UploadPanel />}
            />
            <Route
              path="evaluations"
              element={
                user.is_admin ? (
                  <EvaluationPanel />
                ) : (
                  <Navigate to="/home/chat" replace />
                )
              }
            />
            <Route path="*" element={<Navigate to="/home/chat" replace />} />
          </Routes>
        </div>
      </section>
    </main>
  );
}

interface ChatMessage {
  id: number;
  role: "user" | "assistant";
  content: string;
  sources?: ChatResponse["sources"];
}

function ChatPanel() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [conversationId, setConversationId] = useState("");
  const [draft, setDraft] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const nextId = useRef(0);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const message = draft.trim();
    if (!message || loading) return;

    setDraft("");
    setError("");
    setMessages((current) => [
      ...current,
      { id: nextId.current++, role: "user", content: message },
    ]);
    setLoading(true);
    try {
      const result = await sendChatMessage(
        message,
        conversationId || undefined,
      );
      setConversationId(result.conversation_id);
      setMessages((current) => [
        ...current,
        {
          id: nextId.current++,
          role: "assistant",
          content: result.response,
          sources: result.sources,
        },
      ]);
    } catch (cause) {
      setError(
        cause instanceof ApiError
          ? cause.message
          : "Something went wrong. Please try again.",
      );
    } finally {
      setLoading(false);
    }
  }

  function startNewConversation() {
    setMessages([]);
    setConversationId("");
    setDraft("");
    setError("");
  }

  return (
    <div className="chat-layout">
      <section className="chat-card">
        <div className="chat-card-header">
          <div className="assistant-heading">
            <span className="assistant-avatar" aria-hidden="true">e</span>
            <div>
              <h2>ETIC Assistant</h2>
              <p><span className="status-dot" /> Ready to help</p>
            </div>
          </div>
          <button className="new-chat-button" onClick={startNewConversation} type="button">
            <span aria-hidden="true">＋</span> New chat
          </button>
        </div>

        <div className="chat-thread" aria-live="polite">
          {messages.length === 0 ? (
            <div className="chat-welcome">
              <span className="welcome-spark" aria-hidden="true">✳</span>
              <h2>What would you like to know?</h2>
              <p>Ask a question about your work, or start by uploading a document.</p>
              <div className="suggestion-list">
                <button onClick={() => setDraft("Summarize my uploaded documents")} type="button">
                  Summarize my uploaded documents <span aria-hidden="true">↗</span>
                </button>
                <button onClick={() => setDraft("What are the key points I should know?")} type="button">
                  What are the key points I should know? <span aria-hidden="true">↗</span>
                </button>
              </div>
            </div>
          ) : (
            <div className="message-list">
              {messages.map((message) => (
                <article className={`chat-message ${message.role}`} key={message.id}>
                  {message.role === "assistant" && (
                    <span className="message-avatar" aria-hidden="true">e</span>
                  )}
                  <div className="message-content">
                    <span className="message-author">
                      {message.role === "user" ? "You" : "ETIC Assistant"}
                    </span>
                    <p>{message.content}</p>
                    {message.sources && message.sources.length > 0 && (
                      <div className="source-list">
                        <span className="source-heading">Sources</span>
                        {message.sources.map((source, index) => (
                          <div className="source-item" key={`${source.source}-${index}`}>
                            <span className="source-mark" aria-hidden="true">▤</span>
                            <span>
                              <strong>{source.document_title}</strong>
                              {source.section && <small>{source.section}</small>}
                            </span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </article>
              ))}
              {loading && (
                <div className="chat-loading" role="status">
                  <span className="message-avatar" aria-hidden="true">e</span>
                  <span>Thinking through your question…</span>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="composer-wrap">
          {error && <div className="notice error-notice workspace-notice" role="alert">{error}</div>}
          <form className="chat-composer" onSubmit={handleSubmit}>
            <textarea
              aria-label="Message"
              maxLength={20_000}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  event.currentTarget.form?.requestSubmit();
                }
              }}
              placeholder="Ask anything about your documents…"
              rows={2}
              value={draft}
            />
            <div className="composer-footer">
              <span>ETIC can make mistakes. Check important details.</span>
              <button
                aria-label="Send message"
                className="send-button"
                disabled={!draft.trim() || loading}
                type="submit"
              >
                {loading ? "…" : "↑"}
              </button>
            </div>
          </form>
          <p className="composer-hint">Press <kbd>Enter</kbd> to send · <kbd>Shift + Enter</kbd> for a new line</p>
        </div>
      </section>
      <aside className="chat-side-note">
        <span className="side-note-icon" aria-hidden="true">✳</span>
        <div>
          <strong>Answers grounded in your documents</strong>
          <p>Upload files to your workspace so the assistant can find relevant information.</p>
        </div>
      </aside>
    </div>
  );
}

function UploadPanel() {
  const fileInput = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [visibility, setVisibility] = useState<"private" | "public">("private");
  const [upload, setUpload] = useState<UploadResponse | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  function chooseFile(selectedFile: File | null) {
    setError("");
    setUpload(null);
    if (!selectedFile) {
      setFile(null);
      return;
    }

    const extension = selectedFile.name.slice(selectedFile.name.lastIndexOf(".")).toLowerCase();
    if (!SUPPORTED_EXTENSIONS.includes(extension)) {
      setFile(null);
      setError("Unsupported file type. Upload a PDF, Markdown, text, or CSV file.");
      return;
    }
    if (selectedFile.size > MAX_UPLOAD_BYTES) {
      setFile(null);
      setError("The document must be 10 MiB or smaller.");
      return;
    }
    setFile(selectedFile);
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file || loading) return;
    setError("");
    setUpload(null);
    setLoading(true);
    try {
      setUpload(await uploadDocument(file, visibility));
      setFile(null);
      if (fileInput.current) fileInput.current.value = "";
    } catch (cause) {
      setError(
        cause instanceof ApiError
          ? cause.message
          : "Something went wrong. Please try again.",
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="upload-layout">
      <section className="upload-card">
        <div className="upload-heading">
          <span className="upload-icon" aria-hidden="true">↥</span>
          <div>
            <h2>Add a document</h2>
            <p>Give your assistant useful context to answer your questions.</p>
          </div>
        </div>

        <form onSubmit={handleSubmit}>
          <label
            className={`drop-zone${file ? " has-file" : ""}`}
            htmlFor="document-file"
            onDragOver={(event) => event.preventDefault()}
            onDrop={(event) => {
              event.preventDefault();
              chooseFile(event.dataTransfer.files[0] ?? null);
            }}
          >
            <input
              accept=".pdf,.md,.markdown,.txt,.csv"
              id="document-file"
              onChange={(event) => chooseFile(event.target.files?.[0] ?? null)}
              ref={fileInput}
              type="file"
            />
            <span className="drop-icon" aria-hidden="true">{file ? "▤" : "↥"}</span>
            {file ? (
              <>
                <strong>{file.name}</strong>
                <span>{formatFileSize(file.size)} · Click to choose a different file</span>
              </>
            ) : (
              <>
                <strong>Choose a file to upload</strong>
                <span>or drag and drop it here</span>
                <small>PDF, Markdown, TXT, or CSV · Up to 10 MiB</small>
              </>
            )}
          </label>

          <label className="visibility-label" htmlFor="visibility">Who can use this document?</label>
          <select
            id="visibility"
            onChange={(event) => setVisibility(event.target.value as "private" | "public")}
            value={visibility}
          >
            <option value="private">Private — only me</option>
            <option value="public">Public — available as fallback to other users</option>
          </select>
          <p className="visibility-hint">
            Private documents are only searchable by your account.
          </p>
          {error && <div className="notice error-notice workspace-notice" role="alert">{error}</div>}
          {upload && (
            <div className="notice success-notice upload-success" role="status">
              <strong>{upload.document_title}</strong> uploaded and indexed successfully.
              <span>{upload.chunk_count} searchable {upload.chunk_count === 1 ? "chunk" : "chunks"} added to your workspace.</span>
            </div>
          )}
          <button className="upload-submit" disabled={!file || loading} type="submit">
            {loading ? "Uploading and indexing…" : "Upload document"}
            {!loading && <span aria-hidden="true">↑</span>}
          </button>
        </form>
      </section>

      <aside className="upload-side-note">
        <span className="side-note-icon" aria-hidden="true">✳</span>
        <div>
          <strong>What happens to my file?</strong>
          <p>Your document is split into searchable sections and indexed securely for your account.</p>
        </div>
      </aside>
    </div>
  );
}

function EvaluationPanel() {
  const [result, setResult] = useState<EvaluationResponse | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleRunEvaluation() {
    setLoading(true);
    setError("");
    setResult(null);
    try {
      setResult(await runEvaluation());
    } catch (cause) {
      setError(
        cause instanceof ApiError
          ? cause.message
          : "Something went wrong. Please try again.",
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="evaluation-layout">
      <section className="evaluation-card">
        <div className="evaluation-heading">
          <span className="upload-icon" aria-hidden="true">◷</span>
          <div>
            <h2>Run retrieval evaluation</h2>
            <p>
              Evaluate retrieval relevance and answer correctness with the
              configured LangSmith dataset.
            </p>
          </div>
        </div>
        <div className="evaluation-callout">
          <strong>Administrator evaluation</strong>
          <p>
            This may take a few minutes and uses your account’s documents. The
            results are saved to LangSmith.
          </p>
        </div>
        {error && <div className="notice error-notice workspace-notice" role="alert">{error}</div>}
        <button
          className="upload-submit run-evaluation-button"
          disabled={loading}
          onClick={handleRunEvaluation}
          type="button"
        >
          {loading ? (
            <>
              <span className="evaluation-spinner" aria-hidden="true" />
              Evaluation running…
            </>
          ) : (
            "Run evaluation"
          )}
          {!loading && <span aria-hidden="true">→</span>}
        </button>
        {result && (
          <section className="evaluation-result" aria-live="polite">
            <div className="evaluation-result-heading">
              <div>
                <span className="evaluation-complete"><span className="status-dot" /> Evaluation complete</span>
                <h3>{result.experiment_name}</h3>
              </div>
              {result.experiment_url && (
                <a
                  className="experiment-link"
                  href={result.experiment_url}
                  rel="noreferrer"
                  target="_blank"
                >
                  Open in LangSmith <span aria-hidden="true">↗</span>
                </a>
              )}
            </div>
            <pre className="evaluation-json">{JSON.stringify(result, null, 2)}</pre>
          </section>
        )}
      </section>
      <aside className="upload-side-note">
        <span className="side-note-icon" aria-hidden="true">✳</span>
        <div>
          <strong>What is being evaluated?</strong>
          <p>Each dataset example is scored for retrieval relevance and answer correctness.</p>
        </div>
      </aside>
    </div>
  );
}

function formatFileSize(size: number): string {
  if (size < 1024 * 1024) return `${Math.max(1, Math.round(size / 1024))} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}
