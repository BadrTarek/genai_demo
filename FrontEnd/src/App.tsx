import { useState, type FormEvent, type ReactNode } from "react";
import {
  Link,
  Navigate,
  Route,
  Routes,
  useLocation,
  useNavigate,
} from "react-router-dom";
import {
  ApiError,
  clearSavedSession,
  getSavedSession,
  register,
  saveSession,
  signIn,
  type SignInResponse,
} from "./api";
import HomePage from "./HomePage";

const passwordRules = [
  { label: "At least 12 characters", test: (value: string) => value.length >= 12 },
  { label: "One uppercase letter", test: (value: string) => /[A-Z]/.test(value) },
  { label: "One lowercase letter", test: (value: string) => /[a-z]/.test(value) },
  { label: "One number", test: (value: string) => /\d/.test(value) },
  {
    label: "One special character",
    test: (value: string) => /[^A-Za-z0-9]/.test(value),
  },
];

function App() {
  const location = useLocation();
  const navigate = useNavigate();
  const [registeredEmail, setRegisteredEmail] = useState("");
  const [session, setSession] = useState<SignInResponse | null>(getSavedSession);

  function handleSignedIn(result: SignInResponse) {
    saveSession(result);
    setSession(result);
    navigate("/home/chat", { replace: true });
  }

  function handleSignOut() {
    clearSavedSession();
    setSession(null);
    navigate("/login");
  }

  return (
    <Routes>
      <Route
        path="/home/*"
        element={
          session ? (
            <HomePage user={session.user} onSignOut={handleSignOut} />
          ) : (
            <Navigate to="/login" replace />
          )
        }
      />
      <Route
        path="/login"
        element={
          session ? (
            <Navigate to="/home/chat" replace />
          ) : (
            <AuthLayout pathname={location.pathname}>
              <LoginForm
                initialEmail={registeredEmail}
                onSignedIn={handleSignedIn}
              />
            </AuthLayout>
          )
        }
      />
      <Route
        path="/register"
        element={
          session ? (
            <Navigate to="/home/chat" replace />
          ) : (
            <AuthLayout pathname={location.pathname}>
              <RegisterForm
                onRegistered={(email) => {
                  setRegisteredEmail(email);
                  navigate("/login", { state: { registered: true } });
                }}
              />
            </AuthLayout>
          )
        }
      />
      <Route
        path="*"
        element={<Navigate to={session ? "/home/chat" : "/login"} replace />}
      />
    </Routes>
  );
}

function AuthLayout({
  children,
  pathname,
}: {
  children: ReactNode;
  pathname: string;
}) {
  return (
    <main className="page-shell">
      <section className="auth-card" aria-label="ETIC GenAI account">
        <aside className="brand-panel">
          <Link className="brand" to="/login" aria-label="ETIC GenAI home">
            <span className="brand-mark" aria-hidden="true">e</span>
            <span>etic<span className="brand-light">/genai</span></span>
          </Link>
          <div className="brand-copy">
            <span className="eyebrow">YOUR KNOWLEDGE, IN MOTION</span>
            <h1>Ideas move<br />things forward.</h1>
            <p>A thoughtful space to explore your documents and find the answers that move your work ahead.</p>
          </div>
          <div className="orbit-art" aria-hidden="true">
            <div className="orbit orbit-one" />
            <div className="orbit orbit-two" />
            <div className="orbit-core"><span>e</span></div>
            <i className="orbit-dot dot-one" />
            <i className="orbit-dot dot-two" />
            <i className="orbit-dot dot-three" />
          </div>
          <p className="panel-footer">A little more clarity. A lot more possibility.</p>
        </aside>
        <section className="form-panel">
          <header className="form-header">
            <span className="mobile-brand">etic<span>/genai</span></span>
            <nav className="auth-nav" aria-label="Account">
              <Link className={pathname === "/login" ? "active" : ""} to="/login">
                Log in
              </Link>
              <Link className={pathname === "/register" ? "active" : ""} to="/register">
                Create account
              </Link>
            </nav>
          </header>
          <div className="form-content">{children}</div>
          <footer className="form-footer">
            <span>© 2026 ETIC GenAI</span>
            <span>Built for what’s next</span>
          </footer>
        </section>
      </section>
    </main>
  );
}

function LoginForm({
  initialEmail,
  onSignedIn,
}: {
  initialEmail: string;
  onSignedIn: (result: SignInResponse) => void;
}) {
  const location = useLocation();
  const [email, setEmail] = useState(initialEmail);
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const registered = Boolean(
    (location.state as { registered?: boolean } | null)?.registered,
  );

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setLoading(true);
    try {
      onSignedIn(await signIn(email.trim(), password));
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
    <>
      <span className="eyebrow form-eyebrow">WELCOME BACK</span>
      <h2>Good to see you.</h2>
      <p className="form-intro">Log in to pick up where your ideas left off.</p>
      {registered && (
        <div className="notice success-notice" role="status">
          Account created. You can log in now.
        </div>
      )}
      <form className="auth-form" onSubmit={handleSubmit}>
        <label htmlFor="login-email">Email address</label>
        <input
          autoComplete="email"
          id="login-email"
          name="email"
          onChange={(event) => setEmail(event.target.value)}
          placeholder="you@example.com"
          required
          type="email"
          value={email}
        />
        <div className="label-row">
          <label htmlFor="login-password">Password</label>
        </div>
        <input
          autoComplete="current-password"
          id="login-password"
          name="password"
          onChange={(event) => setPassword(event.target.value)}
          placeholder="Enter your password"
          required
          type="password"
          value={password}
        />
        {error && <div className="notice error-notice" role="alert">{error}</div>}
        <button className="primary-button" disabled={loading} type="submit">
          {loading ? "Logging in…" : "Log in"}
          <span aria-hidden="true">→</span>
        </button>
      </form>
      <p className="switch-prompt">
        New around here? <Link to="/register">Create an account</Link>
      </p>
    </>
  );
}

function RegisterForm({
  onRegistered,
}: {
  onRegistered: (email: string) => void;
}) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const passwordIsValid = passwordRules.every((rule) => rule.test(password));

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");

    if (!passwordIsValid) {
      setError("Please meet all password requirements.");
      return;
    }
    if (password !== confirmPassword) {
      setError("Your passwords do not match.");
      return;
    }

    setLoading(true);
    try {
      const user = await register(email.trim(), password);
      onRegistered(user.email);
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
    <>
      <span className="eyebrow form-eyebrow">START HERE</span>
      <h2>Make room for ideas.</h2>
      <p className="form-intro">Create an account and see where your next question takes you.</p>
      <form className="auth-form register-form" onSubmit={handleSubmit}>
        <label htmlFor="register-email">Email address</label>
        <input
          autoComplete="email"
          id="register-email"
          name="email"
          onChange={(event) => setEmail(event.target.value)}
          placeholder="you@example.com"
          required
          type="email"
          value={email}
        />
        <label htmlFor="register-password">Create password</label>
        <input
          autoComplete="new-password"
          id="register-password"
          name="password"
          onChange={(event) => setPassword(event.target.value)}
          placeholder="Choose a strong password"
          required
          type="password"
          value={password}
        />
        <ul className="password-rules" aria-label="Password requirements">
          {passwordRules.map((rule) => {
            const met = rule.test(password);
            return (
              <li className={met ? "rule-met" : ""} key={rule.label}>
                <span aria-hidden="true">{met ? "✓" : "○"}</span>
                {rule.label}
              </li>
            );
          })}
        </ul>
        <label htmlFor="confirm-password">Confirm password</label>
        <input
          autoComplete="new-password"
          id="confirm-password"
          name="confirmPassword"
          onChange={(event) => setConfirmPassword(event.target.value)}
          placeholder="Enter your password again"
          required
          type="password"
          value={confirmPassword}
        />
        {error && <div className="notice error-notice" role="alert">{error}</div>}
        <button className="primary-button" disabled={loading} type="submit">
          {loading ? "Creating account…" : "Create account"}
          <span aria-hidden="true">→</span>
        </button>
      </form>
      <p className="switch-prompt">
        Already have an account? <Link to="/login">Log in</Link>
      </p>
    </>
  );
}

export default App;
