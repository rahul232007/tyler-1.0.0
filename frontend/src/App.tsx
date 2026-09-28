import { useState, useRef } from "react";

const API_BASE = `${(import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/+$/, "")}/api/v1`;

function App() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");

  const [loggedIn, setLoggedIn] = useState(
    !!localStorage.getItem("jarvis_token")
  );

  const [isRegister, setIsRegister] = useState(false);
  const [message, setMessage] = useState("");
  const [chatMessage, setChatMessage] = useState("");
  const [response, setResponse] = useState("");
  const [provider, setProvider] = useState("");
  const [loading, setLoading] = useState(false);

  // Voice streaming states
  const [isRecording, setIsRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const [voiceTranscription, setVoiceTranscription] = useState("");
  const [voiceResponse, setVoiceResponse] = useState("");
  const [voiceStatus, setVoiceStatus] = useState("");
  const [voiceTelemetry, setVoiceTelemetry] = useState<string | null>(null);
  const audioQueueRef = useRef<string[]>([]);
  const isPlayingRef = useRef(false);

  const login = async () => {
    setMessage("");

    try {
      const res = await fetch(`${API_BASE}/auth/login`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          email,
          password,
        }),
      });

      const data = await res.json();

      if (!res.ok) {
        setMessage(`❌ ${data.detail || "Login failed"}`);
        return;
      }

      localStorage.setItem("jarvis_token", data.access_token);
      setLoggedIn(true);
      setMessage("✅ Login successful");
    } catch (error) {
      setMessage(`❌ Backend connection failed: ${String(error)}`);
    }
  };

  const register = async () => {
    setMessage("");

    try {
      const res = await fetch(`${API_BASE}/auth/register`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          email,
          password,
          display_name: displayName,
        }),
      });

      const data = await res.json();

      if (!res.ok) {
        setMessage(`❌ ${data.detail || "Registration failed"}`);
        return;
      }

      setMessage("✅ Registration successful. Now login.");
      setIsRegister(false);
    } catch (error) {
      setMessage(`❌ Backend connection failed: ${String(error)}`);
    }
  };

  const logout = () => {
    localStorage.removeItem("jarvis_token");
    setLoggedIn(false);
    setResponse("");
    setProvider("");
  };

  const sendMessage = async () => {
    if (!chatMessage.trim()) return;

    setLoading(true);
    setResponse("");
    setProvider("");

    try {
      const token = localStorage.getItem("jarvis_token");

      const res = await fetch(`${API_BASE}/chat/`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          message: chatMessage.trim(),
          stream: true,
          conversation_id: null,
          document_id: null,
          use_tools: true,
        }),
      });

      if (!res.ok) {
        const errorData = await res.json();
        setResponse(`❌ ${errorData.detail || "Chat failed"}`);
        return;
      }

      // Read provider header immediately
      const providerHeader = res.headers.get("x-provider-used") || "ollama";
      setProvider(providerHeader);

      const reader = res.body?.getReader();
      if (!reader) {
        setResponse("❌ ReadableStream not supported in this environment.");
        return;
      }

      const decoder = new TextDecoder();
      let done = false;
      let isFirstChunk = true;

      while (!done) {
        const { value, done: doneReading } = await reader.read();
        done = doneReading;
        if (value) {
          const chunk = decoder.decode(value, { stream: true });
          if (isFirstChunk) {
            setLoading(false); // Immediate TTFT: Clear loading indicator as soon as first token arrives (~2-4s)
            isFirstChunk = false;
          }
          setResponse((prev) => prev + chunk);
        }
      }
    } catch (error) {
      setResponse(`❌ Connection error: ${String(error)}`);
    } finally {
      setLoading(false);
    }
  };

  const playNextAudio = () => {
    if (audioQueueRef.current.length === 0) {
      isPlayingRef.current = false;
      return;
    }
    isPlayingRef.current = true;
    const nextB64 = audioQueueRef.current.shift()!;
    const audio = new Audio(`data:audio/mp3;base64,${nextB64}`);
    audio.onended = () => {
      playNextAudio();
    };
    audio.onerror = () => {
      playNextAudio();
    };
    audio.play().catch((err) => {
      console.warn("Audio play error:", err);
      playNextAudio();
    });
  };

  const queueAudioChunk = (b64: string) => {
    audioQueueRef.current.push(b64);
    if (!isPlayingRef.current) {
      playNextAudio();
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      const chunks: Blob[] = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
      };

      recorder.onstop = async () => {
        const audioBlob = new Blob(chunks, { type: "audio/wav" });
        await sendVoiceStream(audioBlob);
        stream.getTracks().forEach((t) => t.stop());
      };

      recorder.start();
      setMediaRecorder(recorder);
      setIsRecording(true);
      setVoiceTranscription("");
      setVoiceResponse("");
      setVoiceTelemetry(null);
      setVoiceStatus("🎙️ Listening...");
    } catch (err) {
      setVoiceStatus(`❌ Microphone error: ${String(err)}`);
    }
  };

  const stopRecording = () => {
    if (mediaRecorder && isRecording) {
      mediaRecorder.stop();
      setIsRecording(false);
      setVoiceStatus("⚙️ Processing speech...");
    }
  };

  const sendVoiceStream = async (blob: Blob) => {
    const t0 = performance.now();
    let firstAudioTime: number | null = null;
    audioQueueRef.current = [];
    isPlayingRef.current = false;

    try {
      const token = localStorage.getItem("jarvis_token");
      const formData = new FormData();
      formData.append("audio_file", blob, "voice_input.wav");
      formData.append("language", "auto");

      const res = await fetch(`${API_BASE}/voice/stream`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
        },
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        setVoiceStatus(`❌ Voice request failed: ${err.detail || res.statusText}`);
        return;
      }

      const reader = res.body?.getReader();
      if (!reader) {
        setVoiceStatus("❌ ReadableStream not available");
        return;
      }

      const decoder = new TextDecoder();
      let buffer = "";
      setVoiceStatus("⚡ Streaming response...");

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const data = JSON.parse(line.slice(6));
            if (data.type === "transcription") {
              setVoiceTranscription(data.text);
            } else if (data.type === "token") {
              setVoiceResponse((prev) => prev + data.token);
            } else if (data.type === "audio") {
              if (data.audio_b64) {
                if (firstAudioTime === null) {
                  firstAudioTime = Math.round(performance.now() - t0);
                }
                queueAudioChunk(data.audio_b64);
              }
            } else if (data.type === "done") {
              const totalTime = Math.round(performance.now() - t0);
              const ttft = data.ttft_ms ? `${data.ttft_ms}ms` : "N/A";
              const firstAudioStr = firstAudioTime ? `${firstAudioTime}ms` : "N/A";
              setVoiceTelemetry(
                `⚡ First Spoken Audio: ${firstAudioStr} | STT: ${data.stt_ms || "N/A"}ms | LLM TTFT: ${ttft} | Total: ${totalTime}ms`
              );
              setVoiceStatus("✅ Complete");
            } else if (data.type === "error") {
              setVoiceStatus(`❌ Error: ${data.detail}`);
            }
          } catch (e) {
            console.error("SSE parse error", e);
          }
        }
      }
    } catch (err) {
      setVoiceStatus(`❌ Streaming error: ${String(err)}`);
    }
  };

  if (!loggedIn) {
    return (
      <div
        style={{
          minHeight: "100vh",
          background: "#111",
          color: "white",
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          fontFamily: "Arial",
        }}
      >
        <div
          style={{
            width: "380px",
            background: "#222",
            padding: "30px",
            borderRadius: "15px",
          }}
        >
          <h1>🤖 JARVIS</h1>
          <h2>{isRegister ? "Create Account" : "Login"}</h2>

          {isRegister && (
            <input
              placeholder="Display name"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              style={inputStyle}
            />
          )}

          <input
            placeholder="Email"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            style={inputStyle}
          />

          <input
            placeholder="Password"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            style={inputStyle}
          />

          <button
            onClick={isRegister ? register : login}
            style={buttonStyle}
          >
            {isRegister ? "Register" : "Login"}
          </button>

          <button
            onClick={() => setIsRegister(!isRegister)}
            style={{
              ...buttonStyle,
              background: "#444",
              marginTop: "10px",
            }}
          >
            {isRegister
              ? "Already have an account? Login"
              : "Create new account"}
          </button>

          {message && (
            <p style={{ marginTop: "20px" }}>
              {message}
            </p>
          )}
        </div>
      </div>
    );
  }

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#111",
        color: "white",
        padding: "40px",
        fontFamily: "Arial",
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <h1>🤖 JARVIS Model 1</h1>

        <button onClick={logout} style={buttonStyle}>
          Logout
        </button>
      </div>

      <hr />

      <h2>💬 Text Chat Test (Real-Time Token Streaming)</h2>

      <textarea
        value={chatMessage}
        onChange={(e) => setChatMessage(e.target.value)}
        placeholder="Type a message to JARVIS..."
        rows={5}
        style={{
          width: "100%",
          maxWidth: "700px",
          padding: "15px",
          fontSize: "16px",
          background: "#222",
          color: "white",
        }}
      />

      <br />
      <br />

      <button
        onClick={sendMessage}
        disabled={loading}
        style={buttonStyle}
      >
        {loading ? "🤔 JARVIS Thinking..." : "Send"}
      </button>

      <div style={{ marginTop: "30px", maxWidth: "700px" }}>
        <h2>🤖 JARVIS Response</h2>

        <div
          style={{
            background: "#222",
            padding: "20px",
            borderRadius: "10px",
            whiteSpace: "pre-wrap",
            minHeight: "100px",
          }}
        >
          {response || "Waiting for message..."}
        </div>

        {provider && (
          <p>
            <strong>Provider:</strong> {provider}
          </p>
        )}
      </div>

      <hr style={{ margin: "40px 0", borderColor: "#333" }} />

      <h2>🎙️ Voice Chat (Real-Time Sentence Streaming & Spoken Audio)</h2>
      <p style={{ color: "#aaa" }}>
        Low-latency streaming voice: First spoken sentence in &le; 4s, complete response in 5&ndash;8s.
      </p>

      <div style={{ display: "flex", gap: "15px", alignItems: "center", margin: "15px 0" }}>
        {!isRecording ? (
          <button
            onClick={startRecording}
            style={{ ...buttonStyle, background: "#2563eb", marginTop: 0 }}
          >
            🎙️ Start Speaking
          </button>
        ) : (
          <button
            onClick={stopRecording}
            style={{ ...buttonStyle, background: "#dc2626", marginTop: 0 }}
          >
            ⏹️ Stop &amp; Send to JARVIS
          </button>
        )}

        {voiceStatus && <span style={{ fontWeight: "bold" }}>{voiceStatus}</span>}
      </div>

      {voiceTranscription && (
        <div style={{ marginTop: "15px", maxWidth: "700px" }}>
          <strong>You said:</strong>
          <div style={{ background: "#222", padding: "12px", borderRadius: "8px", marginTop: "5px" }}>
            {voiceTranscription}
          </div>
        </div>
      )}

      {voiceResponse && (
        <div style={{ marginTop: "15px", maxWidth: "700px" }}>
          <strong>JARVIS Spoken Response (Live Tokens):</strong>
          <div style={{ background: "#222", padding: "15px", borderRadius: "8px", marginTop: "5px", whiteSpace: "pre-wrap" }}>
            {voiceResponse}
          </div>
        </div>
      )}

      {voiceTelemetry && (
        <div style={{ marginTop: "15px", maxWidth: "700px", background: "#1e293b", padding: "12px", borderRadius: "8px", border: "1px solid #3b82f6" }}>
          <strong style={{ color: "#60a5fa" }}>📊 Voice Latency Telemetry:</strong>
          <p style={{ margin: "5px 0 0 0", fontFamily: "monospace", fontSize: "14px" }}>
            {voiceTelemetry}
          </p>
        </div>
      )}
    </div>
  );
}

const inputStyle = {
  width: "100%",
  boxSizing: "border-box" as const,
  padding: "12px",
  marginTop: "10px",
  background: "#333",
  color: "white",
  border: "1px solid #555",
  borderRadius: "6px",
};

const buttonStyle = {
  padding: "12px 20px",
  marginTop: "15px",
  background: "#555",
  color: "white",
  border: "none",
  borderRadius: "6px",
  cursor: "pointer",
};

export default App;