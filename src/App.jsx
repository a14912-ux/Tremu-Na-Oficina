import { useState, useEffect, useCallback, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import * as handpose from "@tensorflow-models/handpose";
import "./App.css";

const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

const WORDS = [
  "GATO", "BOLA", "MESA", "CASA", "PATO",
  "RATO", "FOCA", "LUVA", "VASO",
  "TETO", "RIMA", "DEDO", "FATO", "MALA",
  "BOTA", "VELA", "SOPA", "FACA", "LATA",
  "NAVE", "PORE",
];

// Padrão de dedos esperado para cada letra (5 booleanos:
// polegar, indicador, médio, anelar, mindinho -> true = esticado)
function fingerPattern(letter) {
  const idx = ALPHABET.indexOf(letter.toUpperCase());
  const bits = [];
  for (let i = 0; i < 5; i++) {
    bits.push(((idx + i * 3 + (idx % 7)) % 5) < 2);
  }
  if (bits.every((b) => b === bits[0])) bits[idx % 5] = !bits[idx % 5];
  return bits;
}

// Desenha o sinal "alvo" para o utilizador imitar
function HandSignSVG({ letter, size = 140 }) {
  const [thumb, index, middle, ring, pinky] = fingerPattern(letter);

  const finger = (x, up, label) => {
    const len = up ? 60 : 28;
    const y2 = 70 - len;
    return (
      <rect
        key={label}
        x={x}
        y={y2}
        width="14"
        height={70 - y2}
        rx="7"
        fill={up ? "#f4c39e" : "#e0a17a"}
        stroke="#7c4a2d"
        strokeWidth="2"
      />
    );
  };

  return (
    <svg viewBox="0 0 140 160" width={size} height={size * (160 / 140)} role="img"
      aria-label={`Sinal manual da letra ${letter}`}>
      <rect x="20" y="60" width="100" height="80" rx="20" fill="#f4c39e" stroke="#7c4a2d" strokeWidth="3" />
      <g transform={`translate(0,0) rotate(${thumb ? -25 : -5} 25 95)`}>
        <rect x="2" y="85" width={thumb ? 50 : 26} height="22" rx="11"
          fill={thumb ? "#f4c39e" : "#e0a17a"} stroke="#7c4a2d" strokeWidth="2" />
      </g>
      <g transform="translate(28,0)">
        {finger(2, index, "i")}
        {finger(22, middle, "m")}
        {finger(42, ring, "r")}
        {finger(62, pinky, "p")}
      </g>
      <rect x="45" y="138" width="50" height="20" rx="6" fill="#e0a17a" stroke="#7c4a2d" strokeWidth="2" />
    </svg>
  );
}

// A partir dos 21 landmarks da mão (modelo handpose), calcula
// quais dedos estão esticados, comparando a distância da ponta
// ao pulso com a distância da junta intermédia ao pulso.
function landmarksToPattern(landmarks) {
  const wrist = landmarks[0];
  const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

  const tips = [4, 8, 12, 16, 20];
  const pips = [2, 6, 10, 14, 18];

  return tips.map((tip, i) => {
    const dTip = dist(landmarks[tip], wrist);
    const dPip = dist(landmarks[pips[i]], wrist);
    return dTip > dPip * 1.15; // esticado se a ponta está bem mais longe
  });
}

function patternsMatch(a, b) {
  let matches = 0;
  for (let i = 0; i < 5; i++) if (a[i] === b[i]) matches++;
  return matches >= 4; // tolerância de 1 dedo
}

// ---------- Webcam + reconhecimento de mão ----------
function WebcamRecognizer({ targetLetter, onMatch, locked }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const modelRef = useRef(null);
  const rafRef = useRef(null);
  const matchFramesRef = useRef(0);

  const [status, setStatus] = useState("off"); // off | loading-cam | loading-model | on | error
  const [errorMsg, setErrorMsg] = useState("");
  const [detected, setDetected] = useState(false);

  const startCamera = async () => {
    setStatus("loading-cam");
    setErrorMsg("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: 320, height: 240 },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setStatus("loading-model");
      await tf.setBackend("webgl");
      await tf.ready();
      if (!modelRef.current) {
        modelRef.current = await handpose.load();
      }

      setStatus("on");
      detectLoop();
    } catch (err) {
      console.error(err);
      setStatus("error");
      setErrorMsg("Não foi possível aceder à câmara ou carregar o modelo.");
    }
  };

  const stopCamera = () => {
    cancelAnimationFrame(rafRef.current);
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    setStatus("off");
    setDetected(false);
    matchFramesRef.current = 0;
  };

  useEffect(() => () => stopCamera(), []);

  const detectLoop = useCallback(async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !modelRef.current) return;

    const ctx = canvas.getContext("2d");
    canvas.width = video.videoWidth || 320;
    canvas.height = video.videoHeight || 240;

    const loop = async () => {
      if (!streamRef.current) return; // câmara desligada

      const predictions = await modelRef.current.estimateHands(video);

      ctx.save();
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1); // espelho
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      let isMatch = false;

      if (predictions.length > 0) {
        const landmarks = predictions[0].landmarks;

        // desenhar pontos da mão
        ctx.fillStyle = "#ffb703";
        landmarks.forEach(([x, y]) => {
          ctx.beginPath();
          ctx.arc(x, y, 4, 0, 2 * Math.PI);
          ctx.fill();
        });

        const pattern = landmarksToPattern(landmarks);
        const target = fingerPattern(targetLetter);
        isMatch = patternsMatch(pattern, target);
      }

      ctx.restore();

      setDetected(isMatch);

      if (isMatch && !locked) {
        matchFramesRef.current += 1;
        // precisa manter o sinal ~0.6s (≈18 frames a 30fps) para confirmar
        if (matchFramesRef.current > 18) {
          matchFramesRef.current = 0;
          onMatch();
        }
      } else {
        matchFramesRef.current = 0;
      }

      rafRef.current = requestAnimationFrame(loop);
    };

    loop();
  }, [targetLetter, locked, onMatch]);

  // reinicia a deteção sempre que muda a letra alvo
  useEffect(() => {
    matchFramesRef.current = 0;
  }, [targetLetter]);

  return (
    <div className="webcam-panel">
      <p className="sign-instruction">📷 Faz o sinal com a tua mão</p>
      <div className="webcam-frame">
        <video ref={videoRef} className="webcam-video-hidden" playsInline muted />
        <canvas ref={canvasRef} className="webcam-canvas" />
        {status !== "on" && (
          <div className="webcam-placeholder">
            {status === "loading-cam" && <span>A ligar câmara…</span>}
            {status === "loading-model" && <span>A carregar modelo de mão…</span>}
            {status === "off" && <span>Câmara desligada</span>}
            {status === "error" && <span className="webcam-error">{errorMsg}</span>}
          </div>
        )}
      </div>

      <div className={`match-indicator ${detected ? "match-ok" : ""}`}>
        {status === "on"
          ? detected
            ? "✅ Sinal correspondente! Mantém a mão…"
            : "🤚 A aguardar sinal correto…"
          : ""}
      </div>

      <div className="webcam-controls">
        {status !== "on" ? (
          <button className="cam-btn" onClick={startCamera}>🎥 Ligar câmara</button>
        ) : (
          <button className="cam-btn cam-btn-stop" onClick={stopCamera}>⏹️ Desligar câmara</button>
        )}
      </div>
    </div>
  );
}
// ---------- Fim Webcam ----------

function pickWord(prev) {
  let w;
  do {
    w = WORDS[Math.floor(Math.random() * WORDS.length)];
  } while (w === prev);
  return w;
}

export default function App() {
  const [word, setWord] = useState(() => pickWord(null));
  const [pos, setPos] = useState(0);
  const [revealed, setRevealed] = useState([]);
  const [score, setScore] = useState(0);
  const [wordsCompleted, setWordsCompleted] = useState(0);
  const [feedback, setFeedback] = useState(null);
  const [locked, setLocked] = useState(false); // evita múltiplas confirmações seguidas

  const newWord = useCallback(() => {
    setWord((prev) => pickWord(prev));
    setPos(0);
    setRevealed([]);
    setFeedback(null);
    setLocked(false);
  }, []);

  const handleMatch = useCallback(() => {
    setLocked(true);
    const correct = word[pos];
    setRevealed((r) => [...r, correct]);
    setScore((s) => s + 10);
    setFeedback({ ok: true, msg: `✅ Sinal da letra "${correct}" reconhecido!` });

    if (pos + 1 === word.length) {
      setScore((s) => s + 20);
      setWordsCompleted((c) => c + 1);
      setTimeout(() => newWord(), 1500);
    } else {
      setTimeout(() => {
        setPos((p) => p + 1);
        setFeedback(null);
        setLocked(false);
      }, 1000);
    }
  }, [word, pos, newWord]);

  return (
    <div className="app">
      <header className="header">
        <h1>👋 TREMU NA OFICINA</h1>
        <p className="subtitle">
          Liga a câmara, mostra a tua mão e reproduz o <strong>sinal manual</strong> de
          cada letra para formares a palavra.
        </p>
      </header>

      <section className="stats">
        <div className="stat">
          <span className="stat-value">{score}</span>
          <span className="stat-label">Pontos</span>
        </div>
        <div className="stat">
          <span className="stat-value">{wordsCompleted}</span>
          <span className="stat-label">Palavras</span>
        </div>
      </section>

      <section className="word-display">
        {word.split("").map((ch, i) => (
          <div key={i} className={`letter-box ${i === pos ? "active" : ""} ${i < revealed.length ? "revealed" : ""}`}>
            {i < revealed.length ? revealed[i] : ""}
          </div>
        ))}
      </section>

      <div className="main-grid">
        <section className="sign-area">
          <p className="sign-instruction">Sinal pedido:</p>
          <div className="sign-card">
            <HandSignSVG letter={word[pos]} size={150} />
          </div>
          <p className="target-letter">Letra alvo: <strong>{word[pos]}</strong></p>
        </section>

        <WebcamRecognizer targetLetter={word[pos]} onMatch={handleMatch} locked={locked} />
      </div>

      {feedback && <p className={`feedback ${feedback.ok ? "ok" : "fail"}`}>{feedback.msg}</p>}

      <footer className="footer">
        <button className="reset-btn" onClick={newWord}>🔄 Nova palavra</button>

      </footer>
    </div>
  );
}