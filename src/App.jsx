import { useState, useEffect, useCallback, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import * as handpose from "@tensorflow-models/handpose";
import "./App.css";

const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

const WORDS = [
  "GATO","BOLA","MESA","CASA","PATO",
  "RATO","FOCA","LUVA","VASO","TETO",
  "RIMA","DEDO","FATO","MALA","BOTA",
  "VELA","SOPA","FACA","LATA","NAVE","PORE",
];

function fingerPattern(letter) {
  const idx = ALPHABET.indexOf(letter.toUpperCase());
  return Array.from({ length: 5 }, (_, i) =>
    ((idx + i * 3 + (idx % 7)) % 5) < 2
  );
}

function HandSignSVG({ letter, size = 150 }) {
  const [thumb, index, middle, ring, pinky] = fingerPattern(letter);

  const finger = (x, up) => {
    const len = up ? 60 : 30;
    return (
      <rect
        x={x}
        y={70 - len}
        width="14"
        height={len}
        rx="7"
        className={up ? "finger up" : "finger down"}
      />
    );
  };

  return (
    <svg viewBox="0 0 140 160" width={size}>
      <rect className="hand-palm" x="20" y="60" width="100" height="80" rx="20" />

      <g transform={`rotate(${thumb ? -25 : -5} 25 95)`}>
        <rect className={thumb ? "thumb up" : "thumb down"} x="2" y="85" width="45" height="22" rx="11" />
      </g>

      <g transform="translate(28,0)">
        {finger(2, index)}
        {finger(22, middle)}
        {finger(42, ring)}
        {finger(62, pinky)}
      </g>
    </svg>
  );
}

function WebcamRecognizer({ targetLetter, onMatch, locked }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const modelRef = useRef(null);
  const rafRef = useRef(null);
  const streamRef = useRef(null);
  const matchRef = useRef(0);

  const [status, setStatus] = useState("off");
  const [detected, setDetected] = useState(false);

  const start = async () => {
    try {
      setStatus("loading");

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" },
      });

      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();

      await tf.setBackend("webgl");
      await tf.ready();

      modelRef.current = modelRef.current || await handpose.load();

      setStatus("on");
      loop();
    } catch {
      setStatus("error");
    }
  };

  const stop = () => {
    cancelAnimationFrame(rafRef.current);
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    setStatus("off");
  };

  const loop = useCallback(async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !modelRef.current) return;

    const ctx = canvas.getContext("2d");
    canvas.width = video.videoWidth || 320;
    canvas.height = video.videoHeight || 240;

    const run = async () => {
      const hands = await modelRef.current.estimateHands(video);

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(video, 0, 0);

      let match = false;

      if (hands.length) {
        const landmarks = hands[0].landmarks;

        const tips = [4,8,12,16,20];
        const wrist = landmarks[0];

        const pattern = tips.map(i =>
          Math.hypot(...landmarks[i].map((v,j)=>v-wrist[j])) > 60
        );

        const target = fingerPattern(targetLetter);
        match = pattern.filter((v,i)=>v===target[i]).length >= 4;
      }

      setDetected(match);

      if (match && !locked) {
        matchRef.current++;
        if (matchRef.current > 15) {
          matchRef.current = 0;
          onMatch();
        }
      } else matchRef.current = 0;

      rafRef.current = requestAnimationFrame(run);
    };

    run();
  }, [targetLetter, locked, onMatch]);

  return (
    <div className="card webcam">
      <p className="label">📷 Faz o sinal</p>

      <div className="video-box">
        <video ref={videoRef} className="hidden" />
        <canvas ref={canvasRef} />

        {status !== "on" && (
          <div className="overlay">
            {status === "loading" ? "A iniciar..." :
             status === "error" ? "Erro na câmara" :
             "Câmara desligada"}
          </div>
        )}
      </div>

      <div className={`status ${detected ? "ok" : ""}`}>
        {status === "on" && (detected ? "✔ correto" : "aguardar sinal")}
      </div>

      <button onClick={status === "on" ? stop : start}>
        {status === "on" ? "Desligar" : "Ligar câmara"}
      </button>
    </div>
  );
}

function pickWord(prev) {
  let w;
  do {
    w = WORDS[Math.floor(Math.random() * WORDS.length)];
  } while (w === prev);
  return w;
}

export default function App() {
  const [word, setWord] = useState(() => pickWord());
  const [pos, setPos] = useState(0);
  const [score, setScore] = useState(0);
  const [locked, setLocked] = useState(false);

  const next = () => {
    setWord(prev => pickWord(prev));
    setPos(0);
    setLocked(false);
  };

  const onMatch = () => {
    setLocked(true);
    setScore(s => s + 10);

    if (pos + 1 >= word.length) next();
    else setTimeout(() => {
      setPos(p => p + 1);
      setLocked(false);
    }, 800);
  };

  return (
    <div className="app">
      <header className="header">
        <h1>👋 Tremu na Oficina</h1>
        <p>Forma palavras com sinais de mão</p>
      </header>

      <div className="stats">
        <div className="card small">⭐ {score}</div>
        <div className="card small">🧠 {word.length - pos} letras</div>
      </div>

      <div className="word">
        {word.split("").map((l,i)=>(
          <span key={i} className={i===pos?"active":""}>
            {i < pos ? l : "_"}
          </span>
        ))}
      </div>

      <div className="grid">
        <div className="card">
          <HandSignSVG letter={word[pos]} />
          <div className="letter">{word[pos]}</div>
        </div>

        <WebcamRecognizer
          targetLetter={word[pos]}
          onMatch={onMatch}
          locked={locked}
        />
      </div>

      <button className="primary" onClick={next}>
        🔄 Nova palavra
      </button>
    </div>
  );
}