// src/app/components/NeuronsBackground.tsx
"use client";

import { useEffect, useState } from "react";
import Particles, { initParticlesEngine } from "@tsparticles/react";
import { loadSlim } from "@tsparticles/slim";
import type { ISourceOptions } from "@tsparticles/engine";

const OPTIONS: ISourceOptions = {
  background: { color: "transparent" },
  fpsLimit: 60,
  interactivity: {
    events: {
      onHover: { enable: true, mode: "repulse" },
      resize: { enable: true },            // <-- fix: object, not boolean
    },
    modes: { repulse: { distance: 120, duration: 0.4 } },
  },
  particles: {
    color: { value: "#14b8a6" },
    links: {
      enable: true,
      color: "#14b8a6",
      distance: 140,
      opacity: 0.2,
      width: 0.8,
    },
    move: {
      enable: true,
      speed: 1,
      outModes: { default: "out" },
    },
    number: {
      value: 45,
      density: { enable: true },
    },
    opacity: { value: 0.35 },
    shape: { type: "circle" },
    size: { value: { min: 1, max: 2.5 } },
  },
  detectRetina: true,
};

export default function NeuronsBackground() {
  const [ready, setReady] = useState(false);

  useEffect(() => {
    initParticlesEngine(async (engine) => {
      await loadSlim(engine); // lightweight bundle (links/move/repulse)
    }).then(() => setReady(true));
  }, []);

  if (!ready) return null;

  return <Particles id="tsparticles" className="fixed inset-0 -z-50" options={OPTIONS} />;
}
