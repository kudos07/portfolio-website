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
      distance: 150,
      opacity: 0.35,
      width: 1,
    },
    move: {
      enable: true,
      speed: 1.6,
      outModes: { default: "out" },
    },
    number: {
      value: 70,
      density: { enable: true}, // fine with current types
    },
    opacity: { value: 0.5 },
    shape: { type: "circle" },
    size: { value: { min: 1, max: 3 } },
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
