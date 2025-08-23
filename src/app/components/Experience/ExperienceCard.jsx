"use client";
import { useState } from "react";
import { FaXmark, FaArrowRight } from "react-icons/fa6";

export default function ExperienceCard({ role, company, duration, location, bullets, tech }) {
  const [open, setOpen] = useState(false);

  return (
    <>
      {/* CARD (grid view) */}
      <article
        onClick={() => setOpen(true)}
        className="relative cursor-pointer select-none rounded-3xl p-6 md:p-8 bg-[rgba(10,20,25,0.65)]
        ring-1 ring-white/10 hover:ring-teal-400/40 shadow-[0_10px_40px_-10px_rgba(0,0,0,0.7)]
        transition-all duration-300 hover:-translate-y-1 group overflow-hidden"
      >
        {/* gradient rim */}
        <div className="pointer-events-none absolute inset-0 rounded-3xl opacity-40 group-hover:opacity-70 blur-2xl
          bg-[radial-gradient(1000px_400px_at_0%_0%,rgba(45,212,191,0.25),transparent_60%),radial-gradient(1000px_400px_at_100%_100%,rgba(59,130,246,0.20),transparent_60%)]" />
        <div className="relative">
          <h3 className="text-2xl font-semibold tracking-tight">{role}</h3>
          <p className="text-gray-300">{company}</p>
          <p className="text-sm text-teal-300">{duration}{location ? ` · ${location}` : ""}</p>

          <div className="mt-6 flex items-center gap-2 text-teal-300 text-sm">
            <span>Open</span>
            <FaArrowRight className="transition-transform duration-300 group-hover:translate-x-1" />
          </div>
        </div>
      </article>

      {/* FULLSCREEN OVERLAY (detail view) */}
      {open && (
        <div className="fixed inset-0 z-[90]">
          {/* backdrop */}
          <div
            className="absolute inset-0 bg-black/70 backdrop-blur-lg animate-[fadeIn_180ms_ease-out]"
            onClick={() => setOpen(false)}
          />
          {/* sheet */}
          <div className="absolute inset-3 md:inset-10 rounded-3xl overflow-hidden
            ring-1 ring-white/10 bg-[rgba(10,15,20,0.8)]
            shadow-[0_40px_120px_-20px_rgba(0,0,0,0.8)]
            animate-[scaleIn_220ms_cubic-bezier(0.2,0.8,0.2,1)]">
            {/* glow border */}
            <div className="pointer-events-none absolute -inset-1 rounded-[2rem] opacity-60 blur-2xl
              bg-[conic-gradient(from_180deg_at_50%_50%,rgba(45,212,191,0.35),rgba(59,130,246,0.3),rgba(236,72,153,0.25),rgba(45,212,191,0.35))]" />
            {/* content */}
            <div className="relative h-full p-6 md:p-10 overflow-y-auto">
              {/* header */}
              <div className="flex items-start justify-between gap-6">
                <div>
                  <h3 className="text-3xl md:text-4xl font-bold leading-tight">{role}</h3>
                  <p className="text-lg text-gray-300">{company}</p>
                  <p className="text-sm text-teal-300">{duration}{location ? ` · ${location}` : ""}</p>
                </div>
                <button
                  onClick={() => setOpen(false)}
                  className="shrink-0 rounded-full p-3 ring-1 ring-white/10 bg-white/5 hover:bg-white/10 transition"
                  aria-label="Close"
                >
                  <FaXmark className="text-xl" />
                </button>
              </div>

              {/* body */}
              <div className="mt-8 grid gap-6 md:grid-cols-[1fr_minmax(260px,0.6fr)]">
                <ul className="list-disc list-inside text-gray-200 space-y-3">
                  {bullets.map((b, i) => (
                    <li key={i} className="leading-relaxed">{b}</li>
                  ))}
                </ul>
                <div className="rounded-2xl p-5 bg-white/5 ring-1 ring-white/10">
                  <h4 className="font-semibold mb-3">Tech</h4>
                  <div className="flex flex-wrap gap-2">
                    {(tech || []).map((t, i) => (
                      <span key={i} className="text-sm bg-teal-400/10 text-teal-200 ring-1 ring-teal-400/30 px-3 py-1 rounded-full">
                        {t}
                      </span>
                    ))}
                  </div>
                </div>
              </div>

              {/* footer */}
              <div className="mt-10 text-xs text-gray-400">
                Tip: Click outside or press <kbd className="px-1 py-0.5 bg-white/10 rounded">Esc</kbd> to close.
              </div>
            </div>
          </div>
          {/* esc key */}
          <EscHandler onClose={() => setOpen(false)} />
        </div>
      )}
    </>
  );
}

/** Close on ESC without re-rendering noise */
function EscHandler({ onClose }) {
  if (typeof window !== "undefined") {
    window.onkeydown = (e) => e.key === "Escape" && onClose();
  }
  return null;
}
