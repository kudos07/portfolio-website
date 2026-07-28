"use client";

import { useEffect, useState } from "react";
import { FaGithub, FaInstagram, FaLinkedin } from "react-icons/fa";
import { SiKaggle, SiLeetcode } from "react-icons/si";
import Image from "next/image";

import Experience from "./components/Experience/Experience";
import Projects from "./components/Projects/Projects";
import Skills from "./components/Skills/Skills";
import OpenSourceAndWriting from "./components/OpenSourceAndWriting/OpenSourceAndWriting";

const NAV_ITEMS = [
  { id: "home", label: "About me", hideOnSmall: false },
  { id: "experience", label: "Experience", hideOnSmall: false },
  { id: "projects", label: "Projects", hideOnSmall: false },
  { id: "skills", label: "Skills", hideOnSmall: false },
  { id: "open-source", label: "Open Source", hideOnSmall: true },
  { id: "writing", label: "Writing", hideOnSmall: true },
  { id: "contact", label: "Contact", hideOnSmall: false },
] as const;

export default function Home() {
  const [showScrollHint, setShowScrollHint] = useState(true);
  const [activeSection, setActiveSection] = useState("home");

  useEffect(() => {
    const onScroll = () => {
      setShowScrollHint(window.scrollY < 80);

      const probe = window.scrollY + 140;
      let current = "home";

      for (const item of NAV_ITEMS) {
        const el = document.getElementById(item.id);
        if (el && el.offsetTop <= probe) {
          current = item.id;
        }
      }
      setActiveSection(current);
    };

    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const navBtnClass = (id: string, hideOnSmall = false) =>
    `${hideOnSmall ? "hidden sm:inline " : ""}text-sm font-medium transition ${
      activeSection === id
        ? "text-slate-900 border-b-2 border-blue-700 pb-0.5"
        : "text-slate-600 hover:text-blue-700"
    }`;

  const scrollToSection = (id: string) => {
    if (id === "home") {
      window.scrollTo({ top: 0, behavior: "smooth" });
      return;
    }

    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "start" });
    } else {
      console.warn(`Section with ID '${id}' not found`);
    }
  };

  return (
    <div className="relative min-h-screen text-slate-900 overflow-x-hidden">
      <nav className="fixed top-0 left-0 right-0 z-20 flex flex-wrap items-center justify-between gap-4 border-b border-slate-900/10 bg-white/90 px-4 py-4 text-slate-900 backdrop-blur-xl shadow-[0_8px_24px_rgba(15,23,42,0.06)] sm:px-8 lg:px-12">
        <button
          onClick={() => scrollToSection("home")}
          className={`text-base font-semibold tracking-tight transition ${
            activeSection === "home" ? "text-slate-900" : "text-slate-900 hover:text-blue-700"
          }`}
        >
          Saransh Surana
        </button>
        <div className="flex flex-wrap items-center justify-end gap-3 sm:gap-5">
          {NAV_ITEMS.map((item) => (
            <button
              key={item.id}
              onClick={() => scrollToSection(item.id)}
              className={navBtnClass(item.id, item.hideOnSmall)}
              aria-current={activeSection === item.id ? "page" : undefined}
            >
              {item.label}
            </button>
          ))}
        </div>
      </nav>

      <section
        id="home"
        className="relative z-10 flex flex-col items-center justify-center px-4 pt-24 pb-12 sm:px-8 lg:px-12 sm:pt-28 sm:pb-14"
      >
        <div className="grid w-full gap-8 lg:grid-cols-[320px_1fr]">
          <div className="rounded-2xl bg-white/90 ring-1 ring-slate-200 p-6 shadow-sm flex flex-col items-center text-center">
            <Image
              src="/profile.png"
              alt="Saransh Surana"
              width={220}
              height={220}
              className="h-52 w-52 rounded-full border-2 border-blue-200 object-cover shadow-2xl shadow-slate-900/10 sm:h-56 sm:w-56"
            />
            <h1 className="mt-5 text-2xl font-bold text-slate-900">
              Saransh Surana
            </h1>
            <p className="mt-1 text-sm text-slate-600">
              Data Science - ML - AI
            </p>
            <div className="mt-4 flex gap-5 text-xl text-slate-500 [&>a:hover]:text-blue-700 [&>a]:transition">
              <a href="https://github.com/kudos07" target="_blank" rel="noopener noreferrer" aria-label="GitHub"><FaGithub /></a>
              <a href="https://linkedin.com/in/saransh-surana" target="_blank" rel="noopener noreferrer" aria-label="LinkedIn"><FaLinkedin /></a>
              <a href="https://www.instagram.com/saransh_07rm/" target="_blank" rel="noopener noreferrer" aria-label="Instagram"><FaInstagram /></a>
              <a href="https://www.kaggle.com/saranshsurana07" target="_blank" rel="noopener noreferrer" aria-label="Kaggle"><SiKaggle /></a>
              <a href="https://leetcode.com/u/etiUzVdrA3/" target="_blank" rel="noopener noreferrer" aria-label="LeetCode"><SiLeetcode /></a>
            </div>
          </div>

          <div className="min-w-0 rounded-2xl bg-white/90 ring-1 ring-slate-200 p-6 sm:p-8 shadow-sm">
            <p className="text-xs uppercase tracking-[0.2em] text-slate-500 mb-2">
              Saransh • Data Science • AI
            </p>
            <h2 className="mb-3 text-3xl sm:text-4xl font-semibold text-slate-900">
              I ship AI systems that actually get used.
            </h2>
            <p className="text-sm leading-relaxed text-slate-700 sm:text-base">
              From messy data to deployed models – I care less about leaderboard scores and more about shipped systems
              that move metrics for real teams.
            </p>
            <p className="mt-3 text-xs sm:text-sm text-slate-500">
              Python • LLMs &amp; RAG • MLOps • Experimentation • Evaluation
            </p>
            <div className="mt-5 flex flex-wrap gap-3">
              <button
                type="button"
                onClick={() => scrollToSection("experience")}
                className="inline-flex items-center justify-center rounded-lg bg-slate-900 px-5 py-2.5 text-sm font-semibold text-white shadow-[0_10px_24px_rgba(15,23,42,0.2)] transition hover:bg-slate-800"
              >
                See what I&apos;ve shipped
              </button>
            </div>
            <div className="mt-8 grid gap-4 sm:grid-cols-2">
              <div className="rounded-xl bg-slate-50 ring-1 ring-slate-200 p-4">
                <h3 className="mb-1 text-xs font-semibold uppercase tracking-widest text-slate-500">
                  What I like working on
                </h3>
                <ul className="space-y-1.5 text-sm text-slate-700">
                  <li>Turning vague ideas into concrete experiments.</li>
                  <li>Building RAG + LLM systems that don&apos;t hallucinate.</li>
                  <li>Making dashboards and pipelines boringly reliable.</li>
                </ul>
              </div>
              <div className="rounded-xl bg-slate-50 ring-1 ring-slate-200 p-4">
                <h3 className="mb-1 text-xs font-semibold uppercase tracking-widest text-slate-500">
                  Where I&apos;ve been
                </h3>
                <p className="text-sm text-slate-700">
                  Coupa (AI Engineer), S&amp;PAA (RAG &amp; data), Ford (Data Science), Stony Brook (MS Data Science),
                  and startups where I shipped models instead of slide decks.
                </p>
              </div>
            </div>
          </div>
        </div>

        {showScrollHint && (
          <div className="fixed bottom-6 left-1/2 z-10 -translate-x-1/2 animate-pulse text-xs font-medium tracking-wider text-slate-500" aria-hidden>
            Scroll
          </div>
        )}
      </section>

      <div className="h-px bg-gradient-to-r from-transparent via-slate-300 to-transparent" aria-hidden />

      <Experience />
      <Projects />
      <Skills />
      <OpenSourceAndWriting />
    </div>
  );
}
