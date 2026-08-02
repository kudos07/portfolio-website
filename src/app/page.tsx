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
  { id: "home", label: "About", hideOnSmall: false },
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
  const [navScrolled, setNavScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => {
      setShowScrollHint(window.scrollY < 80);
      setNavScrolled(window.scrollY > 24);

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
    `${hideOnSmall ? "hidden sm:inline " : ""}text-sm font-medium transition-colors ${
      activeSection === id
        ? "text-[var(--text-main)] border-b border-[var(--accent)] pb-0.5"
        : "text-[var(--text-soft)] hover:text-[var(--accent)]"
    }`;

  const scrollToSection = (id: string) => {
    if (id === "home") {
      window.scrollTo({ top: 0, behavior: "smooth" });
      return;
    }

    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  return (
    <div className="relative min-h-screen overflow-x-hidden text-[var(--text-main)]">
      <nav
        className={`fixed top-0 left-0 right-0 z-20 flex flex-wrap items-center justify-between gap-4 border-b px-4 py-3.5 sm:px-8 lg:px-12 transition-all duration-300 ${
          navScrolled
            ? "border-[var(--line-soft)] bg-[var(--bg-1)]/90 backdrop-blur-md"
            : "border-transparent bg-transparent"
        }`}
      >
        <button
          onClick={() => scrollToSection("home")}
          className="font-display text-base font-semibold tracking-tight text-[var(--text-main)] transition hover:text-[var(--accent)]"
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
        className="relative z-10 min-h-[100svh] px-4 pt-24 pb-16 sm:px-8 lg:px-12 sm:pt-28 sm:pb-20"
      >
        <div className="mx-auto grid min-h-[calc(100svh-8rem)] w-full max-w-6xl items-stretch gap-10 lg:grid-cols-[1.05fr_0.95fr] lg:gap-0">
          <div className="flex flex-col justify-center lg:pr-12">
            <p className="animate-fade-up text-xs font-medium uppercase tracking-[0.22em] text-[var(--text-soft)]">
              AI Engineer · Data Science · ML
            </p>
            <h1 className="animate-fade-up-delay font-display mt-4 text-5xl font-medium leading-[1.05] tracking-tight text-[var(--text-main)] sm:text-6xl lg:text-7xl">
              Saransh Surana
            </h1>
            <p className="animate-fade-up-delay mt-5 max-w-lg text-xl font-medium leading-snug text-[var(--text-main)] sm:text-2xl">
              I ship AI systems that actually get used.
            </p>
            <p className="animate-fade-up-delay-2 mt-4 max-w-md text-base leading-relaxed text-[var(--text-soft)]">
              From messy data to deployed models — shipped systems that move metrics for real teams.
              Coupa · S&amp;PAA · Ford · Stony Brook.
            </p>

            <div className="animate-fade-up-delay-2 mt-8 flex flex-wrap items-center gap-3">
              <button
                type="button"
                onClick={() => scrollToSection("experience")}
                className="inline-flex items-center justify-center rounded-md bg-[var(--text-main)] px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-[var(--accent)]"
              >
                See what I&apos;ve shipped
              </button>
              <a
                href="/resume/saransh_surana_resume.pdf"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center justify-center rounded-md border border-[var(--line-soft)] bg-[var(--bg-1)] px-5 py-2.5 text-sm font-semibold text-[var(--text-main)] transition hover:border-[var(--accent)] hover:text-[var(--accent)]"
              >
                Download resume
              </a>
            </div>

            <div className="animate-fade-up-delay-2 mt-8 flex gap-5 text-lg text-[var(--text-soft)] [&>a:hover]:text-[var(--accent)] [&>a]:transition">
              <a href="https://github.com/kudos07" target="_blank" rel="noopener noreferrer" aria-label="GitHub">
                <FaGithub />
              </a>
              <a href="https://linkedin.com/in/saransh-surana" target="_blank" rel="noopener noreferrer" aria-label="LinkedIn">
                <FaLinkedin />
              </a>
              <a href="https://www.instagram.com/saransh_07rm/" target="_blank" rel="noopener noreferrer" aria-label="Instagram">
                <FaInstagram />
              </a>
              <a href="https://www.kaggle.com/saranshsurana07" target="_blank" rel="noopener noreferrer" aria-label="Kaggle">
                <SiKaggle />
              </a>
              <a href="https://leetcode.com/u/etiUzVdrA3/" target="_blank" rel="noopener noreferrer" aria-label="LeetCode">
                <SiLeetcode />
              </a>
            </div>
          </div>

          <div className="animate-fade-in relative min-h-[22rem] overflow-hidden border border-[var(--line-soft)] bg-[var(--bg-1)] sm:min-h-[28rem] lg:min-h-full lg:border-y-0 lg:border-r-0 lg:border-l">
            <Image
              src="/profile.png"
              alt="Saransh Surana"
              fill
              priority
              sizes="(max-width: 1024px) 100vw, 48vw"
              className="object-cover object-top"
            />
          </div>
        </div>

        {showScrollHint && (
          <div
            className="pointer-events-none absolute bottom-6 left-1/2 -translate-x-1/2 text-[10px] font-medium uppercase tracking-[0.28em] text-[var(--text-soft)]"
            aria-hidden
          >
            Scroll
          </div>
        )}
      </section>

      <Experience />
      <Projects />
      <Skills />
      <OpenSourceAndWriting />
    </div>
  );
}
