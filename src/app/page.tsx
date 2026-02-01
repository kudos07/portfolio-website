"use client";

import { useState, useEffect } from "react";
import { FaGithub, FaLinkedin, FaInstagram } from "react-icons/fa";
import { SiKaggle, SiLeetcode } from "react-icons/si";
import Image from "next/image";

import Experience from "./components/Experience/Experience";
import Projects from "./components/Projects/Projects";
import Skills from "./components/Skills/Skills";
import Contact from "./components/Contact/Contact";
import OpenSourceAndWriting from "./components/OpenSourceAndWriting/OpenSourceAndWriting";

export default function Home() {
  const [showScrollHint, setShowScrollHint] = useState(true);
  useEffect(() => {
    const onScroll = () => setShowScrollHint(window.scrollY < 80);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

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
    <div className="relative min-h-screen text-white overflow-x-hidden">
      {/* Navbar */}
      <nav className="flex flex-wrap justify-between items-center gap-4 px-4 sm:px-6 py-4 bg-black/70 backdrop-blur-md text-white fixed top-0 left-0 right-0 z-20 border-b border-white/[0.06]">
        <button onClick={() => scrollToSection("home")} className="text-base font-semibold tracking-tight text-white hover:text-teal-300 transition">
          Saransh Surana
        </button>
        <div className="flex flex-wrap items-center justify-end gap-3 sm:gap-5">
          <button onClick={() => scrollToSection("experience")} className="text-sm font-medium text-gray-400 hover:text-teal-300 transition">Experience</button>
          <button onClick={() => scrollToSection("projects")} className="text-sm font-medium text-gray-400 hover:text-teal-300 transition">Projects</button>
          <button onClick={() => scrollToSection("skills")} className="text-sm font-medium text-gray-400 hover:text-teal-300 transition">Skills</button>
          <button onClick={() => scrollToSection("open-source")} className="text-sm font-medium text-gray-400 hover:text-teal-300 transition hidden sm:inline">Writing</button>
          <button onClick={() => scrollToSection("contact")} className="text-sm font-medium text-gray-400 hover:text-teal-300 transition">Contact</button>
        </div>
      </nav>

      {/* Hero — gradient name only here for focus */}
      <section
        id="home"
        className="relative z-10 flex flex-col items-center justify-center min-h-screen px-4 sm:px-6 pt-24 sm:pt-28 pb-20"
      >
        <div className="flex flex-col md:flex-row items-center md:items-start gap-10 md:gap-14 max-w-5xl w-full">
          <div className="flex flex-col items-center text-center md:text-left md:items-start shrink-0">
            <Image
              src="/profile.png"
              alt="Saransh Surana"
              width={220}
              height={220}
              className="w-52 h-52 sm:w-56 sm:h-56 rounded-full border-2 border-teal-400/50 object-cover shadow-2xl shadow-teal-900/20"
            />
            <h1 className="text-2xl sm:text-3xl font-bold mt-5 bg-clip-text text-transparent bg-gradient-to-r from-teal-200 via-cyan-100 to-teal-300">
              Saransh Surana
            </h1>
            <p className="text-sm sm:text-base text-gray-400 mt-1">
              Data Science · ML · AI
            </p>
            <div className="flex gap-5 mt-4 text-xl text-gray-500 [&>a:hover]:text-teal-300 [&>a]:transition">
              <a href="https://github.com/kudos07" target="_blank" rel="noopener noreferrer" aria-label="GitHub"><FaGithub /></a>
              <a href="https://linkedin.com/in/saransh-surana" target="_blank" rel="noopener noreferrer" aria-label="LinkedIn"><FaLinkedin /></a>
              <a href="https://www.instagram.com/saransh_07rm/" target="_blank" rel="noopener noreferrer" aria-label="Instagram"><FaInstagram /></a>
              <a href="https://www.kaggle.com/saranshsurana07" target="_blank" rel="noopener noreferrer" aria-label="Kaggle"><SiKaggle /></a>
              <a href="https://leetcode.com/u/etiUzVdrA3/" target="_blank" rel="noopener noreferrer" aria-label="LeetCode"><SiLeetcode /></a>
            </div>
          </div>

          <div className="flex-1 min-w-0">
            <h2 className="text-lg font-semibold text-gray-200 mb-3">About me</h2>
            <p className="text-gray-300 text-sm sm:text-base leading-relaxed">
              I&apos;m a Data Scientist & AI Engineer with expertise in machine learning,
              deep learning, and large-scale data systems. My work focuses on
              building scalable, end-to-end ML solutions that solve real-world problems,
              from data preprocessing to deployment.
            </p>
            <p className="text-gray-300 text-sm sm:text-base leading-relaxed mt-4">
              I enjoy working at the intersection of AI research and practical applications,
              turning complex data into insights and intelligent systems. I aim to contribute
              to cutting-edge AI innovation—LLMs, generative AI, optimization-driven ML—while
              driving measurable business impact.
            </p>
            <a
              href="/resume/saransh_surana_resume.pdf"
              download
              className="mt-5 inline-block px-5 py-2.5 text-sm font-semibold rounded-lg bg-teal-400/20 text-teal-300 ring-1 ring-teal-400/30 hover:bg-teal-400/30 hover:ring-teal-400/50 transition"
            >
              Download Resume
            </a>
            <div className="flex flex-col sm:flex-row gap-8 mt-8">
              <div>
                <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-widest mb-2">Interests</h3>
                <p className="text-gray-400 text-sm">AI · ML · Deep Learning · Data Engineering · Statistics</p>
              </div>
              <div>
                <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-widest mb-2">Education</h3>
                <p className="text-gray-400 text-sm">M.S. Data Science — Stony Brook · B.E. ECE — Andhra University</p>
              </div>
            </div>
          </div>
        </div>
        {showScrollHint && (
          <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-10 text-gray-400 text-xs font-medium tracking-wider animate-pulse" aria-hidden>
            Scroll
          </div>
        )}
      </section>

      {/* Section divider */}
      <div className="h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" aria-hidden />

      <Experience />
      <Projects />
      <Skills />
      <OpenSourceAndWriting />
      <Contact />
    </div>
  );
}
