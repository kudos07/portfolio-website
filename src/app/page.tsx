"use client";

import { FaGithub, FaLinkedin, FaInstagram } from "react-icons/fa";
import { SiKaggle, SiLeetcode } from "react-icons/si";
import { useEffect, useRef } from "react";
import Image from "next/image";
// ⬇️ import your Experience section
import Experience from "./components/Experience/Experience";
import Projects from "./components/Projects/Projects";
import Skills from "./components/Skills/Skills";
import Contact from "./components/Contact/Contact";

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Animated background
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const particles: { x: number; y: number; vx: number; vy: number }[] = [];
    for (let i = 0; i < 60; i++) {
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.5,
        vy: (Math.random() - 0.5) * 0.5,
      });
    }

    function animate() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      particles.forEach((p, i) => {
        p.x += p.vx;
        p.y += p.vy;

        if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
        if (p.y < 0 || p.y > canvas.height) p.vy *= -1;

        ctx.beginPath();
        ctx.arc(p.x, p.y, 2, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(0,255,180,0.7)";
        ctx.fill();

        for (let j = i + 1; j < particles.length; j++) {
          const p2 = particles[j];
          const dist = Math.sqrt((p.x - p2.x) ** 2 + (p.y - p2.y) ** 2);
          if (dist < 120) {
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.strokeStyle = `rgba(0,255,180,${1 - dist / 120})`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        }
      });

      requestAnimationFrame(animate);
    }

    animate();

    const handleResize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
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
    <div className="relative min-h-screen text-white overflow-hidden">
      {/* Background */}
      <canvas ref={canvasRef} className="absolute inset-0 z-00" />

      {/* Navbar */}
      <nav className="flex justify-end space-x-8 p-6 bg-black/70 text-white fixed top-0 left-0 right-0 shadow-md z-20">
        <button onClick={() => scrollToSection("home")} className="cursor-pointer hover:text-teal-300">Home</button>
        <button onClick={() => scrollToSection("experience")} className="cursor-pointer hover:text-teal-300">Experience</button>
        <button onClick={() => scrollToSection("projects")} className="cursor-pointer hover:text-teal-300">Projects</button>
        <button onClick={() => scrollToSection("skills")} className="cursor-pointer hover:text-teal-300">Skills</button>
        <button onClick={() => scrollToSection("contact")} className="cursor-pointer hover:text-teal-300">Leave a Message</button>
      </nav>

      {/* Hero/About Section */}
      <section
        id="home"
        className="relative z-10 flex flex-col items-center justify-center min-h-screen px-6 pt-32"
      >
        <div className="flex flex-col md:flex-row items-center md:items-start gap-12 max-w-6xl w-full">
          {/* Left side */}
          <div className="flex flex-col items-center text-center">
            <Image
              src="/profile.jpg"
              alt="Profile Picture"
              width={256}
              height={256}
              className="w-64 h-64 rounded-full border-4 border-teal-400 shadow-lg object-cover"
            />
            <h1 className="text-3xl font-bold mt-6">Saransh Surana</h1>
            <p className="text-lg text-gray-300">
              Data Science • Machine Learning • Artificial Intelligence
            </p>

            {/* Social Icons */}
            <div className="flex space-x-6 mt-4 text-2xl">
              <a href="https://github.com/kudos07" target="_blank" rel="noopener noreferrer"><FaGithub /></a>
              <a href="https://linkedin.com/in/saransh-surana" target="_blank" rel="noopener noreferrer"><FaLinkedin /></a>
              <a href="https://www.instagram.com/saransh_07rm/" target="_blank" rel="noopener noreferrer"><FaInstagram /></a>
              <a href="https://www.kaggle.com/saranshsurana07" target="_blank" rel="noopener noreferrer"><SiKaggle /></a>
              <a href="https://leetcode.com/u/etiUzVdrA3/" target="_blank" rel="noopener noreferrer"><SiLeetcode /></a>
            </div>
          </div>

          {/* Right side */}
          <div className="flex-1">
            <h2 className="text-2xl font-bold mb-4">About Me</h2>
            <p className="text-gray-200 leading-relaxed">
              I’m a Data Scientist & AI Engineer with expertise in machine learning,
              deep learning, and large-scale data systems. My work focuses on
              building scalable, end-to-end ML solutions that solve real-world problems,
              from data preprocessing to deployment.
              <br /><br />
              I enjoy working at the intersection of AI research and practical applications,
              turning complex data into insights and intelligent systems. Looking ahead,
              I aim to contribute to cutting-edge AI innovation, particularly in areas like
              LLMs, generative AI, and optimization-driven ML systems, while driving
              measurable business impact.
            </p>
            <a
  href="/resume/saransh_surana_resume.pdf"
  download
  className="mt-6 inline-block px-6 py-3 bg-white text-green-800 font-semibold rounded-md shadow hover:bg-gray-200"
>
  Download Resume
</a>

            {/* Interests + Education */}
            <div className="flex flex-col md:flex-row gap-12 mt-10">
              <div>
                <h3 className="text-xl font-semibold mb-3">Interests</h3>
                <ul className="list-disc list-inside text-gray-200">
                  <li>Artificial Intelligence</li>
                  <li>Machine Learning</li>
                  <li>Deep Learning</li>
                  <li>Data Engineering</li>
                  <li>Statistics</li>
                </ul>
              </div>
              <div>
                <h3 className="text-xl font-semibold mb-3">Education</h3>
                <ul className="list-disc list-inside text-gray-200">
                  <li>M.S. Data Science – Stony Brook University</li>
                  <li>B.E. Electronics & Communication – Andhra University</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ⬇️ Experience (flash cards) */}
      <Experience />
      <Projects />
      <Skills />
      <Contact />

      {/* Other Sections */}
      {/* <section id="projects" className="min-h-screen pt-32 flex items-center justify-center bg-black bg-opacity-50">
        <h1 className="text-4xl font-bold">Projects</h1>
      </section> */}

      {/* <section id="skills" className="min-h-screen pt-32 flex items-center justify-center bg-black bg-opacity-50">
        <h1 className="text-4xl font-bold">Skills</h1>
      </section> */}

      {/* <section id="contact" className="min-h-screen pt-32 flex items-center justify-center bg-black bg-opacity-50">
        <h1 className="text-4xl font-bold">Leave a Message</h1>
      </section> */}
    </div>
  );
}
