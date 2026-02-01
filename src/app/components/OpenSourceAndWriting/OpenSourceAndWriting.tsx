"use client";
import React from "react";

interface CardItem {
  title: string;
  description: string;
  link: string;
}

const openSource: CardItem[] = [
  {
    title: "Statsmodels",
    description:
      "Contributed documentation improvements (merged PR #9660 — clarified `Gamma` `loglike_obs` and `weights` parameterization) and currently implementing rotated copula support for enhanced tail-dependence modeling.",
    link: "https://github.com/statsmodels/statsmodels/pull/9660",
  },
  {
    title: "Skrub",
    description:
      "Authored two contributions — one improving documentation and example clarity (merged), and another proposing a new `select_list` / `is_list` selector feature for list-like columns (closed after backend discussions). Both contributions deepened library understanding and informed future selector design.",
      link: "https://github.com/skrub-data/skrub/pull/1670",
  },
];

const writings: CardItem[] = [
  {
    title: "Prompt engineering was a phase, prompt design is the craft!",
    description:
      "Published under AI Mind. Covers the shift from prompt engineering to prompt design as a disciplined creative process.",
    link: "https://medium.com/ai-mind-labs/prompt-engineering-was-a-phase-prompt-design-is-the-craft-8a7027ce9d06",
  },
  {
    title: "How Text Chunking Works: The Foundation of Every RAG System",
    description:
      "10-minute technical deep dive explaining why chunking strategies determine retrieval quality in RAG pipelines.",
    link: "https://medium.com/@saranshsurana/how-text-chunking-works-the-foundation-of-every-rag-system-c162a8ad211b",
  },
  {
    title:
      "How Do You Measure an LLM’s Intelligence? A Complete Guide to Evaluation Strategies",
    description:
      "27-minute, 6.6 K-word guide on LLM evaluation metrics — 16 claps and 4 reader highlights.",
    link: "https://medium.com/@saranshsurana/how-do-you-measure-an-llms-intelligence-a-complete-guide-to-evaluation-strategies-0a75a1cce3ba",
  },
];

export default function OpenSourceAndWriting() {
  return (
    <section
      id="open-source"
      className="pt-28 sm:pt-32 pb-16 px-4 sm:px-6 bg-transparent text-gray-100 flex flex-col items-center section-top"
    >
      <h2 className="section-title text-3xl sm:text-4xl text-center mb-10">
        <span className="section-title-accent">Open Source & Writing</span>
      </h2>
      <div className="grid md:grid-cols-2 gap-8 max-w-5xl w-full">
        {/* ---------- Open Source Section ---------- */}
        <div>
          <h3 className="text-2xl font-semibold mb-6">
            Open Source Contributions
          </h3>
          <ul className="space-y-6">
            {openSource.map((item, i) => (
              <li
                key={i}
                className="border border-teal-400/20 bg-black/20 backdrop-blur-sm p-4 rounded-xl hover:border-teal-400/60 hover:shadow-[0_0_15px_rgba(45,212,191,0.4)] transition-all duration-300"
              >
                <a
                  href={item.link}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <h4 className="text-xl font-semibold text-teal-400 mb-2 hover:underline">
                    {item.title}
                  </h4>
                  <p className="text-gray-300">{item.description}</p>
                </a>
              </li>
            ))}
          </ul>
        </div>

        {/* ---------- Writing Section ---------- */}
        <div>
          <h3 className="text-2xl font-semibold mb-6">
            Writing & Publications
          </h3>
          <ul className="space-y-6">
            {writings.map((item, i) => (
              <li
                key={i}
                className="border border-teal-400/20 bg-black/20 backdrop-blur-sm p-4 rounded-xl hover:border-teal-400/60 hover:shadow-[0_0_15px_rgba(45,212,191,0.4)] transition-all duration-300"
              >
                <a
                  href={item.link}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <h4 className="text-xl font-semibold text-teal-400 mb-2 hover:underline">
                    {item.title}
                  </h4>
                  <p className="text-gray-300">{item.description}</p>
                </a>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}
