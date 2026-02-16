"use client";
import React from "react";
import Contact from "../Contact/Contact";

interface CardItem {
  title: string;
  description: string;
  link: string;
}

const openSource: CardItem[] = [
  {
    title: "Haystack PR #2841",
    description: "Jina integration tests for embedders and ranker.",
    link: "https://github.com/deepset-ai/haystack-core-integrations/pull/2841",
  },
  {
    title: "Haystack PR #2821",
    description: "Added run_async to LlamaCppChatGenerator.",
    link: "https://github.com/deepset-ai/haystack-core-integrations/pull/2821",
  },
  {
    title: "Haystack PR #2802",
    description: "Removed archived Google integration workflows.",
    link: "https://github.com/deepset-ai/haystack-core-integrations/pull/2802",
  },
  {
    title: "Haystack PR #2805",
    description: "Fixed llama-stack >=0.4.0 integration defaults.",
    link: "https://github.com/deepset-ai/haystack-core-integrations/pull/2805",
  },
  {
    title: "Statsmodels #9660",
    description: "Gamma docs and weights parameterization clarification.",
    link: "https://github.com/statsmodels/statsmodels/pull/9660",
  },
  {
    title: "Skrub #1670",
    description: "Docs improvements and list-selector proposal.",
    link: "https://github.com/skrub-data/skrub/pull/1670",
  },
  {
    title: "Outlines #1814",
    description: "Parametrized Transformers tokenizer smoke tests.",
    link: "https://github.com/dottxt-ai/outlines/pull/1814",
  },
];

const writings: CardItem[] = [
  {
    title: "Prompt engineering was a phase, prompt design is the craft!",
    description:
      "Shift from prompt engineering to prompt design as a disciplined creative process.",
    link: "https://medium.com/ai-mind-labs/prompt-engineering-was-a-phase-prompt-design-is-the-craft-8a7027ce9d06",
  },
  {
    title: "How Text Chunking Works: The Foundation of Every RAG System",
    description:
      "Why chunking strategy directly affects retrieval quality in RAG pipelines.",
    link: "https://medium.com/@saranshsurana/how-text-chunking-works-the-foundation-of-every-rag-system-c162a8ad211b",
  },
  {
    title:
      "How Do You Measure an LLM's Intelligence? A Complete Guide to Evaluation Strategies",
    description:
      "Long-form guide to practical LLM evaluation metrics and strategies.",
    link: "https://medium.com/@saranshsurana/how-do-you-measure-an-llms-intelligence-a-complete-guide-to-evaluation-strategies-0a75a1cce3ba",
  },
];

export default function OpenSourceAndWriting() {
  return (
    <>
      <section
        id="open-source"
        className="pt-28 sm:pt-32 pb-10 px-4 sm:px-8 lg:px-12 bg-transparent text-slate-900 flex flex-col items-center section-top"
      >
        <div className="w-full">
          <h2 className="section-title text-3xl sm:text-4xl text-center mb-8">
            Open Source
          </h2>
          <p className="text-center text-slate-600 mb-10">
            Selected merged contributions across Haystack, Statsmodels, Skrub, and Outlines.
          </p>

          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-6 place-items-center">
            {openSource.map((item, i) => (
              <a
                key={i}
                href={item.link}
                target="_blank"
                rel="noopener noreferrer"
                className="group h-56 w-56 rounded-full bg-white ring-1 ring-slate-300 shadow-sm p-5 flex flex-col items-center justify-center text-center hover:shadow-md hover:-translate-y-1 transition"
                title={item.description}
              >
                <span className="text-base font-semibold text-slate-900 group-hover:text-blue-700">
                  {item.title}
                </span>
                <span className="mt-2 text-xs text-slate-500 leading-snug">
                  {item.description}
                </span>
              </a>
            ))}
          </div>
        </div>
      </section>

      <section
        id="writing"
        className="pt-2 sm:pt-4 pb-16 px-4 sm:px-8 lg:px-12 bg-transparent text-slate-900 flex flex-col items-center"
      >
        <div className="w-full">
          <h2 className="section-title text-3xl sm:text-4xl text-center mb-10">
            Writing
          </h2>

          <div className="grid gap-5 lg:grid-cols-3">
            {writings.map((item, i) => (
              <article
                key={i}
                className="rounded-2xl bg-white/95 ring-1 ring-slate-200 shadow-sm hover:shadow-md transition overflow-hidden"
              >
                <div className="h-1.5 bg-gradient-to-r from-slate-800 to-blue-700" />
                <div className="p-5">
                  <p className="text-[11px] uppercase tracking-[0.2em] text-slate-500 mb-2">
                    Medium Article
                  </p>
                  <a href={item.link} target="_blank" rel="noopener noreferrer">
                    <h3 className="text-lg font-semibold text-slate-900 hover:text-blue-700 leading-snug">
                      {item.title}
                    </h3>
                  </a>
                  <p className="mt-3 text-sm text-slate-600 leading-relaxed">
                    {item.description}
                  </p>
                </div>
              </article>
            ))}
          </div>
        </div>
      </section>

      <Contact />
    </>
  );
}
