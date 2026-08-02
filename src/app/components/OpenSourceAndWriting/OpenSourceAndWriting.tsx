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
    title: "Haystack PR #3233",
    description:
      "Vespa integration with document store and keyword/embedding retrievers.",
    link: "https://github.com/deepset-ai/haystack-core-integrations/pull/3233",
  },
  {
    title: "Haystack PR #2932",
    description: "Added SUPPORTED_MODELS to AnthropicVertexChatGenerator (Claude on Vertex AI).",
    link: "https://github.com/deepset-ai/haystack-core-integrations/pull/2932",
  },
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
    title: "Skrub #2027",
    description: "Added has_dtype selector for column selection by dtype.",
    link: "https://github.com/skrub-data/skrub/pull/2027",
  },
  {
    title: "Skrub #1975",
    description:
      "Advanced tabular_pipeline guide examples for custom pipelines.",
    link: "https://github.com/skrub-data/skrub/pull/1975",
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
  {
    title: "Outlines Issue #1829",
    description: "Tokenizer fragility: SPIECE_UNDERLINE dependency in transformers.file_utils.",
    link: "https://github.com/dottxt-ai/outlines/issues/1829",
  },
  {
    title: "Outlines Issue #1819",
    description: "LlamaCppTokenizer: EOS masked as padding, fallback vocab truncation collisions.",
    link: "https://github.com/dottxt-ai/outlines/issues/1819",
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
        className="section-top flex flex-col items-center px-4 pb-12 pt-24 text-[var(--text-main)] sm:px-8 sm:pt-28 lg:px-12"
      >
        <div className="mx-auto w-full max-w-3xl">
          <h2 className="section-title reveal-title mb-3 text-center text-3xl sm:text-4xl">
            <span className="section-title-accent">Open Source</span>
          </h2>
          <p className="mb-10 text-center text-sm text-[var(--text-soft)]">
            Selected merged contributions across Haystack, Statsmodels, Skrub, and Outlines.
          </p>

          <div className="divide-y divide-[var(--line-soft)] border-y border-[var(--line-soft)]">
            {openSource.map((item, i) => (
              <a
                key={i}
                href={item.link}
                target="_blank"
                rel="noopener noreferrer"
                className="group block py-4 transition-colors hover:bg-[var(--bg-1)]/70"
                title={item.description}
              >
                <span className="font-medium text-[var(--text-main)] group-hover:text-[var(--accent)]">
                  {item.title}
                </span>
                <span className="mt-1 block text-sm leading-snug text-[var(--text-soft)]">
                  {item.description}
                </span>
              </a>
            ))}
          </div>
        </div>
      </section>

      <section
        id="writing"
        className="flex flex-col items-center px-4 pb-16 pt-8 text-[var(--text-main)] sm:px-8 lg:px-12"
      >
        <div className="mx-auto w-full max-w-5xl">
          <h2 className="section-title reveal-title mb-10 text-center text-3xl sm:text-4xl">
            <span className="section-title-accent">Writing</span>
          </h2>

          <div className="grid gap-6 lg:grid-cols-3">
            {writings.map((item, i) => (
              <article
                key={i}
                className="border border-[var(--line-soft)] bg-[var(--bg-1)] p-5 transition-colors hover:border-[var(--accent)]"
              >
                <p className="mb-2 text-[11px] uppercase tracking-[0.18em] text-[var(--text-soft)]">
                  Medium
                </p>
                <a href={item.link} target="_blank" rel="noopener noreferrer">
                  <h3 className="font-display text-lg font-medium leading-snug text-[var(--text-main)] hover:text-[var(--accent)]">
                    {item.title}
                  </h3>
                </a>
                <p className="mt-3 text-sm leading-relaxed text-[var(--text-soft)]">
                  {item.description}
                </p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <Contact />
    </>
  );
}
