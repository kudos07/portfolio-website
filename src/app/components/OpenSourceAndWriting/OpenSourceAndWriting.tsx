"use client";
import React from "react";
import Contact from "../Contact/Contact";

interface Contribution {
  label: string;
  description: string;
  link: string;
}

interface RepoGroup {
  name: string;
  org: string;
  items: Contribution[];
}

const openSourceGroups: RepoGroup[] = [
  {
    name: "Haystack",
    org: "deepset-ai",
    items: [
      {
        label: "#3233",
        description: "Vespa integration with document store and keyword/embedding retrievers.",
        link: "https://github.com/deepset-ai/haystack-core-integrations/pull/3233",
      },
      {
        label: "#2932",
        description: "Added SUPPORTED_MODELS to AnthropicVertexChatGenerator (Claude on Vertex AI).",
        link: "https://github.com/deepset-ai/haystack-core-integrations/pull/2932",
      },
      {
        label: "#2841",
        description: "Jina integration tests for embedders and ranker.",
        link: "https://github.com/deepset-ai/haystack-core-integrations/pull/2841",
      },
      {
        label: "#2821",
        description: "Added run_async to LlamaCppChatGenerator.",
        link: "https://github.com/deepset-ai/haystack-core-integrations/pull/2821",
      },
      {
        label: "#2802",
        description: "Removed archived Google integration workflows.",
        link: "https://github.com/deepset-ai/haystack-core-integrations/pull/2802",
      },
      {
        label: "#2805",
        description: "Fixed llama-stack >=0.4.0 integration defaults.",
        link: "https://github.com/deepset-ai/haystack-core-integrations/pull/2805",
      },
    ],
  },
  {
    name: "Skrub",
    org: "skrub-data",
    items: [
      {
        label: "#2027",
        description: "Added has_dtype selector for column selection by dtype.",
        link: "https://github.com/skrub-data/skrub/pull/2027",
      },
      {
        label: "#1975",
        description: "Advanced tabular_pipeline guide examples for custom pipelines.",
        link: "https://github.com/skrub-data/skrub/pull/1975",
      },
      {
        label: "#1670",
        description: "Docs improvements and list-selector proposal.",
        link: "https://github.com/skrub-data/skrub/pull/1670",
      },
    ],
  },
  {
    name: "Outlines",
    org: "dottxt-ai",
    items: [
      {
        label: "#1814",
        description: "Parametrized Transformers tokenizer smoke tests.",
        link: "https://github.com/dottxt-ai/outlines/pull/1814",
      },
      {
        label: "#1829",
        description: "Tokenizer fragility: SPIECE_UNDERLINE dependency in transformers.file_utils.",
        link: "https://github.com/dottxt-ai/outlines/issues/1829",
      },
      {
        label: "#1819",
        description: "LlamaCppTokenizer: EOS masked as padding, fallback vocab truncation collisions.",
        link: "https://github.com/dottxt-ai/outlines/issues/1819",
      },
    ],
  },
  {
    name: "Statsmodels",
    org: "statsmodels",
    items: [
      {
        label: "#9660",
        description: "Gamma docs and weights parameterization clarification.",
        link: "https://github.com/statsmodels/statsmodels/pull/9660",
      },
    ],
  },
];

interface CardItem {
  title: string;
  description: string;
  link: string;
}

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
  const total = openSourceGroups.reduce((n, g) => n + g.items.length, 0);

  return (
    <>
      <section
        id="open-source"
        className="section-top flex flex-col items-center px-4 pb-12 pt-24 text-[var(--text-main)] sm:px-8 sm:pt-28 lg:px-12"
      >
        <div className="mx-auto w-full max-w-4xl">
          <h2 className="section-title reveal-title mb-3 text-center text-3xl sm:text-4xl">
            <span className="section-title-accent">Open Source</span>
          </h2>
          <p className="mb-10 text-center text-sm text-[var(--text-soft)]">
            {total} contributions across {openSourceGroups.length} projects — hover a number for details.
          </p>

          <div className="grid gap-4 sm:grid-cols-2">
            {openSourceGroups.map((group) => (
              <div
                key={group.name}
                className="border border-[var(--line-soft)] bg-[var(--bg-1)] p-5"
              >
                <div className="mb-3 flex items-baseline justify-between gap-2">
                  <div>
                    <h3 className="font-display text-lg font-medium text-[var(--text-main)]">
                      {group.name}
                    </h3>
                    <p className="text-xs text-[var(--text-soft)]">{group.org}</p>
                  </div>
                  <span className="tag">{group.items.length}</span>
                </div>
                <div className="flex flex-wrap gap-2">
                  {group.items.map((item) => (
                    <a
                      key={item.link}
                      href={item.link}
                      target="_blank"
                      rel="noopener noreferrer"
                      title={item.description}
                      className="rounded-md border border-[var(--line-soft)] bg-[var(--bg-0)] px-2.5 py-1 text-xs font-medium text-[var(--text-main)] transition hover:border-[var(--accent)] hover:text-[var(--accent)]"
                    >
                      {item.label}
                    </a>
                  ))}
                </div>
              </div>
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
