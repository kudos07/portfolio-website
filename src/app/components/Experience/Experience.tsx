"use client";
import { useState } from "react";
import { FaChevronDown } from "react-icons/fa";

const EXPERIENCES = [
  {
    role: "AI Engineer",
    company: "Coupa",
    duration: "Jul 2026 - Present",
    location: "Bangalore, KA, India",
    bullets: [],
    tech: [],
  },
  {
    role: "AI Software Research Volunteer",
    company: "Schizophrenia & Psychosis Action Alliance",
    duration: "Jul 2025 - Jun 2026",
    location: "Remote, US",
    bullets: [
      "Built RAG-based structured extraction with JSON Schema guardrails, lifting parse success from 78% to 96% and cutting hallucinations by 25%.",
      "Ran 50+ MLflow experiments across models, prompts, chunking, and retrieval to systematically tune extraction quality.",
      "Shipped a FastAPI + Celery + Redis + PostgreSQL pipeline that processes 1.2-2k records/hour with under 2% timeouts and 35% fewer repeat LLM calls.",
      "Stored model outputs in PostgreSQL and performed slice analysis by county and service tag to identify error patterns, prioritize fixes, and support downstream product and operational decisions.",
      "Built production dbt models with incremental loads, snapshots, Jinja macros, tests, and documentation to monitor schema compliance and data quality across large datasets.",
      "Partnered with business stakeholders to ship a React and TypeScript review interface for sampling and triage, and communicated quality trends and model behavior clearly to support feature planning and decision-making.",
    ],
    tech: ["Python", "RAG", "Gemini Flash", "MLflow", "FastAPI", "Celery", "Redis", "PostgreSQL", "React.js", "TypeScript"],
  },
  {
    role: "Research Assistant",
    company: "Stony Brook University",
    duration: "Jan 2025 - May 2025",
    location: "Stony Brook, NY, US",
    bullets: [
      "Built Python pipelines to clean and preprocess unstructured data from web pages, PDFs, and other raw formats, version-controlled with Git for reproducibility and collaboration",
    ],
    tech: ["PySpark", "NumPy", "Pandas"],
  },
  {
    role: "Data Science Intern",
    company: "Ford Motor Company",
    duration: "May 2024 - Aug 2024",
    location: "Dearborn, MI, US",
    bullets: [
      "Built scalable ETL pipelines in BigQuery and GCP to support end-to-end ML workflows for anomaly detection on manufacturing sensor streams.",
      "Trained Isolation Forest and One-Class SVM models that achieved 78% recall and 73% precision for early fault detection.",
      "Turned anomaly insights into fixes by explaining model behavior and findings to both engineering and leadership teams.",
    ],
    tech: ["Python", "scikit-learn", "Pandas", "matplotlib", "GCP", "Anomaly Detection"],
  },
  {
    role: "Data Science Intern",
    company: "Napuor",
    duration: "Aug 2022 - Jan 2023",
    location: "Banaglore, KA, India",
    bullets: [
      "Deployed XGBoost demand-forecasting models on GCP with FastAPI and Docker, reducing forecast error by 18% across 30+ SKUs.",
      "Designed Spark / Hive / Kafka pipelines and SQL-based ETL that cut deployment time by 40% and supported 1K+ events per day.",
      "Lifted marketing ROI by 30% through clustering 10K+ customers into high-value segments and validating promotions with A/B tests (+7% conversion).",
    ],
    tech: [
      "XGBoost",
      "FastAPI",
      "Docker",
      "BigQuery",
      "GCP",
      "A/B Testing",
      "Kafka",
      "Spark",
      "Hive",
      "scikit-learn",
      "Clustering",
      "Marketing",
    ],
  },
];

export default function Experience() {
  const [openIndex, setOpenIndex] = useState<number | null>(null);

  const toggle = (i: number) => {
    setOpenIndex(openIndex === i ? null : i);
  };

  return (
    <section id="experience" className="pt-28 sm:pt-32 px-4 sm:px-6 pb-16 section-top">
      <div className="mx-auto max-w-4xl">
        <h2 className="section-title text-3xl sm:text-4xl mb-10 text-center">
          <span className="section-title-accent">Experience</span>
        </h2>

        <div className="space-y-4">
          {EXPERIENCES.map((exp, i) => (
            <div key={i} className="rounded-xl bg-white ring-1 ring-slate-200 shadow-sm hover:shadow-md transition">
              <button className="w-full flex items-center justify-between p-5 text-left hover:bg-slate-50/70 transition-colors" onClick={() => toggle(i)}>
                <div>
                  <h3 className="text-xl font-semibold text-slate-900">{exp.role}</h3>
                  <p className="text-slate-600">{exp.company}</p>
                  <p className="text-sm text-blue-700">{exp.duration} - {exp.location}</p>
                </div>
                <FaChevronDown
                  className={`transition-transform duration-300 ${openIndex === i ? "rotate-180 text-blue-700" : "text-slate-400"}`}
                />
              </button>

              <div
                className={`overflow-hidden transition-all duration-500 ${openIndex === i ? "max-h-[1000px] opacity-100" : "max-h-0 opacity-0"}`}
              >
                <div className="p-5 pt-0">
                  {exp.bullets.length > 0 ? (
                    <ul className="list-disc list-inside text-slate-700 space-y-2">
                      {exp.bullets.map((b, idx) => (
                        <li key={idx}>{b}</li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-sm text-slate-500 italic">Early in role — details coming soon.</p>
                  )}
                  {exp.tech?.length > 0 && (
                    <div className="mt-4 flex flex-wrap gap-2">
                      {exp.tech.map((t, idx) => (
                        <span key={idx} className="px-3 py-1 text-xs rounded-full bg-blue-50 text-blue-700 ring-1 ring-blue-200">
                          {t}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
