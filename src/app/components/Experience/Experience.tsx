"use client";
import { useState } from "react";
import { FaChevronDown } from "react-icons/fa";

const EXPERIENCES = [
  {
    role: "AI Software Research Volunteer",
    company: "Schizophrenia & Psychosis Action Alliance",
    duration: "Jul 2025 - Present",
    location: "Remote, US",
    bullets: [
      "Scraped and structured 50,000+ housing and social service records across multiple counties into machine-readable datasets.",
      "Automated web data extraction using Playwright with asynchronous concurrent scraping, reducing collection time by ~70%",
      "Designed a deduplication framework to merge duplicate organizations while preserving unique attributes, cutting redundancy by ~35%.",
      "Implemented a relevance-filtering prompt system that improved classification accuracy of housing-related records to >60% precision.",
      "Delivered a final cleaned dataset for NGO partners, enabling more accurate housing service mapping and supporting advocacy for individuals with serious illness.",
    ],
    tech: ["Python", "Playwright", "Gemini Pro", "Pandas"],
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
      "Built scalable ETL pipelines in BigQuery and SQL on GCP to support end-to-end ML workflows for anomaly detection.",
      "Enhanced early fault detection by identifying spikes and irregular patterns in manufacturing time-series data through Z-score thresholds, facilitating effective data visualization for analysis.",
      "Trained unsupervised models (Isolation Forest, One-Class SVM) to detect anomalies in manufacturing sensor data achieving 78% recall and 73% precision, supporting early fault detection.",
      "Explained model results to technical and non-technical teams and engaged with data science experts to learn more about the field, supporting fault resolution and alignment.",
    ],
    tech: ["Python", "scikit-learn", "Pandas", "matplotlib", "GCP", "Anomaly Detection"],
  },
  {
    role: "Data Science Intern",
    company: "Napuor",
    duration: "Aug 2022 - Jan 2023",
    location: "Banaglore, KA, India",
    bullets: [
      "Developed real-time demand forecasting and inventory optimization by deploying XGBoost models on GCP using FAST API and Docker, reducing forecast error by 18% across 30+ SKUs.",
      "Designed end-to-end ML data pipelines on unstructured data and SQL-based ETL workflows using Spark, Hive, and Kafka, accelerating deployment time by 40% and supporting analysis of 1K+ events daily.",
      "Drove 30% marketing ROI uplift by applying clustering on 10K+ customer profiles, enabling business teams to target high-value segments effectively.",
      "Conducted A/B testing on promotional strategies and new product placements across multiple regions, identifying winning variants that increased sales conversion by 7%.",
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
                  <ul className="list-disc list-inside text-slate-700 space-y-2">
                    {exp.bullets.map((b, idx) => (
                      <li key={idx}>{b}</li>
                    ))}
                  </ul>
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
