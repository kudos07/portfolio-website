"use client";

import { useMemo, useState } from "react";

const SKILLS = [
  { name: "Python", cat: "Languages" },
  { name: "SQL", cat: "Languages" },
  { name: "Java", cat: "Languages" },
  { name: "R", cat: "Languages" },
  { name: "Bash", cat: "Languages" },
  { name: "C/C++", cat: "Languages" },
  { name: "NoSQL", cat: "Languages" },
  { name: "Ocaml", cat: "Languages" },
  { name: "PyTorch", cat: "ML & AI" },
  { name: "TensorFlow", cat: "ML & AI" },
  { name: "Keras", cat: "ML & AI" },
  { name: "scikit-learn", cat: "ML & AI" },
  { name: "XGBoost", cat: "ML & AI" },
  { name: "LightGBM", cat: "ML & AI" },
  { name: "CatBoost", cat: "ML & AI" },
  { name: "FAISS", cat: "ML & AI" },
  { name: "Hugging Face Transformers", cat: "ML & AI" },
  { name: "spaCy", cat: "ML & AI" },
  { name: "Vision Transformers (ViT)", cat: "ML & AI" },
  { name: "Reinforcement Learning", cat: "ML & AI" },
  { name: "Recommendation Systems", cat: "ML & AI" },
  { name: "Pandas", cat: "Libraries" },
  { name: "NumPy", cat: "Libraries" },
  { name: "SciPy", cat: "Libraries" },
  { name: "Statsmodels", cat: "Libraries" },
  { name: "Seaborn", cat: "Libraries" },
  { name: "Plotly", cat: "Libraries" },
  { name: "NLTK", cat: "Libraries" },
  { name: "OpenCV", cat: "Libraries" },
  { name: "Matplotlib", cat: "Libraries" },
  { name: "CUDA", cat: "Libraries" },
  { name: "Playwright", cat: "Libraries" },
  { name: "MySQL", cat: "Databases" },
  { name: "MongoDB", cat: "Databases" },
  { name: "Hadoop", cat: "Databases" },
  { name: "Hive", cat: "Databases" },
  { name: "BigQuery", cat: "Databases" },
  { name: "GCP", cat: "Cloud & Infra" },
  { name: "Azure", cat: "Cloud & Infra" },
  { name: "Docker", cat: "Cloud & Infra" },
  { name: "Kubernetes", cat: "Cloud & Infra" },
  { name: "MLflow", cat: "Cloud & Infra" },
  { name: "Jenkins", cat: "Cloud & Infra" },
  { name: "Kafka", cat: "Cloud & Infra" },
  { name: "Spark", cat: "Cloud & Infra" },
  { name: "Tableau", cat: "Analytics & Viz" },
  { name: "Power BI", cat: "Analytics & Viz" },
  { name: "Excel", cat: "Analytics & Viz" },
  { name: "Grafana", cat: "Analytics & Viz" },
  { name: "Weights & Biases", cat: "Analytics & Viz" },
  { name: "Regression", cat: "Statistics" },
  { name: "Classification", cat: "Statistics" },
  { name: "A/B Testing", cat: "Statistics" },
  { name: "Time Series", cat: "Statistics" },
  { name: "Bayesian Inference", cat: "Statistics" },
  { name: "Hypothesis Testing", cat: "Statistics" },
  { name: "SHAP", cat: "Statistics" },
];

const CATS = [
  "Languages",
  "ML & AI",
  "Libraries",
  "Databases",
  "Cloud & Infra",
  "Analytics & Viz",
  "Statistics",
];

function Tile({ name }) {
  return (
    <div className="tile text-sm font-medium text-[var(--text-main)]">
      <span>{name}</span>
    </div>
  );
}

export default function Skills() {
  const [active, setActive] = useState(CATS[0]);
  const [q, setQ] = useState("");

  const filtered = useMemo(() => {
    const list = SKILLS.filter((s) => s.cat === active);
    const t = q.trim().toLowerCase();
    return t ? list.filter((s) => s.name.toLowerCase().includes(t)) : list;
  }, [active, q]);

  return (
    <section id="skills" className="section-top px-4 pb-20 pt-24 sm:px-8 sm:pt-28 lg:px-12">
      <div className="mx-auto w-full max-w-6xl">
        <h2 className="section-title reveal-title mb-10 text-center text-3xl sm:text-4xl">
          <span className="section-title-accent">Skills</span>
        </h2>
        <div className="mb-8 flex flex-wrap items-center justify-center gap-2">
          {CATS.map((c) => (
            <button
              key={c}
              onClick={() => setActive(c)}
              className={`rounded-md px-3 py-1.5 text-sm transition ${
                active === c
                  ? "bg-[var(--text-main)] text-white"
                  : "border border-[var(--line-soft)] bg-[var(--bg-1)] text-[var(--text-soft)] hover:border-[var(--accent)] hover:text-[var(--accent)]"
              }`}
            >
              {c}
            </button>
          ))}
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search a skill..."
            className="ml-1 w-52 rounded-md border border-[var(--line-soft)] bg-[var(--bg-1)] px-3 py-1.5 text-sm text-[var(--text-main)] placeholder:text-[var(--text-soft)] focus:border-[var(--accent)] focus:outline-none"
          />
        </div>

        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
          {filtered.map((s) => (
            <Tile key={s.name} name={s.name} />
          ))}
        </div>
      </div>
    </section>
  );
}
