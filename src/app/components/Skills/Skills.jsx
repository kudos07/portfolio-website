"use client";

import { useMemo, useState } from "react";

/** ================== Data ================== **/
const SKILLS = [
  // Languages
  { name: "Python", cat: "Languages" },
  { name: "SQL", cat: "Languages" },
  { name: "Java", cat: "Languages" },
  { name: "R", cat: "Languages" },
  { name: "Bash", cat: "Languages" },
  { name: "C/C++", cat: "Languages" },
  { name: "NoSQL", cat: "Languages" },
  { name: "Ocaml", cat: "Languages" },

  // ML & AI
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

  // Libraries
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


  // Databases
  { name: "MySQL", cat: "Databases" },
  { name: "MongoDB", cat: "Databases" },
  { name: "Hadoop", cat: "Databases" },
  { name: "Hive", cat: "Databases" },
  { name: "BigQuery", cat: "Databases" },

  // Cloud & Infra
  { name: "GCP", cat: "Cloud & Infra" },
  { name: "Azure", cat: "Cloud & Infra" },
  { name: "Docker", cat: "Cloud & Infra" },
  { name: "Kubernetes", cat: "Cloud & Infra" },
  { name: "MLflow", cat: "Cloud & Infra" },
  { name: "Jenkins", cat: "Cloud & Infra" },
  { name: "Kafka", cat: "Cloud & Infra" },
  { name: "Spark", cat: "Cloud & Infra" },

  // Analytics & Viz
  { name: "Tableau", cat: "Analytics & Viz" },
  { name: "Power BI", cat: "Analytics & Viz" },
  { name: "Excel", cat: "Analytics & Viz" },
  { name: "Grafana", cat: "Analytics & Viz" },
  { name: "Weights & Biases", cat: "Analytics & Viz" },

  // Statistics
  { name: "Regression", cat: "Statistics" },
  { name: "Classification", cat: "Statistics" },
  { name: "A/B Testing", cat: "Statistics" },
  { name: "Time Series", cat: "Statistics" },
  { name: "Bayesian Inference", cat: "Statistics" },
  { name: "Hypothesis Testing", cat: "Statistics" },
  { name: "SHAP", cat: "Statistics" },
];

/** Categories for filter chips */
const CATS = [
  "Languages",
  "ML & AI",
  "Libraries",
  "Databases",
  "Cloud & Infra",
  "Analytics & Viz",
  "Statistics",
];

/** ================== Tile ================== **/
function Tile({ name }) {
  return (
    <div
      className="tile flex items-center justify-center rounded-xl px-4 py-6 text-center font-semibold
                 bg-gradient-to-br from-teal-900/20 via-cyan-900/20 to-indigo-900/20
                 text-gray-100 hover:text-white
                 ring-1 ring-white/10 hover:ring-teal-400/40
                 shadow-md hover:shadow-teal-400/20
                 transition transform hover:-translate-y-1 hover:scale-105"
    >
      <span className="bg-clip-text text-transparent bg-gradient-to-r from-teal-300 via-cyan-200 to-indigo-300">
        {name}
      </span>
    </div>
  );
}

/** ================== Component ================== **/
export default function Skills() {
  const [active, setActive] = useState(CATS[0]); // default first category
  const [q, setQ] = useState("");

  const filtered = useMemo(() => {
    const list = SKILLS.filter(s => s.cat === active);
    const t = q.trim().toLowerCase();
    return t ? list.filter(s => s.name.toLowerCase().includes(t)) : list;
  }, [active, q]);

  return (
    <section id="skills" className="min-h-screen pt-28 px-6">
      <div className="mx-auto max-w-7xl">
        {/* Header */}
        <h2 className="text-4xl font-extrabold text-center mb-6">
          <span className="bg-clip-text text-transparent bg-gradient-to-r from-teal-300 via-cyan-200 to-indigo-300">
            Skills
          </span>
        </h2>

        {/* Filters */}
        <div className="flex flex-wrap items-center gap-2 justify-center mb-8">
          {CATS.map(c => (
            <button
              key={c}
              onClick={() => setActive(c)}
              className={`px-3.5 py-1.5 rounded-xl text-sm transition ${
                active === c
                  ? "bg-teal-400/20 text-teal-100 ring-1 ring-teal-400/40"
                  : "bg-white/5 text-gray-200 ring-1 ring-white/10 hover:bg-white/10"
              }`}
            >
              {c}
            </button>
          ))}
          <input
            value={q}
            onChange={e => setQ(e.target.value)}
            placeholder="Search a skill…"
            className="ml-2 w-56 px-3 py-1.5 rounded-xl bg-white/5 ring-1 ring-white/10 text-gray-100 placeholder:text-gray-400 focus:outline-none"
          />
        </div>

        {/* Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-5">
          {filtered.map((s) => (
            <Tile key={s.name} name={s.name} />
          ))}
        </div>
      </div>
    </section>
  );
}
