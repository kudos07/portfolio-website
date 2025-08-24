"use client";
import { useMemo, useState } from "react";
import Image, { StaticImageData } from "next/image";
import { FaGithub, FaExternalLinkAlt, FaTimes, FaDownload } from "react-icons/fa";

/* ====== Local hero & project images (NEXT TO THIS FILE) ======
   src/app/components/Projects/images/*
================================================================ */
import dsmlHero from "./images/cat-dsml.png";
import dlHero from "./images/cat-dl.png";
import llmHero from "./images/lln.png";
import nlpHero from "./images/nlp.png";
import face1 from "./images/histogram.png";
import stry from "./images/strybd-1.png";
import arch from "./images/storyboard-architecture.png"; // architecture diagram
import mhcb from "./images/chatbot.png"
import wall from "./images/chatbot-wall.webp"
import img2 from "./images/dsml-p2.jpeg"
import img1 from "./images/CTR_at_k_episode.png"
import img3 from "./images/NDCG_at_k.png"
import img4 from "./images/rlpro.jpg"
import img5 from "./images/cvb.jpg"
import img6 from "./images/img1.png"
import img7 from "./images/img2.png"
import img8 from "./images/img3.png"
// import img9 from "./images/img4.png"
// import img10 from "./images/img5.png"

/* ======================= Types ======================= */
export type ProjectDetails = {
  problem?: string;
  data?: string;
  approach?: string[] | string;   // allow either short paragraph or bullet list
  impact?: string;
  architecture?: string[];        // compact architecture bullets (optional)
  resultsImages?: (string | StaticImageData)[];
  resultsCaption?: string;
};

export type Project = {
  title: string;
  subtitle?: string;
  bullets?: string[];
  tech?: string[];
  images?: (string | StaticImageData)[];   // imported images or /public paths
  architectureImg?: string | StaticImageData; // diagram image (optional)
  reportUrl?: string;                       // e.g., "/reports/xyz.pdf/html"
  github?: string;
  demo?: string;
  year?: number;
  details?: ProjectDetails;
};

export type Category = {
  name: string;
  heroImg?: string | StaticImageData;
  heroOverlayText?: string;
  projects: Project[];
};

/* ===================== Helpers ====================== */
const PLACEHOLDER =
  "data:image/svg+xml;utf8," +
  encodeURIComponent(
    `<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 400 240'>
      <defs>
        <linearGradient id='g' x1='0' y1='0' x2='1' y2='1'>
          <stop offset='0%' stop-color='#14b8a6' stop-opacity='0.2'/>
          <stop offset='100%' stop-color='#60a5fa' stop-opacity='0.2'/>
        </linearGradient>
      </defs>
      <rect width='400' height='240' fill='url(#g)'/>
      <text x='50%' y='50%' dominant-baseline='middle' text-anchor='middle'
        font-family='Inter,Arial' font-size='16' fill='#9ca3af'>No preview</text>
    </svg>`
  );

const fileNameFromUrl = (url: string, fallback = "report.pdf") => {
  try {
    const clean = url.split("#")[0].split("?")[0];
    return clean.split("/").pop() || fallback;
  } catch {
    return fallback;
  }
};

/* ============== Data (PERSONAL PROJECTS) ============== */
/* NOTE: Put PDFs/HTML reports in public/reports/* */
const CATEGORIES: Category[] = [
  {
    name: "Data Science & ML",
    heroImg: dsmlHero,
    heroOverlayText: "Experiments • Forecasting • Segmentation",
    projects: [
      {
        title: "Writing Quality Prediction",
        subtitle: "Linking typing behavior to essay scores (Kaggle dataset)",
        tech: ["R", "Statistics", "Regression Analysis", "Data Visualization"],
        reportUrl: "/reports/project_report.pdf",
        images: [face1],
        details: {
          problem:
            "Traditional essay scoring ignores HOW essays are written. We test whether keystroke dynamics predict writing quality.",
          data:
            "2,000+ essays with keystroke logs (event timings, text changes, word counts) from Kaggle.",
          approach: [
            "Engineered process features (pause length, bursts, insertions, deletions).",
            "Z‑tests comparing high vs low scoring groups.",
            "Multiple linear regression models with diagnostics (VIF, residuals, normality)."
          ],
          impact:
            "Explained ~50% of score variance; uninterrupted bursts and revision frequency were strongest predictors."
        },
        github:
          "https://github.com/kudos07/Data-Analysis---Linking-Writing-Processes-to-Writing-Quality"
      },
      {
  title: "Financial Fraud Detection",
  subtitle: "EDA → smart encoding → model zoo (PR‑AUC‑first)",
  tech: ["Python", "Pandas", "NumPy", "Scikit-learn", "Seaborn", "Matplotlib", "XGBoost", "RandomForest", "LightGBM", "CatBoost", "SHAP"],
  images: [img2],
  details: {
    problem:
      "Detect rare fraudulent transactions with high recall while keeping false positives low to protect revenue and customer trust.",
    data:
      "Synthetic mobile‑money transactions over ~30 days (CASH‑IN, CASH‑OUT, PAYMENT, TRANSFER, DEBIT), origin/destination IDs, balances/amounts; label: isFraud (0/1).",
    approach: [
      "EDA: schema, missingness, class imbalance; fraud rate by transaction type; key distribution plots.",
      "Encoding: One‑Hot for low‑cardinality categoricals (e.g., type); Frequency encoding for high‑cardinality IDs (nameOrig/nameDest). No plain LabelEncoder.",
      "Split: Time‑aware 80/20 split if timestamp available; otherwise stratified split to preserve class ratio.",
      "Models: Logistic Regression baseline → Random Forest → XGBoost (primary); optional LightGBM/CatBoost for comparison.",
      "Evaluation: PR‑AUC (primary) and ROC‑AUC; PR curves; cost‑optimal threshold selection; confusion matrix & classification report.",
      "Extras: Probability calibration (Platt), feature importance & SHAP explanations; artifacts saved (metrics.json, trained model)."
    ],
    impact:
      "PR‑AUC up to ~0.966 with XGBoost; cost‑tuned operating threshold improved recall at low false‑positive cost; reproducible, portfolio‑ready notebook."
  },
  github: "https://github.com/kudos07/Fraud_detection/tree/main" // replace with your link
},
   {
  title: "Mall Customer Segmentation",
  subtitle: "EDA → preprocessing → KMeans (K=5 personas)",
  tech: ["Python", "Pandas", "NumPy", "Scikit-learn", "Matplotlib", "Seaborn"],
  images: [img1],
  details: {
    problem:
      "Identify distinct customer segments to support targeted marketing, loyalty programs, and personalized offers instead of treating all shoppers the same.",
    data:
      "Kaggle Mall Customers dataset (200 records): Gender, Age, Annual Income (k$), Spending Score (1–100).",
    approach: [
      "EDA: distributions of age, income, spending; scatter plots of income vs. spending score.",
      "Preprocessing: Dropped CustomerID; encoded Gender; scaled numeric features.",
      "Clustering: Ran KMeans across K=2–10; used Elbow + Silhouette methods to evaluate.",
      "Selected K=5 for interpretability; projected clusters via PCA scatter plot.",
      "Profiles: Derived 5 personas (Premium Spenders, Budget-Conscious Older Adults, Young Value Seekers, Affluent but Reserved, Mid-tier Regulars)."
    ],
    impact:
      "Delivered 5 interpretable customer personas with actionable strategies (VIP perks, student discounts, upselling, essentials, bundles), enabling data-driven segmentation and marketing."
  },
  github: "https://github.com/kudos07/Mall_Customers_dataset" // replace with your link
},
{
  title: "Marketing Campaign Effectiveness (Causal Inference)",
  subtitle: "PSM → covariate balance → ATT estimation",
  tech: ["Python", "Pandas", "NumPy", "Scikit-learn", "Matplotlib", "Statsmodels"],
  images: [img2],
  details: {
    problem:
      "Determine whether a marketing campaign causally increased purchases or if differences were driven by demographics and confounding factors.",
    data:
      "Kaggle Marketing Campaign dataset (~2,240 customers): demographics (Age, Income, Education, Marital Status), product spending, and campaign response (Response).",
    approach: [
      "Engineered treatment (`Treated`=1 if Response=1) and outcomes (`NumStorePurchases`, `TotalSpend`).",
      "Applied Propensity Score Matching (nearest-neighbor, caliper=0.05) to balance treatment vs control on covariates.",
      "Checked covariate balance with Standardized Mean Differences and Love plots.",
      "Estimated ATT (Average Treatment Effect on the Treated) with 95% confidence intervals.",
      "Compared in-store purchases vs. total spending; visualized distributions and diagnostic plots."
    ],
    impact:
      "Found campaign significantly reduced in-store purchases (~1.8 fewer per treated customer, CI [–2.5, –1.0]); no statistically significant lift in total spend. Insights guided redesign of campaign targeting and messaging."
  },
  github: "https://github.com/kudos07/Marketing-PsM" // replace with your link
},
{
  title: "Kaggle Playground S5E1: Forecasting sticker sales",
  subtitle: "Time features → encoding → LightGBM (MAPE‑first)",
  tech: ["Python", "Pandas", "NumPy", "Scikit‑learn", "LightGBM", "Optuna", "Matplotlib", "Seaborn"],
  images: [img3],
  details: {
    problem:
      "Predict the number of units sold (`num_sold`) from structured tabular data; optimize for business‑friendly error (MAPE) to support demand planning.",
    data:
      "Kaggle Playground Series (Season 5, Episode 1): train/test with `num_sold` target and categorical/date fields; includes `sample_submission.csv`.",
    approach: [
      "EDA: target distribution and calendar effects; category frequencies; leakage checks.",
      "Feature engineering: parsed dates → year/month/day‑of‑week; optional week/quarter; interaction terms as needed.",
      "Encoding: LabelEncoding for categoricals used by tree models; kept numerics raw.",
      "Validation: 5‑fold K‑Fold cross‑validation (shuffle, seed=42) with MAPE as the primary score.",
      "Model: LightGBM Regressor as the main learner; hyperparameter tuning via Optuna; early stopping on validation folds.",
      "Evaluation: MAPE (primary), plus MAE/RMSE for sanity; tracked fold scores and variance; generated test predictions and submission CSV."
    ],
    impact:
      "Built a leaderboard‑ready pipeline with consistent cross‑validated MAPE and robust generalization. The workflow (time features + LightGBM + Optuna + K‑Fold) is reusable for retail demand planning and other tabular forecasting/regression tasks."
  },
  github: "https://github.com/kudos07/Kaggle-Playground-series/tree/main/S5-E1/Code" // replace with your link
}

    ]
  }
,
  {
    name: "Deep Learning",
    heroImg: dlHero,
    heroOverlayText: "Vision • RL • Representation Learning",
    projects: [
      {
        title: "OptiMorphic-Precision-Vision Framework",
        subtitle: "Encoder–decoder U-Net → supervised image restoration",
  tech : ["Python", "TensorFlow", "Keras", "NumPy", "TensorBoard"],
  images: [img5], // keep covers empty if you don't want a cover
  github: "https://github.com/kudos07/OptiMorphic-Precision-Vision-OPV-Framework",
  details: {
problem:
  "Restore degraded images with a supervised U-Net image-to-image model that minimizes MAE and preserves structure (edges, textures).",
data:
  "Paired input→target images (e.g., noisy→clean or blurred→sharp). Images are normalized to [0,1], optionally patchified, and fed as (input, target) batches for train/val.",
    approach: [
  "Framed as paired image-to-image translation (input → target). Inputs normalized to [0,1] for training and denormalized back to 8-bit for previews/exports.",
  "Implemented a U-Net in TensorFlow/Keras: encoder–decoder with skip connections, ReLU activations, MaxPooling down, UpSampling up, and a 3-channel output head; deeper blocks use Dropout for regularization. Tiny baselines (`identity`, `simplest`) included for ablations.",
  "Packaged training into a `Model_Train` class that builds the model, sets up an Adam optimizer with exponential-decay learning rate (floored at a minimum LR), TensorBoard writer, and a CheckpointManager for robust recovery.",
  "Compiled the training step with `@tf.function` (graph mode). Forward pass → compute L1 (MAE) loss against the paired target → backprop with GradientTape → apply gradients to generator weights.",
  "Logged both scalars and images to TensorBoard at intervals: loss curves plus visual triptychs `(input | prediction | target)` and quick concatenations for qualitative QA.",
  "Persisted state with `tf.train.Checkpoint` (model, optimizer, global step) and rolling `CheckpointManager` (keep=3). Exposed a `.save()` helper for explicit snapshots.",
  "Dataloading via an iterator yielding `(paired_input, paired_target)` batches; `train_step` returns a concise log string and preview images, making it easy to surface progress in a UI.",
  "Architecture is configurable (e.g., base filters via `unet_16()`), allowing depth/width adjustments without changing the training loop."
],
    impact:
    "U-Net beat identity and simplest baselines with lower MAE and higher PSNR/SSIM; triptych previews show sharper edges and fewer artifacts; fully reproducible with TensorBoard logs and checkpoints."

      }
    },
      {
  title: "RL-based Book Recommendation",
  subtitle: "Custom Gym env + PPO agent → sequential recsys",
  tech: ["Python", "PyTorch", "Stable-Baselines3 (PPO)", "Gymnasium", "Matplotlib", "NumPy"],
  images: [img4], // keep covers empty if you don't want a cover
  github: "https://github.com/kudos07/RL-Book-Recommender",
  details: {
    problem:
      "Model book recommendation as sequential decision-making to maximize long‑term engagement.",
    data:
      "Goodbooks‑10k; env encodes user history as state, action = book, reward = engagement hit.",
    approach: [
  "Framed recommendation as an MDP. Each episode simulates a user session. State captures recent interactions and simple user features. Action is picking the next book to recommend. Reward is 1 if the user would accept the book in the held-out positives, else 0.",
  "Prepared data from Goodbooks-10k into session-like sequences. Built per-user positives for evaluation and ensured no leakage by separating train and eval interactions.",
  "Built a Gymnasium environment (goodbooks_env). reset() initializes a session and exposes the first state. step(action) records the choice, assigns reward using ground truth positives, advances the session window, and prevents repeats. info['is_hit'] flags true positives for clean metric tracking.",
  "Kept the action space tractable by exposing a candidate slate at each step (for example popularity or simple similarity). The env maps the chosen index back to a book id and enforces that already shown items are masked.",
  "Trained a policy with PPO using on-policy rollouts collected from the simulator. The agent learns a stochastic policy π(a|s) and a value baseline V(s). Clipped policy updates keep learning stable while GAE supplies low-variance advantages.",
  "Established simple non-RL baselines. Random samples uniformly from the slate. Popularity always shows the most common unseen items. These baselines anchor expected performance.",
  "Evaluated offline on held-out sessions. Computed CTR@k and NDCG@k per step and per episode using the env’s info flags. Aggregated over many episodes to reduce variance and plotted the distributions.",
  "Logged training curves and episode stats with Monitor. Saved artifacts (trained policy, logs, result plots) for reproducibility. Seeding ensures runs are repeatable."
],
    impact:
      "PPO beat random on CTR@1 (~3×) and NDCG@1; reproducible pipeline.",
    resultsCaption: "CTR@k_episode and NDCG@k across models.",
    resultsImages: [img1,img3]
  },
}
]
  },
  {
    name: "LLMs & Generative AI",
    heroImg: llmHero,
    heroOverlayText: "RAG • Agents • Multimodal Generation",
    projects: [
      {
        title: "Meeting → Storyboard Generator",
        subtitle: "Turning meeting audio into summaries and storyboard images",
        tech: [
          "React", "Vite", "Tailwind", "FastAPI", "FFmpeg",
          "Whisper.cpp", "Ollama", "LLaMA 3", "SDXL", "PyTorch",
          "Redis", "PostgreSQL", "S3/MinIO"
        ],
        images: [stry],                   // card cover
        architectureImg: arch,           // full architecture diagram
        reportUrl: "/reports/project_report.pdf",
        details: {
          problem:
            "Meeting transcripts are often long and difficult to digest, making it hard for teams to recall key decisions and action items.",
          data:
            "5–60 minute meeting audio recordings, preprocessed with FFmpeg (resampling, chunking) and converted into transcripts, summaries, and scene beats.",
          approach: [
            "Frontend in React + Vite + Tailwind for audio uploads, progress tracking, and report viewing.",
            "Backend with FastAPI (single service) orchestrating steps and saving artifacts to disk (/public) or S3; optional Redis/PostgreSQL for job/state metadata.",
            "Pipeline: FFmpeg preprocess → Whisper.cpp (ASR) → Ollama (LLaMA 3) for summarization & scene extraction → SDXL (PyTorch) for storyboard frames.",
            "Final reports combine transcripts, executive summaries, decisions/action items, and a storyboard grid into HTML/PDF."
          ],
          impact:
            "Reduced processing time from ~2 minutes to ~40 seconds; delivered clear, shareable summaries and visual storyboards that improved recall and decision‑tracking.",
          architecture: [
            "React/Vite uploads → FastAPI presigns & orchestrates",
            "FFmpeg normalize/chunk → Whisper.cpp (ASR)",
            "Ollama (LLaMA 3) → summary + scene beats",
            "SDXL (PyTorch) → storyboard frames → HTML/PDF report (Disk/S3)"
          ]
        },
        github: "https://github.com/kudos07/EchoFrames",
      },
      {
  title: "Mental Health Chatbot",
  subtitle: "AI-powered support with RAG + Gemini for empathetic guidance",
  tech: [
    "Streamlit", "FastAPI", "FAISS", "Sentence Transformers",
    "Google Gemini (Generative AI)", "Fernet Encryption", "Python"
  ],
  images: [wall],                // card cover
  architectureImg: mhcb,         // full architecture diagram
  reportUrl: "/reports/final_report.pdf",
  details: {
    problem:
      "Many mental health chatbots provide generic, one-size-fits-all replies and neglect data privacy, leaving users without personalized or secure support.",
    data:
      "Counseling Q&A datasets and FAQ pairs embedded with Sentence Transformers, indexed in FAISS for retrieval. User interactions stored securely in encrypted memory.",
    approach: [
      "Frontend built with Streamlit for intuitive, real-time chat interface.",
      "Chatbot logic orchestrates retrieval + generation pipeline and maintains context across turns.",
      "Pipeline: User query → Sentence Transformers embeddings → FAISS k-NN retrieval → Constructed RAG prompt (user profile + past concerns + few-shot examples) → Gemini LLM response.",
      "Integrated FAQ fallback for common queries; encrypted user memory (Fernet) for personalization and privacy."
    ],
    impact:
      "Delivered empathetic, context-aware responses tailored to user profile and previous concerns; ensured HIPAA-level privacy compliance; achieved fast, relevant retrieval with FAISS while enabling user-controlled data deletion.",
    architecture: [
      "Streamlit UI → Chatbot logic",
      "Embeddings (Sentence Transformers) → FAISS retriever",
      "Retrieved context + few-shot examples → Gemini LLM",
      "Encrypted user memory (Fernet) → personalization",
      "Final empathetic response → Streamlit output"
    ]
  },
  github: "https://github.com/yourusername/mental_health_chatbot",
}
    ]
  },
  {
    name: "NLP",
    heroImg: nlpHero,
    heroOverlayText: "NER • Sentiment • Topic/Intent",
    projects: [
      {
  title: "AG News: TF-IDF vs TextCNN vs TextCNN+",
  subtitle: "Keyword baseline vs CNNs → 4-class news classification",
  tech: ["Python", "PyTorch", "TorchText", "scikit-learn", "Matplotlib", "pandas", "tqdm"],
  images: [img8], // keep empty if you don't want a cover
  github: "https://github.com/yourrepo/textcnn-agnews", // replace with your link
  details: {
    problem:
      "Classify news headlines into four categories and compare a strong TF-IDF+LogReg baseline against compact and improved CNN architectures.",
    data:
      "AG News dataset via TorchText; tokenization with basic_english; custom vocab (<pad>, <unk>), sequences padded/truncated to fixed length; standard train/val/test splits.",
    approach: [
      "Baseline: TF-IDF (1–2 grams) + Logistic Regression in scikit-learn for a strong keyword model.",
      "Neural: TextCNN (embedding → parallel Conv1d k∈{3,4,5} → ReLU → max-over-time pool → concat → dropout → linear).",
      "Improved (TextCNN+): wider filters, longer input windows, dropout regularization, AdamW + OneCycleLR, optional GloVe init, and early stopping on validation.",
      "Training scripts produce metrics JSONs, per-class reports, confusion matrices, prediction CSVs, and comparison plots for side-by-side review."
    ],
    impact:
      "TextCNN+ surpassed the keyword baseline (Accuracy 0.893, Macro-F1 0.892) vs TF-IDF+LR (0.865/0.864) and the original TextCNN (0.797/0.797). Reproducible pipelines with saved metrics, plots, and error analyses.",
    resultsCaption:
      "3-way comparison (overall metrics + confusion matrices) and per-class diagnostics.",
    resultsImages: [img6,img7]
  }
},
      {
        title: "Toxicity / Sentiment Moderator",
        subtitle: "DistilBERT on Civil Comments",
        bullets: ["ROC‑AUC 0.96; macro‑F1 0.81; calibrated thresholds.", "Streaming inference API + live UI."],
        tech: ["Transformers", "HF Datasets", "FastAPI"],
        images: [],
        year: 2024
      }
    ]
  }
];

/* ===================== Component ===================== */
export default function Projects() {
  const [openCategory, setOpenCategory] = useState<number | null>(null);
  const [openProject, setOpenProject] = useState<{ cat: number; idx: number } | null>(null);

  const current = useMemo(
    () => (openProject ? CATEGORIES[openProject.cat].projects[openProject.idx] : null),
    [openProject]
  );

  return (
    <section id="projects" className="min-h-screen pt-32 px-6">
      <div className="mx-auto max-w-6xl">
        <h2 className="text-4xl font-extrabold mb-8 text-center">
          <span className="bg-clip-text text-transparent bg-gradient-to-r from-teal-300 via-cyan-200 to-indigo-300 drop-shadow text-glow">
            Projects
          </span>
        </h2>

        {/* ===== Level 1: FOUR BIG HERO BOXES ===== */}
        <div className="grid md:grid-cols-2 gap-6">
          {CATEGORIES.map((cat, i) => (
            <button
              key={i}
              onClick={() => setOpenCategory(i)}
              className="relative group rounded-3xl overflow-hidden h-44 md:h-56 text-left
                         ring-1 ring-white/10 bg-gray-900/70 shadow-lg hover:shadow-teal-400/20
                         transition transform hover:-translate-y-1 card-glow"
            >
              {cat.heroImg ? (
                <Image
                  src={cat.heroImg}
                  alt={`${cat.name} background`}
                  fill
                  sizes="(max-width: 768px) 100vw, 50vw"
                  className="object-cover opacity-40 group-hover:opacity-50 transition"
                  priority={i < 2}
                />
              ) : (
                <div className="absolute inset-0 bg-[radial-gradient(circle_at_10%_10%,rgba(45,212,191,.2),transparent_40%),radial-gradient(circle_at_90%_90%,rgba(59,130,246,.18),transparent_45%)]" />
              )}
              <div className="absolute inset-0 bg-gradient-to-br from-black/50 via-black/30 to-black/50" />
              <div className="relative z-10 p-6">
                <h3 className="text-2xl font-semibold text-white text-glow">{cat.name}</h3>
                {cat.heroOverlayText && (
                  <p className="mt-1 text-sm text-gray-300">{cat.heroOverlayText}</p>
                )}
                <p className="mt-4 text-xs text-teal-300 opacity-90">
                  Click to explore {cat.projects.length} project{cat.projects.length > 1 ? "s" : ""}
                </p>
              </div>
            </button>
          ))}
        </div>

        {/* ===== Level 2: FULLSCREEN CATEGORY SHEET ===== */}
        {openCategory !== null && (
          <div className="fixed inset-0 z-50">
            <div
              className="absolute inset-0 bg-black/70 backdrop-blur-lg"
              onClick={() => setOpenCategory(null)}
            />
            <div className="absolute inset-2 md:inset-8 rounded-3xl overflow-hidden ring-1 ring-white/10 bg-[rgba(10,15,20,0.92)]
                            shadow-[0_40px_120px_-20px_rgba(0,0,0,0.8)]">
              <div className="relative h-full p-6 md:p-10 overflow-y-auto">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-3xl md:text-4xl font-extrabold tracking-tight text-white text-glow">
                    {CATEGORIES[openCategory].name}
                  </h3>
                  <button
                    onClick={() => setOpenCategory(null)}
                    className="rounded-full p-3 ring-1 ring-white/10 bg-white/5 hover:bg-white/10 transition"
                    aria-label="Close category"
                  >
                    <FaTimes />
                  </button>
                </div>

                <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6">
                  {CATEGORIES[openCategory].projects.map((p, idx) => (
                    <div
                      key={idx}
                      onClick={() => setOpenProject({ cat: openCategory, idx })}
                      className="cursor-pointer group relative rounded-2xl bg-gray-900/70 p-5 ring-1 ring-white/10 hover:ring-teal-400/30
                                 shadow hover:shadow-teal-400/20 transition transform hover:-translate-y-1 card-glow"
                    >
                      <div className="relative w-full h-36 mb-4">
                        <Image
                          src={p.images?.[0] || PLACEHOLDER}
                          alt={`${p.title} preview`}
                          fill
                          sizes="(max-width: 1024px) 50vw, 33vw"
                          className="object-cover rounded-lg opacity-80 group-hover:opacity-100 transition"
                        />
                      </div>
                      <h4 className="text-lg font-semibold text-white text-glow">{p.title}</h4>
                      {p.subtitle && <p className="text-gray-400 text-sm">{p.subtitle}</p>}
                      {p.year && <p className="text-xs text-teal-300 mt-1">{p.year}</p>}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ===== Level 3: PROJECT MODAL ===== */}
        {current && (
          <div className="fixed inset-0 z-[60]">
            <div
              className="absolute inset-0 bg-black/70 backdrop-blur-lg"
              onClick={() => setOpenProject(null)}
            />
            <div className="absolute inset-3 md:inset-10 rounded-3xl overflow-hidden ring-1 ring-white/10 bg-[rgba(10,15,20,0.92)]
                            shadow-[0_40px_120px_-20px_rgba(0,0,0,0.8)]">
              <div className="relative h-full p-6 md:p-10 overflow-y-auto">
                {/* Header */}
                <div className="flex items-start justify-between gap-6 mb-6">
                  <div>
                    <h3 className="text-3xl md:text-4xl font-extrabold leading-tight tracking-tight text-white">
                      <span className="bg-clip-text text-transparent bg-gradient-to-r from-teal-300 via-cyan-200 to-indigo-300 drop-shadow">
                        {current.title}
                      </span>
                    </h3>
                    {current.subtitle && <p className="text-gray-300">{current.subtitle}</p>}
                    {current.tech?.length ? (
                      <div className="mt-3 flex flex-wrap gap-2">
                        {current.tech.map((t, i) => (
                          <span
                            key={i}
                            className="px-3 py-1 text-xs rounded-full bg-teal-400/10 text-teal-200 ring-1 ring-teal-400/30"
                          >
                            {t}
                          </span>
                        ))}
                      </div>
                    ) : null}
                  </div>
                  <button
                    onClick={() => setOpenProject(null)}
                    className="rounded-full p-3 ring-1 ring-white/10 bg-white/5 hover:bg-white/10 transition"
                    aria-label="Close project"
                  >
                    <FaTimes className="text-xl" />
                  </button>
                </div>

                <div className="space-y-10">
                  {/* Case-study section */}
                  {current.details && (
                    <section>
                      <h4 className="text-xl font-bold text-white text-glow mb-4">Case Study</h4>

                      <div className="grid md:grid-cols-2 gap-5">
                        {current.details.problem && (
                          <div className="relative rounded-2xl p-5 bg-white/5 ring-1 ring-white/10 card-glow">
                            <div className="glow-halo"></div>
                            <div className="text-sm font-semibold uppercase tracking-wider text-teal-300/90 mb-2">
                              Problem
                            </div>
                            <p className="text-gray-100 leading-relaxed drop-shadow">
                              {current.details.problem}
                            </p>
                          </div>
                        )}
                        {current.details.data && (
                          <div className="relative rounded-2xl p-5 bg-white/5 ring-1 ring-white/10 card-glow">
                            <div className="glow-halo"></div>
                            <div className="text-sm font-semibold uppercase tracking-wider text-teal-300/90 mb-2">
                              Data
                            </div>
                            <p className="text-gray-100 leading-relaxed drop-shadow">
                              {current.details.data}
                            </p>
                          </div>
                        )}
                        {current.details.approach && (
                          <div className="relative rounded-2xl p-5 bg-white/5 ring-1 ring-white/10 card-glow md:col-span-2">
                            <div className="glow-halo"></div>
                            <div className="text-sm font-semibold uppercase tracking-wider text-teal-300/90 mb-2">
                              Approach
                            </div>
                            {Array.isArray(current.details.approach) ? (
                              <ul className="space-y-2 text-gray-100">
                                {current.details.approach.map((a, i) => (
                                  <li key={i} className="relative pl-6 drop-shadow">
                                    <span className="absolute bullet-dot" />
                                    {a}
                                  </li>
                                ))}
                              </ul>
                            ) : (
                              <p className="text-gray-100 drop-shadow">{current.details.approach}</p>
                            )}
                          </div>
                        )}
                        {current.details.impact && (
                          <div className="relative rounded-2xl p-5 bg-white/5 ring-1 ring-white/10 card-glow md:col-span-2">
                            <div className="glow-halo"></div>
                            <div className="text-sm font-semibold uppercase tracking-wider text-teal-300/90 mb-2">
                              Impact
                            </div>
                            <p className="text-gray-100 drop-shadow">{current.details.impact}</p>
                          </div>
                        )}
                      </div>
                    </section>
                  )}

                  {/* Optional highlights */}
                  {current.bullets?.length ? (
                    <section>
                      <h4 className="text-xl font-bold text-white text-glow mb-3">Highlights</h4>
                      <ul className="space-y-2 text-gray-100">
                        {current.bullets.map((b, i) => (
                          <li key={i} className="relative pl-6 drop-shadow">
                            <span className="absolute bullet-dot" />
                            {b}
                          </li>
                        ))}
                      </ul>
                    </section>
                  ) : null}

                  {/* Gallery (skip the first image because it's used as cover) */}
                  {(() => {
                    const galleryImgs = Array.isArray(current.images) ? current.images.slice(1) : [];
                    return galleryImgs.length ? (
                      <section>
                        <h4 className="text-xl font-bold text-white text-glow mb-3">Gallery</h4>
                        <div className="grid sm:grid-cols-2 gap-4">
                          {galleryImgs.map((src, i) => (
                            <div key={i} className="relative w-full max-h-64 h-64">
                              <Image
                                src={src}
                                alt={`gallery-${i + 1}`}
                                fill
                                sizes="(max-width: 1024px) 50vw, 33vw"
                                className="object-cover rounded-xl border border-white/10"
                              />
                            </div>
                          ))}
                        </div>
                      </section>
                    ) : null;
                  })()}

                  {/* Architecture (bullets + diagram) */}
                  {(current.details?.architecture?.length || current.architectureImg) ? (
                    <section>
                      <h4 className="text-xl font-bold text-white text-glow mb-3">Architecture</h4>

                      {current.details?.architecture?.length ? (
                        <ol className="list-decimal pl-5 mb-4 space-y-1 text-gray-100">
                          {current.details.architecture.map((line, i) => (
                            <li key={i}>{line}</li>
                          ))}
                        </ol>
                      ) : null}

                      {current.architectureImg ? (
                        <div className="relative w-full max-h-[420px] h-[420px]">
                          <Image
                            src={current.architectureImg}
                            alt={`${current.title} architecture`}
                            fill
                            sizes="100vw"
                            className="object-contain rounded-xl border border-white/10 bg-black/10"
                          />
                        </div>
                      ) : null}
                    </section>
                  ) : null}

                  


                  {/* Report (download link) */}
                  {current?.reportUrl ? (
                    <section>
                      <h4 className="text-xl font-bold text-white text-glow mb-3">Report</h4>
                      <a
                        href={`${current.reportUrl}${current.reportUrl.includes("?") ? "&" : "?"}download=1`}
                        download={fileNameFromUrl(current.reportUrl)}
                        className="inline-flex items-center gap-2 mt-1 px-4 py-2 rounded-md bg-white/10 hover:bg-white/20 ring-1 ring-white/15"
                        aria-label={`Download ${fileNameFromUrl(current.reportUrl)}`}
                      >
                        <FaDownload />
                        Download PDF
                      </a>
                    </section>
                  ) : null}

                  {/* Results (images + download) */}
{current?.details?.resultsImages?.length ? (
  <section className="mt-6">
    <h4 className="text-xl font-bold text-white text-glow mb-3">Results</h4>

    {current.details.resultsCaption ? (
      <p className="text-sm text-white/70 mb-4">
        {current.details.resultsCaption}
      </p>
    ) : null}

    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
  {current.details.resultsImages.map((imgSrc: string | StaticImageData, i: number) => {
    const src = typeof imgSrc === "string" ? imgSrc : imgSrc.src; // no 'any' needed
    const fname = fileNameFromUrl(src) || `result_${i + 1}.png`;
        return (
          <figure
            key={i}
            className="rounded-2xl overflow-hidden ring-1 ring-white/10 bg-white/5"
          >
            <Image
              src={src}
              alt={`Result ${i + 1}`}
              width={1600}
              height={900}
              className="w-full h-auto"
              priority={i === 0}
            />
            <figcaption className="flex items-center justify-between px-3 py-2 bg-black/30">
              <span className="text-xs text-white/70">Result {i + 1}</span>
              <a
                href={`${src}${src.includes("?") ? "&" : "?"}download=1`}
                download={fname}
                className="inline-flex items-center gap-2 px-3 py-1 rounded-md bg-white/10 hover:bg-white/20 ring-1 ring-white/15 text-sm"
                aria-label={`Download ${fname}`}
                title={`Download ${fname}`}
              >
                <FaDownload />
                Download
              </a>
            </figcaption>
          </figure>
        );
      })}
    </div>
  </section>
) : null}


                  {/* Links */}
                  {(current.github || current.demo) && (
                    <section>
                      <h4 className="text-xl font-bold text-white text-glow mb-3">Links</h4>
                      <div className="flex gap-4">
                        {current.github && (
                          <a
                            href={current.github}
                            target="_blank"
                            className="flex items-center gap-2 text-teal-300 hover:underline"
                          >
                            <FaGithub /> GitHub
                          </a>
                        )}
                        {current.demo && (
                          <a
                            href={current.demo}
                            target="_blank"
                            className="flex items-center gap-2 text-teal-300 hover:underline"
                          >
                            <FaExternalLinkAlt /> Live Demo
                          </a>
                        )}
                      </div>
                    </section>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
