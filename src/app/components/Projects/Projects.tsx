"use client";
import { useEffect, useMemo, useState } from "react";
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
import wall from "./images/chatbot-wall.png"
import img2 from "./images/fraud.png"
import img1 from "./images/CTR_at_k_episode.png"
import img3 from "./images/NDCG_at_k.png"
import img4 from "./images/rlpro.png"
import img5 from "./images/opv.png"
import img6 from "./images/img1.png"
import img7 from "./images/img2.png"
import img8 from "./images/img3.png"
import img9 from "./images/elbow.png"
import img10 from "./images/plot.png"
import img11 from "./images/forecast.png"
import img12 from "./images/quora.png"
import grantscoutHero from "./images/grantscout-ai.png"
import letterfitHero from "./images/letterfit-ai.png"
import minijudgeHero from "./images/minijudge.png"
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
            "1. Engineered process features (pause length, bursts, insertions, deletions).",
            "2. Z‑tests comparing high vs low scoring groups.",
            "3. Multiple linear regression models with diagnostics (VIF, residuals, normality)."
          ],
          impact:
            "Explained ~50% of score variance; uninterrupted bursts and revision frequency were strongest predictors."
        },
        github:
          "https://github.com/kudos07/Data-Analysis---Linking-Writing-Processes-to-Writing-Quality"
      },
      {
  title: "Financial Fraud Detection",
  subtitle: "Detecting fraudulent transactions with ML (PR-AUC focus)",
  tech: ["Python", "Pandas", "NumPy", "Scikit-learn", "Seaborn", "Matplotlib", "XGBoost", "RandomForest", "LightGBM", "CatBoost", "SHAP"],
  images: [img2],
  details: {
    problem:
      "Detect rare fraudulent transactions with high recall while keeping false positives low to protect revenue and customer trust.",
    data:
      "Synthetic mobile‑money transactions over ~30 days (CASH‑IN, CASH‑OUT, PAYMENT, TRANSFER, DEBIT), origin/destination IDs, balances/amounts; label: isFraud (0/1).",
    approach: [
  "1. EDA: check schema, missing values, class imbalance; analyze fraud rate by transaction type; plot key distributions.",
  "2. Encoding: One-Hot for low-cardinality categoricals (e.g., type); Frequency encoding for high-cardinality IDs (nameOrig/nameDest).",
  "3. Split: Time-aware 80/20 split if timestamps available; otherwise stratified split to preserve class ratio.",
  "4. Models: Logistic Regression baseline → Random Forest → XGBoost (primary); optional LightGBM/CatBoost for comparison.",
  "5. Evaluation: PR-AUC (primary) and ROC-AUC; PR curves; cost-sensitive threshold tuning; confusion matrix & classification report.",
  "6. Extras (optional): probability calibration (Platt), feature importance & SHAP explanations; saved artifacts (metrics.json, trained model)."
],
    impact:
      "PR‑AUC up to ~0.966 with XGBoost; cost‑tuned operating threshold improved recall at low false‑positive cost; reproducible, portfolio‑ready notebook."
  },
  github: "https://github.com/kudos07/Fraud_detection/tree/main" // replace with your link
},
   {
  title: "Mall Customer Segmentation",
  subtitle: "Clustering mall customers into 5 personas with KMeans",
  tech: ["Python", "Pandas", "NumPy", "Scikit-learn", "Matplotlib", "Seaborn"],
  images: [img9],
  details: {
    problem:
      "Identify distinct customer segments to support targeted marketing, loyalty programs, and personalized offers instead of treating all shoppers the same.",
    data:
      "Kaggle Mall Customers dataset (200 records): Gender, Age, Annual Income (k$), Spending Score (1–100).",
    approach: [
      "1. EDA: distributions of age, income, spending; scatter plots of income vs. spending score.",
      "2. Preprocessing: Dropped CustomerID; encoded Gender; scaled numeric features.",
      "3. Clustering: Ran KMeans across K=2–10; used Elbow + Silhouette methods to evaluate.",
      "4. Selected K=5 for interpretability; projected clusters via PCA scatter plot.",
      "5. Profiles: Derived 5 personas (Premium Spenders, Budget-Conscious Older Adults, Young Value Seekers, Affluent but Reserved, Mid-tier Regulars)."
    ],
    impact:
      "Delivered 5 interpretable customer personas with actionable strategies (VIP perks, student discounts, upselling, essentials, bundles), enabling data-driven segmentation and marketing."
  },
  github: "https://github.com/kudos07/Mall_Customers_dataset" // replace with your link
},
{
  title: "Marketing Campaign Effectiveness (Causal Inference)",
  subtitle: "Measuring true campaign impact with Propensity Score Matching (PSM)",
  tech: ["Python", "Pandas", "NumPy", "Scikit-learn", "Matplotlib", "Statsmodels"],
  images: [img10],
  details: {
    problem:
      "Determine whether a marketing campaign causally increased purchases or if differences were driven by demographics and confounding factors.",
    data:
      "Kaggle Marketing Campaign dataset (~2,240 customers): demographics (Age, Income, Education, Marital Status), product spending, and campaign response (Response).",
    approach: [
      "1. Engineered treatment (`Treated`=1 if Response=1) and outcomes (`NumStorePurchases`, `TotalSpend`).",
      "2. Applied Propensity Score Matching (nearest-neighbor, caliper=0.05) to balance treatment vs control on covariates.",
      "3. Checked covariate balance with Standardized Mean Differences and Love plots.",
      "4. Estimated ATT (Average Treatment Effect on the Treated) with 95% confidence intervals.",
      "5. Compared in-store purchases vs. total spending; visualized distributions and diagnostic plots."
    ],
    impact:
      "Found campaign significantly reduced in-store purchases (~1.8 fewer per treated customer, CI [–2.5, –1.0]); no statistically significant lift in total spend. Insights guided redesign of campaign targeting and messaging."
  },
  github: "https://github.com/kudos07/Marketing-PsM" // replace with your link
},
{
  title: "Kaggle Playground S5E1: Forecasting sticker sales",
  subtitle: "Forecasting daily sticker sales with time-series ML (LightGBM)",
  tech: ["Python", "Pandas", "NumPy", "Scikit‑learn", "LightGBM", "Optuna", "Matplotlib", "Seaborn"],
  images: [img11],
  details: {
    problem:
      "Predict the number of units sold (`num_sold`) from structured tabular data; optimize for business‑friendly error (MAPE) to support demand planning.",
    data:
      "Kaggle Playground Series (Season 5, Episode 1): train/test with `num_sold` target and categorical/date fields; includes `sample_submission.csv`.",
    approach: [
      "1. EDA: target distribution and calendar effects; category frequencies; leakage checks.",
      "2. Feature engineering: parsed dates → year/month/day‑of‑week; optional week/quarter; interaction terms as needed.",
      "3. Encoding: LabelEncoding for categoricals used by tree models; kept numerics raw.",
      "4. Validation: 5‑fold K‑Fold cross‑validation (shuffle, seed=42) with MAPE as the primary score.",
      "5. Model: LightGBM Regressor as the main learner; hyperparameter tuning via Optuna; early stopping on validation folds.",
      "6. Evaluation: MAPE (primary), plus MAE/RMSE for sanity; tracked fold scores and variance; generated test predictions and submission CSV."
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
        subtitle: "Restoring degraded images with a U-Net model (supervised image-to-image)",
  tech : ["Python", "TensorFlow", "Keras", "NumPy", "TensorBoard"],
  images: [img5], // keep covers empty if you don't want a cover
  github: "https://github.com/kudos07/OptiMorphic-Precision-Vision-OPV-Framework",
  details: {
problem:
  "Restore degraded images with a supervised U-Net image-to-image model that minimizes MAE and preserves structure (edges, textures).",
data:
  "Paired input→target images (e.g., noisy→clean or blurred→sharp). Images are normalized to [0,1], optionally patchified, and fed as (input, target) batches for train/val.",
    approach: [
  "1. Treat image restoration as paired input→target translation; normalize inputs to [0,1] and denormalize for previews.",
  "2. Build a configurable U-Net in TensorFlow/Keras with skip connections, dropout, and simple baselines for comparison.",
  "3. Train with Adam (decaying LR) using MAE loss; compiled in graph mode with GradientTape for efficient backprop.",
  "4. Log loss curves and input|prediction|target samples in TensorBoard; save checkpoints for recovery and reproducibility."
],

    impact:
    "U-Net beat identity and simplest baselines with lower MAE and higher PSNR/SSIM; triptych previews show sharper edges and fewer artifacts; fully reproducible with TensorBoard logs and checkpoints."

      }
    },
      {
  title: "RL-based Book Recommendation",
  subtitle: "Recommending books with Reinforcement Learning (PPO agent on custom Gym env)",
  tech: ["Python", "PyTorch", "Stable-Baselines3 (PPO)", "Gymnasium", "Matplotlib", "NumPy"],
  images: [img4], // keep covers empty if you don't want a cover
  github: "https://github.com/kudos07/RL-Book-Recommender",
  details: {
    problem:
      "Model book recommendation as sequential decision-making to maximize long‑term engagement.",
    data:
      "Goodbooks‑10k; env encodes user history as state, action = book, reward = engagement hit.",
    approach: [
  "1. Frame it as a small-slate contextual bandit: one pick per step from a tiny candidate set; reward = 1 on hit; mask already-seen books.",
  "2. Build the slate simply (e.g., top-popular + a few embedding-similar to last read) to keep the action space small and stable.",
  "3. Train baselines first (random, ε-greedy over a simple score like popularity or dot-product), then swap in PPO on the same setup.",
  "4. Evaluate offline with fixed seeds: report CTR@1 and NDCG@1 over many steps; log curves and save artifacts for easy reproducibility."
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
        title: "MiniJudge",
        subtitle: "Can a 1.7B model become a reliable pairwise LLM judge? QLoRA + reliability eval, fully local",
        images: [minijudgeHero],
        tech: [
          "Python", "Qwen3", "QLoRA", "PEFT", "TRL",
          "PyTorch", "Hugging Face", "Chatbot Arena"
        ],
        details: {
          problem:
            "Paid LLM judges are expensive and opaque; small local judges often flip answers when A/B order swaps. Need measurable accuracy and reliability on consumer GPUs.",
          data:
            "LMSYS Chatbot Arena preference pairs, a synthetic bias suite, and optional JudgeBench — all evaluated fully locally with no paid APIs.",
          approach: [
            "1. Prompt-only baselines on Qwen3-0.6B and Qwen3-1.7B for pairwise A/B judging.",
            "2. QLoRA fine-tune of the 1.7B judge on Arena preferences (4-bit, ~6–8 GB VRAM).",
            "3. Reliability ablations: position swap + majority vote to measure consistency vs conflict.",
            "4. Bias-suite eval and a results dashboard comparing arena accuracy, F1, conflict rate, latency, and VRAM."
          ],
          impact:
            "Best Arena accuracy 60.8% with 1.7B QLoRA + swap (vs ~49.6% prompt-only); conflict rate cut from ~71% to ~39%, all on a local consumer GPU.",
          architecture: [
            "configs/ YAML → scripts pipeline (data → baseline → train → eval → bias)",
            "src/minijudge → data prep, judge, QLoRA train, metrics",
            "outputs/ experiment JSON → dashboard.html comparison table + charts",
            "results/final → committed Arena + bias snapshots"
          ]
        },
        github: "https://github.com/kudos07/minijudge",
      },
      {
        title: "GrantScout AI",
        subtitle: "Agentic grant discovery that ranks funding opportunities from your profile",
        images: [grantscoutHero],
        tech: [
          "Next.js", "React", "TypeScript", "Tailwind CSS",
          "Python", "Mistral AI", "Zod"
        ],
        details: {
          problem:
            "Finding grants, fellowships, and scholarships is fragmented across sources; applicants need a ranked shortlist with evidence, not a pile of search results.",
          data:
            "User mission brief (profile, location, status, interests) plus web sources discovered through multi-round search and page inspection.",
          approach: [
            "1. Next.js UI collects applicant profile, search signals, and agent settings.",
            "2. Python pipeline plans search strategy, runs multi-round web discovery, and opens candidate pages.",
            "3. Mistral extracts structured opportunity data (deadlines, requirements, eligibility) with evidence trails.",
            "4. Eligibility scoring and ranking produce a shortlist with checklists, draft answers, and JSON export."
          ],
          impact:
            "Delivers a goal-driven agent workflow—search, extract, score, recommend—with saved shortlists and exportable reports for faster grant applications.",
          architecture: [
            "Next.js UI → POST /api/run",
            "GrantScout pipeline → web search + page reading",
            "Mistral → structured extraction and reasoning",
            "Ranker → eligibility scoring → decision cards + export"
          ]
        },
        github: "https://github.com/kudos07/Grantscout-AI",
      },
      {
        title: "LetterFit AI",
        subtitle: "Tailored tech cover letters from resume, job description, and tone presets",
        images: [letterfitHero],
        tech: [
          "React", "Vite", "Tailwind CSS", "FastAPI",
          "Mistral AI", "PyMuPDF", "python-docx"
        ],
        details: {
          problem:
            "Generic cover letters miss role-specific evidence and tone; applicants need fast, ATS-aware drafts grounded in their actual resume.",
          data:
            "Uploaded resume (PDF/DOCX), pasted job description, optional company name for Wikipedia/web research, and tone/length preferences.",
          approach: [
            "1. React + Vite frontend for resume upload, JD input, style/length selection, and editable output.",
            "2. FastAPI backend extracts resume text, runs evidence selection, and calls Mistral for generation.",
            "3. Five tone presets (Professional, Qualifications, Hype, Mix, Bold) with quality analysis (ATS keywords, tone score).",
            "4. Paragraph-level regenerate, style compare, and DOCX/PDF export with browser localStorage for form state."
          ],
          impact:
            "Full-stack cover letter generator with company research, evidence-backed writing, quality scoring, and export—built as a stateless MVP on Mistral AI.",
          architecture: [
            "React/Vite UI → FastAPI API",
            "Resume parser (PyMuPDF, python-docx) → evidence selector",
            "Company research (Wikipedia + web) → Mistral generation",
            "Quality analysis → editable output → DOCX/PDF export"
          ]
        },
        github: "https://github.com/kudos07/LetterFit-AI",
      },
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
        details: {
          problem:
            "Meeting transcripts are often long and difficult to digest, making it hard for teams to recall key decisions and action items.",
          data:
            "5–60 minute meeting audio recordings, preprocessed with FFmpeg (resampling, chunking) and converted into transcripts, summaries, and scene beats.",
          approach: [
            "1. Frontend in React + Vite + Tailwind for audio uploads, progress tracking, and report viewing.",
            "2. Backend with FastAPI (single service) orchestrating steps and saving artifacts to disk (/public) or S3; optional Redis/PostgreSQL for job/state metadata.",
            "3. Pipeline: FFmpeg preprocess → Whisper.cpp (ASR) → Ollama (LLaMA 3) for summarization & scene extraction → SDXL (PyTorch) for storyboard frames.",
            "4. Final reports combine transcripts, executive summaries, decisions/action items, and a storyboard grid into HTML/PDF."
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
  subtitle: "AI-powered empathetic support using RAG + Gemini",
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
      "1. Frontend built with Streamlit for intuitive, real-time chat interface.",
      "2. Chatbot logic orchestrates retrieval + generation pipeline and maintains context across turns.",
      "3. Pipeline: User query → Sentence Transformers embeddings → FAISS k-NN retrieval → Constructed RAG prompt (user profile + past concerns + few-shot examples) → Gemini LLM response.",
      "4. Integrated FAQ fallback for common queries; encrypted user memory (Fernet) for personalization and privacy."
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
  github: "https://github.com/kudos07/MentalHealth-Chatbot",
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
  subtitle: "Can a simple keyword model match deep learning for classifying news headlines?",
  tech: ["Python", "PyTorch", "TorchText", "scikit-learn", "Matplotlib", "pandas", "tqdm"],
  images: [img8], // keep empty if you don't want a cover
  github: "https://github.com/kudos07/AG-News-TF-IDF-vs-TextCNN-vs-TextCNN", // replace with your link
  details: {
    problem:
      "Classify news headlines into four categories and compare a strong TF-IDF+LogReg baseline against compact and improved CNN architectures.",
    data:
      "AG News dataset via TorchText; tokenization with basic_english; custom vocab (<pad>, <unk>), sequences padded/truncated to fixed length; standard train/val/test splits.",
    approach: [
      "1. Baseline: TF-IDF (1–2 grams) + Logistic Regression in scikit-learn for a strong keyword model.",
      "2. Neural: TextCNN (embedding → parallel Conv1d k∈{3,4,5} → ReLU → max-over-time pool → concat → dropout → linear).",
      "3. Improved (TextCNN+): wider filters, longer input windows, dropout regularization, AdamW + OneCycleLR, optional GloVe init, and early stopping on validation.",
      "4. Training scripts produce metrics JSONs, per-class reports, confusion matrices, prediction CSVs, and comparison plots for side-by-side review."
    ],
    impact:
      "TextCNN+ surpassed the keyword baseline (Accuracy 0.893, Macro-F1 0.892) vs TF-IDF+LR (0.865/0.864) and the original TextCNN (0.797/0.797). Reproducible pipelines with saved metrics, plots, and error analyses.",
    resultsCaption:
      "3-way comparison (overall metrics + confusion matrices) and per-class diagnostics.",
    resultsImages: [img6,img7]
  }
},
      {
  title: "Quora Duplicate Question Detection",
  subtitle: "Do these two questions mean the same thing?",
  tech: ["Python", "PyTorch", "spaCy", "scikit-learn", "Matplotlib", "pandas", "seaborn", "joblib"],
  images: [img12], // no cover image for now
  github: "https://github.com/kudos07/Duplicate-Question-Detection-on-Quora-Pairs/tree/main", // replace with your actual repo
  details: {
    problem:
      "Detect whether two questions from Quora mean the same thing using a supervised binary classification setup. Compare a traditional ML baseline vs deep learning with sequence models.",
    data:
      "Quora Question Pairs (QQP) dataset (~400k labeled pairs). Cleaned and split with stratified sampling. Used raw question text and binary labels. Preprocessed with token normalization and spaCy embeddings; padded sequences for BiLSTM.",
    approach: [
      "1. EDA: Label distribution, question length histograms, Jaccard lexical overlap, exact duplicates, and visualization of class imbalance.",
      "2. Baseline: Averaged spaCy embeddings per question → engineered pairwise features (cosine, L1, Hadamard, etc.) → Logistic Regression.",
      "3. Siamese BiLSTM: Shared encoder over tokenized questions → final hidden states → comparison via abs-diff + elementwise product → MLP classifier.",
      "4. Model outputs: Metrics JSONs, ROC/PR/confusion plots, calibration curve, predictions CSVs, saved model artifacts (joblib, .pt, vocab, etc.)."
    ],
    impact:
      "Demonstrated ~19% accuracy lift and major ROC/PR-AUC gains from the deep BiLSTM approach (~0.82 accuracy) over the feature-based baseline (~0.63). Encoded deeper semantics via sequence modeling and aligned evaluation artifacts for clarity.",
  }
}

    ]
  }
];

/* ===================== Component ===================== */
export default function Projects() {
  const [activeIndex, setActiveIndex] = useState(0);
  const [openProject, setOpenProject] = useState<{ cat: number; idx: number } | null>(null);

  const rotatingProjects = useMemo(
    () =>
      CATEGORIES.flatMap((cat, catIndex) =>
        cat.projects.map((project, idx) => ({
          cat: catIndex,
          idx,
          category: cat.name,
          project,
        }))
      ),
    []
  );

  useEffect(() => {
    if (!rotatingProjects.length) return;
    const timer = setInterval(() => {
      setActiveIndex((prev) => (prev + 1) % rotatingProjects.length);
    }, 2300);
    return () => clearInterval(timer);
  }, [rotatingProjects.length]);

  const goPrev = () => {
    setActiveIndex((prev) =>
      prev === 0 ? rotatingProjects.length - 1 : prev - 1
    );
  };

  const goNext = () => {
    setActiveIndex((prev) => (prev + 1) % rotatingProjects.length);
  };

  const visibleProjects = useMemo(() => {
    if (!rotatingProjects.length) return [];
    return Array.from({ length: Math.min(4, rotatingProjects.length) }, (_, offset) => {
      const idx = (activeIndex + offset) % rotatingProjects.length;
      return rotatingProjects[idx];
    });
  }, [activeIndex, rotatingProjects]);

  const current = useMemo(
    () => (openProject ? CATEGORIES[openProject.cat].projects[openProject.idx] : null),
    [openProject]
  );

  return (
    <section id="projects" className="section-top px-4 pb-20 pt-24 sm:px-8 sm:pt-28 lg:px-12">
      <div className="mx-auto w-full max-w-6xl">
        <h2 className="section-title reveal-title mb-10 text-center text-3xl sm:text-4xl">
          <span className="section-title-accent">Projects</span>
        </h2>

        {visibleProjects.length > 0 && (
          <div className="w-full">
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
              {visibleProjects.map((entry, idx) => {
                const highlighted = idx === 0;
                return (
                  <button
                    key={`${entry.cat}-${entry.idx}-${activeIndex}-${idx}`}
                    onClick={() => setOpenProject({ cat: entry.cat, idx: entry.idx })}
                    className={`group relative overflow-hidden border text-left transition-colors ${
                      highlighted
                        ? "border-[var(--accent)]"
                        : "border-[var(--line-soft)] hover:border-[var(--accent)]"
                    }`}
                  >
                    <div className={`relative ${highlighted ? "h-72" : "h-56"}`}>
                      <Image
                        src={entry.project.images?.[0] || PLACEHOLDER}
                        alt={`${entry.project.title} preview`}
                        fill
                        sizes="(max-width: 1280px) 50vw, 25vw"
                        className={`object-cover transition-transform duration-500 ${highlighted ? "scale-[1.01]" : "group-hover:scale-105"}`}
                      />
                      <div className={`absolute inset-0 ${highlighted ? "bg-gradient-to-t from-black/70 via-black/35 to-transparent" : "bg-gradient-to-t from-black/65 via-black/20 to-transparent"}`} />
                      <div className="absolute inset-x-0 bottom-0 p-4">
                        <p className="mb-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-white/80">
                          {entry.category}
                        </p>
                        <h3 className={`text-white ${highlighted ? "text-lg font-semibold" : "text-sm font-medium"}`}>
                          {entry.project.title}
                        </h3>
                        {highlighted && (
                          <p className="mt-2 line-clamp-2 text-xs text-white/85">
                            {entry.project.subtitle}
                          </p>
                        )}
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>

            <div className="mt-5 flex flex-wrap items-center gap-2">
              <button
                onClick={() => {
                  const first = visibleProjects[0];
                  if (first) setOpenProject({ cat: first.cat, idx: first.idx });
                }}
                className="rounded-md bg-[var(--accent)] px-3 py-1.5 text-sm text-white transition hover:bg-[var(--text-main)]"
              >
                View Highlighted
              </button>
              <button
                onClick={goPrev}
                className="rounded-md border border-[var(--line-soft)] bg-[var(--bg-1)] px-3 py-1.5 text-sm text-[var(--text-soft)] transition hover:border-[var(--accent)] hover:text-[var(--accent)]"
                aria-label="Previous project"
              >
                Prev
              </button>
              <button
                onClick={goNext}
                className="rounded-md bg-[var(--text-main)] px-3 py-1.5 text-sm text-white transition hover:bg-[var(--accent)]"
                aria-label="Next project"
              >
                Next
              </button>
            </div>

            <div className="mt-3 flex items-center gap-1.5">
              {rotatingProjects.map((entry, idx) => (
                <button
                  key={`${entry.cat}-${entry.idx}`}
                  onClick={() => setActiveIndex(idx)}
                  className={`h-1.5 rounded-sm transition-all ${
                    idx === activeIndex ? "w-6 bg-[var(--accent)]" : "w-2 bg-[var(--line-soft)] hover:bg-[var(--text-soft)]"
                  }`}
                  aria-label={`Go to project ${idx + 1}`}
                />
              ))}
            </div>
          </div>
        )}

        {current && (
          <div className="fixed inset-0 z-[60]">
            <div
              className="absolute inset-0 bg-[var(--bg-0)]/70 backdrop-blur-sm"
              onClick={() => setOpenProject(null)}
            />
            <div className="absolute inset-3 overflow-hidden border border-[var(--line-soft)] bg-[var(--bg-1)] md:inset-10">
              <div className="relative h-full overflow-y-auto p-6 md:p-10">
                <div className="mb-6 flex items-start justify-between gap-6">
                  <div>
                    <h3 className="font-display text-3xl font-medium leading-tight tracking-tight text-[var(--text-main)] md:text-4xl">
                      {current.title}
                    </h3>
                    {current.subtitle && <p className="mt-1 text-[var(--text-soft)]">{current.subtitle}</p>}
                    {current.tech?.length ? (
                      <div className="mt-3 flex flex-wrap gap-2">
                        {current.tech.map((t, i) => (
                          <span key={i} className="tag">
                            {t}
                          </span>
                        ))}
                      </div>
                    ) : null}
                  </div>
                  <button
                    onClick={() => setOpenProject(null)}
                    className="rounded-md border border-[var(--line-soft)] bg-[var(--bg-1)] p-3 transition hover:border-[var(--accent)]"
                    aria-label="Close project"
                  >
                    <FaTimes className="text-xl" />
                  </button>
                </div>

                <div className="space-y-10">
                  {current.details && (
                    <section>
                      <h4 className="font-display mb-4 text-xl font-medium text-[var(--text-main)]">Case Study</h4>
                      <div className="grid gap-4 md:grid-cols-2">
                        {current.details.problem && (
                          <div className="border border-[var(--line-soft)] p-5">
                            <div className="mb-2 text-sm font-semibold uppercase tracking-wider text-[var(--accent)]">Problem</div>
                            <p className="text-sm leading-relaxed text-[var(--text-soft)]">{current.details.problem}</p>
                          </div>
                        )}
                        {current.details.data && (
                          <div className="border border-[var(--line-soft)] p-5">
                            <div className="mb-2 text-sm font-semibold uppercase tracking-wider text-[var(--accent)]">Data</div>
                            <p className="text-sm leading-relaxed text-[var(--text-soft)]">{current.details.data}</p>
                          </div>
                        )}
                        {current.details.approach && (
                          <div className="border border-[var(--line-soft)] p-5 md:col-span-2">
                            <div className="mb-2 text-sm font-semibold uppercase tracking-wider text-[var(--accent)]">Approach</div>
                            {Array.isArray(current.details.approach) ? (
                              <ul className="list-disc space-y-1.5 pl-5 text-sm leading-relaxed text-[var(--text-soft)]">
                                {current.details.approach.map((a, i) => (
                                  <li key={i}>{a}</li>
                                ))}
                              </ul>
                            ) : (
                              <p className="text-sm leading-relaxed text-[var(--text-soft)]">{current.details.approach}</p>
                            )}
                          </div>
                        )}
                        {current.details.impact && (
                          <div className="border border-[var(--line-soft)] p-5 md:col-span-2">
                            <div className="mb-2 text-sm font-semibold uppercase tracking-wider text-[var(--accent)]">Impact</div>
                            <p className="text-sm leading-relaxed text-[var(--text-soft)]">{current.details.impact}</p>
                          </div>
                        )}
                      </div>
                    </section>
                  )}

                  {current.bullets?.length ? (
                    <section>
                      <h4 className="font-display mb-3 text-xl font-medium text-[var(--text-main)]">Highlights</h4>
                      <ul className="list-disc space-y-1.5 pl-5 text-sm text-[var(--text-soft)]">
                        {current.bullets.map((b, i) => (
                          <li key={i}>{b}</li>
                        ))}
                      </ul>
                    </section>
                  ) : null}

                  {(() => {
                    const galleryImgs = Array.isArray(current.images) ? current.images.slice(1) : [];
                    return galleryImgs.length ? (
                      <section>
                        <h4 className="font-display mb-3 text-xl font-medium text-[var(--text-main)]">Gallery</h4>
                        <div className="grid gap-4 sm:grid-cols-2">
                          {galleryImgs.map((src, i) => (
                            <div key={i} className="relative h-64 w-full max-h-64">
                              <Image
                                src={src}
                                alt={`gallery-${i + 1}`}
                                fill
                                sizes="(max-width: 1024px) 50vw, 33vw"
                                className="border border-[var(--line-soft)] object-cover"
                              />
                            </div>
                          ))}
                        </div>
                      </section>
                    ) : null;
                  })()}

                  {(current.details?.architecture?.length || current.architectureImg) ? (
                    <section>
                      <h4 className="font-display mb-3 text-xl font-medium text-[var(--text-main)]">Architecture</h4>
                      {current.details?.architecture?.length ? (
                        <ol className="mb-4 list-decimal space-y-1 pl-5 text-sm text-[var(--text-soft)]">
                          {current.details.architecture.map((line, i) => (
                            <li key={i}>{line}</li>
                          ))}
                        </ol>
                      ) : null}
                      {current.architectureImg ? (
                        <div className="relative h-[420px] w-full max-h-[420px]">
                          <Image
                            src={current.architectureImg}
                            alt={`${current.title} architecture`}
                            fill
                            sizes="100vw"
                            className="border border-[var(--line-soft)] bg-[var(--bg-0)] object-contain"
                          />
                        </div>
                      ) : null}
                    </section>
                  ) : null}

                  {current?.reportUrl ? (
                    <section>
                      <h4 className="font-display mb-3 text-xl font-medium text-[var(--text-main)]">Report</h4>
                      <a
                        href={`${current.reportUrl}${current.reportUrl.includes("?") ? "&" : "?"}download=1`}
                        download={fileNameFromUrl(current.reportUrl)}
                        className="mt-1 inline-flex items-center gap-2 rounded-md bg-[var(--text-main)] px-4 py-2 text-white hover:bg-[var(--accent)]"
                        aria-label={`Download ${fileNameFromUrl(current.reportUrl)}`}
                      >
                        <FaDownload />
                        Download PDF
                      </a>
                    </section>
                  ) : null}

                  {current?.details?.resultsImages?.length ? (
                    <section className="mt-6">
                      <h4 className="font-display mb-3 text-xl font-medium text-[var(--text-main)]">Results</h4>
                      {current.details.resultsCaption ? (
                        <p className="mb-4 text-sm text-[var(--text-soft)]">{current.details.resultsCaption}</p>
                      ) : null}
                      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                        {current.details.resultsImages.map((imgSrc: string | StaticImageData, i: number) => {
                          const src = typeof imgSrc === "string" ? imgSrc : imgSrc.src;
                          const fname = fileNameFromUrl(src) || `result_${i + 1}.png`;
                          return (
                            <figure key={i} className="overflow-hidden border border-[var(--line-soft)] bg-[var(--bg-1)]">
                              <Image
                                src={src}
                                alt={`Result ${i + 1}`}
                                width={1600}
                                height={900}
                                className="h-auto w-full"
                                priority={i === 0}
                              />
                              <figcaption className="flex items-center justify-between bg-[var(--bg-0)] px-3 py-2">
                                <span className="text-xs text-[var(--text-soft)]">Result {i + 1}</span>
                                <a
                                  href={`${src}${src.includes("?") ? "&" : "?"}download=1`}
                                  download={fname}
                                  className="inline-flex items-center gap-2 rounded-md border border-[var(--line-soft)] bg-[var(--bg-1)] px-3 py-1 text-sm text-[var(--text-soft)] hover:border-[var(--accent)] hover:text-[var(--accent)]"
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

                  {(current.github || current.demo) && (
                    <section>
                      <h4 className="font-display mb-3 text-xl font-medium text-[var(--text-main)]">Links</h4>
                      <div className="flex gap-4">
                        {current.github && (
                          <a
                            href={current.github}
                            target="_blank"
                            className="flex items-center gap-2 text-[var(--accent)] hover:underline"
                          >
                            <FaGithub /> GitHub
                          </a>
                        )}
                        {current.demo && (
                          <a
                            href={current.demo}
                            target="_blank"
                            className="flex items-center gap-2 text-[var(--accent)] hover:underline"
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
