# AvianFlu Genomic Intelligence Research Website

A production-ready research platform built with Next.js 16 (App Router), TypeScript, TailwindCSS v4, Framer Motion, and Recharts showcasing the **Avian Influenza Genomic AI Framework**.

![Hero Preview](hero-preview.png)

## Features

- **Hero Section**: Animated DNA particle background, gradient headlines, live metric badges.
- **Interactive Pipeline**: 7-stage interactive data processing workflow with algorithm details, inputs, and outputs.
- **System Architecture**: Tensor flow diagram, parameter breakdown (10.21M), and module specifications.
- **Verified Benchmark Results**: Recharts bar charts, radar profile maps, per-class evaluation tables, and confusion matrix.
- **Technical Documentation**: Researcher manual with copyable CLI commands, requirements, and troubleshooting guides.
- **Download Center**: Access source code repositories, pre-trained weights, and CSV/JSON/HTML reports.
- **Visual Gallery**: Output plots and charts with interactive lightbox preview modals.
- **Dark Mode**: Native `next-themes` dark/light mode toggle with custom glassmorphic styling.
- **SEO & Performance**: Pre-configured OpenGraph tags, Twitter cards, meta tags, and static site generation.

---

## Getting Started

### Prerequisites

- Node.js 18.0 or higher
- npm, pnpm, yarn, or bun

### Local Development

1. Navigate to the website directory:
   ```bash
   cd website
   ```

2. Install dependencies (if not already installed):
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

4. Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## Project Structure

```
website/
├── app/                      # Next.js App Router pages
│   ├── page.tsx              # Home Page (Hero, Stats, Features)
│   ├── about/                # About Page (Motivation, Objectives, Highlights)
│   ├── pipeline/             # Interactive 7-Stage Pipeline
│   ├── architecture/         # High-Res System Architecture & Tensor Flow
│   ├── models/               # PyTorch Module Specifications & Code Snippets
│   ├── experiments/          # Hyperparameters, Dataset Stats & Hardware
│   ├── results/              # Verified Benchmark Metrics, Charts & Confusion Matrix
│   ├── reports/              # Generated JSON, CSV, MD & HTML Reports
│   ├── documentation/        # Technical Developer Manual & Commands
│   ├── downloads/            # Download Cards (Weights, Code, Reports)
│   ├── gallery/              # PNG Output Plots with Lightbox Modal
│   ├── research/             # Scientific Abstract, Baselines & BibTeX
│   ├── not-found.tsx         # Custom 404 Page
│   ├── globals.css           # Tailwind v4, Glassmorphism, Theme Tokens
│   └── layout.tsx            # Root Layout with Inter + JetBrains Mono Fonts
├── components/
│   ├── layout/               # Navbar & Footer
│   ├── home/                 # Hero, StatsBar, FeaturesSection
│   ├── ui/                   # Reusable Button, Card, Badge, MetricCard
│   └── charts/               # Recharts MetricsChart & RadarChart
├── content/
│   └── project.json          # Single Source of Truth for all Project Data
├── lib/
│   ├── content.ts            # Data helper bindings
│   └── utils.ts              # Classname merge & formatting utilities
├── tailwind.config.ts        # Extended Tailwind theme configuration
└── package.json
```

---

## Deployment Guide

### Deploying to Vercel (Recommended)

1. Push your repository to GitHub / GitLab.
2. Import the project into [Vercel](https://vercel.com).
3. Set Root Directory to `website`.
4. Vercel will automatically detect Next.js and build the application.

### Deploying to Netlify

1. Connect your repository to [Netlify](https://netlify.com).
2. Set build command: `npm run build`
3. Set publish directory: `.next`

---

## License

Released under the MIT License.
