import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import { ThemeProvider } from "next-themes";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains",
  display: "swap",
});

export const metadata: Metadata = {
  metadataBase: new URL("https://agfx.vercel.app"),
  title: {
    default: "AvianFlu Genomic Intelligence | AI Genomic Classification",
    template: "%s | AGFX Research",
  },
  description:
    "A deep learning framework for classifying avian influenza genomic sequences into Benign, Likely Benign, Likely Pathogenic, and Pathogenic categories using HyenaDNA + Transformer architecture.",
  keywords: [
    "avian influenza",
    "genomic classification",
    "deep learning",
    "HyenaDNA",
    "DNABERT",
    "bioinformatics",
    "pathogenicity prediction",
    "AI research",
    "genomic AI",
    "pandemic surveillance",
  ],
  authors: [{ name: "Vishnu Varthan Senthilkumar" }],
  creator: "Vishnu Varthan Senthilkumar",
  openGraph: {
    type: "website",
    locale: "en_US",
    url: "https://agfx.vercel.app",
    siteName: "AGFX Research",
    title: "AvianFlu Genomic Intelligence",
    description:
      "Deep Learning for Avian Influenza Genomic Sequence Classification",
    images: [
      {
        url: "/og-image.png",
        width: 1200,
        height: 630,
        alt: "AGFX Research Platform",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "AvianFlu Genomic Intelligence",
    description:
      "Deep Learning for Avian Influenza Genomic Sequence Classification",
    images: ["/og-image.png"],
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      "max-video-preview": -1,
      "max-image-preview": "large",
      "max-snippet": -1,
    },
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${inter.variable} ${jetbrainsMono.variable}`}
    >
      <body className="min-h-screen bg-slate-950 text-slate-100 antialiased">
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          enableSystem
          disableTransitionOnChange={false}
        >
          {children}
        </ThemeProvider>
      </body>
    </html>
  );
}
