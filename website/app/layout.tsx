import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Navigation from "@/components/Navigation";
import Footer from "@/components/Footer";
import { ThemeProvider } from "@/components/ThemeProvider";
import { GoogleAnalytics } from "@/components/GoogleAnalytics";
import { GoogleTagManager } from "@/components/GoogleTagManager";
import AnalyticsBootstrapClient from "@/components/AnalyticsBootstrapClient";

const inter = Inter({ 
  subsets: ["latin"],
  display: 'swap',
});

export const metadata: Metadata = {
  metadataBase: new URL('https://runmat.org'),
  manifest: "/manifest.webmanifest",
  title: {
    default: "RunMat - Fast, Free, Modern MATLAB Runtime",
    template: "%s | RunMat"
  },
  description: "RunMat is a pre-release MATLAB-style runtime for early adopters: the core runtime and GPU engine deliver our published speedups, while plotting presently covers simple 2D line/scatter views with richer charts still in progress.",
  keywords: [
    "MATLAB", "Octave", "Rust", "JIT compiler", "scientific computing", 
    "numerical computing", "open source", "high performance", "plotting",
    "mathematics", "engineering", "simulation", "drop-in replacement", "dystr",
    "jupyter kernel", "jupyter matlab", "blas", "lapack", "matlab jupyter"
  ],
  authors: [{ name: "RunMat", url: "https://runmat.org" }],
  creator: "RunMat",
  publisher: "RunMat",
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
  openGraph: {
    type: 'website',
    locale: 'en_US',
    url: 'https://runmat.org',
    siteName: 'RunMat',
    title: 'RunMat - High-Performance MATLAB/Octave Runtime',
    description: 'Pre-release MATLAB-style runtime delivering benchmarked CPU/GPU speedups; plotting currently covers simple 2D line/scatter while 3D and richer chart types are still being built.',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'RunMat - High-Performance MATLAB/Octave Runtime',
    description: 'RunMat is in pre-release: fast MATLAB-style runtime with proven benchmarks today and plotting limited to simple 2D line/scatter until richer charts land.',
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link rel="icon" href="/favicon.ico" sizes="32x32" />
        <link rel="icon" href="/favicon.svg" type="image/svg+xml" />
        <link rel="apple-touch-icon" href="/apple-touch-icon.png" />
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              "@context": "https://schema.org",
              "@graph": [
                {
                  "@type": "Organization",
                  "@id": "https://runmat.org/#organization",
                  "name": "RunMat",
                  "alternateName": ["RunMat by Dystr", "Dystr"],
                  "legalName": "Dystr Inc.",
                  "url": "https://runmat.org",
                  "logo": {
                    "@type": "ImageObject",
                    "url": "https://runmat.org/runmat-logo.svg",
                    "caption": "RunMat"
                  },
                  "description":
                    "RunMat is a high-performance, open-source runtime for math that lets you run MATLAB-syntax code in the browser, on the desktop, or from the CLI, while getting GPU-speed execution.",
                  "sameAs": [
                    "https://github.com/runmat-org/runmat",
                    "https://x.com/runmat_org",
                    "https://dystr.com"
                  ],
                  "knowsAbout": [
                    "Scientific Computing",
                    "High Performance Computing",
                    "MATLAB",
                    "WebGPU",
                    "Compiler Design"
                  ],
                  "contactPoint": {
                    "@type": "ContactPoint",
                    "contactType": "customer support",
                    "email": "team@runmat.com"
                  }
                },
                {
                  "@type": "WebSite",
                  "@id": "https://runmat.org/#website",
                  "url": "https://runmat.org",
                  "name": "RunMat",
                  "description":
                    "The Fastest Runtime for Your Math. RunMat fuses back-to-back ops into fewer GPU steps and intelligently manages memory.",
                  "publisher": { "@id": "https://runmat.org/#organization" },
                  "image": "https://web.runmatstatic.com/runmat-sandbox-dark.png",
                  "potentialAction": {
                    "@type": "SearchAction",
                    "target": {
                      "@type": "EntryPoint",
                      "urlTemplate": "https://runmat.org/search?q={search_term_string}"
                    },
                    "query-input": "required name=search_term_string"
                  }
                },
                {
                  "@type": "SoftwareApplication",
                  "@id": "https://runmat.org/#software",
                  "name": "RunMat",
                  "description":
                    "RunMat is a high-performance, open-source runtime for math that lets you run MATLAB-syntax code in the browser, on the desktop, or from the CLI, while getting GPU-speed execution.",
                  "license": "https://opensource.org/licenses/MIT",
                  "applicationCategory": "ScientificApplication",
                  "applicationSubCategory": "Numerical Analysis & Simulation",
                  "operatingSystem": ["Windows", "macOS", "Linux", "Browser"],
                  "softwareVersion": "Beta",
                  "featureList": [
                    "JIT-accelerated MATLAB-style syntax",
                    "RunMat Desktop: Full IDE experience with code editor, file explorer, and live plotting in-browser",
                    "Automatic GPU Fusion & Memory Management",
                    "Cross-platform binary (Metal, Vulkan, DX12) and CLI support"
                  ],
                  "offers": {
                    "@type": "Offer",
                    "price": "0",
                    "priceCurrency": "USD",
                    "availability": "https://schema.org/InStock"
                  },
                  "author": { "@id": "https://runmat.org/#organization" },
                  "publisher": { "@id": "https://runmat.org/#organization" },
                  "downloadUrl": "https://runmat.org/download",
                  "mainEntityOfPage": { "@id": "https://runmat.org/#website" },
                  "screenshot": {
                    "@type": "ImageObject",
                    "url": "https://web.runmatstatic.com/runmat-sandbox-dark.png",
                    "caption": "RunMat Desktop and Browser Sandbox"
                  }
                }
              ]
            })
          }}
        />
        <GoogleAnalytics />
      </head>
      <body className={`${inter.className} antialiased`} suppressHydrationWarning>
        <GoogleTagManager />
        <AnalyticsBootstrapClient />
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          enableSystem={false}
          disableTransitionOnChange={false}
        >
          <div className="relative flex min-h-screen flex-col">
            <Navigation />
            <main className="flex-1">
              {children}
            </main>
            <Footer />
          </div>
        </ThemeProvider>
      </body>
    </html>
  );
}
