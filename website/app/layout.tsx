import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Navigation from "@/components/Navigation";
import Footer from "@/components/Footer";
import { ThemeProvider } from "@/components/ThemeProvider";
import { GoogleAnalytics } from "@/components/GoogleAnalytics";
import { GoogleTagManager } from "@/components/GoogleTagManager";

const inter = Inter({ 
  subsets: ["latin"],
  display: 'swap',
});

export const metadata: Metadata = {
  metadataBase: new URL('https://runmat.org'),
  title: {
    default: "RunMat - Fast, Free, Modern MATLAB Runtime",
    template: "%s | RunMat"
  },
  description: "Drop-in replacement for MATLAB and GNU Octave with the same syntax but dramatically faster performance. Jupyter kernel, BLAS/LAPACK, JIT compilation, generational garbage collection, and GPU-accelerated plotting. Open source and free.",
  keywords: [
    "MATLAB", "Octave", "Rust", "JIT compiler", "scientific computing", 
    "numerical computing", "open source", "high performance", "plotting",
    "mathematics", "engineering", "simulation", "drop-in replacement", "dystr",
    "jupyter kernel", "jupyter matlab", "blas", "lapack", "matlab jupyter"
  ],
  authors: [{ name: "Dystr Inc.", url: "https://dystr.com" }],
  creator: "Dystr Inc.",
  publisher: "Dystr Inc.",
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
    description: 'Drop-in replacement for MATLAB and GNU Octave with the same syntax but dramatically faster performance. Jupyter kernel, BLAS/LAPACK, JIT compilation, and GPU-accelerated plotting.',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'RunMat - High-Performance MATLAB/Octave Runtime',
    description: 'Drop-in replacement for MATLAB and GNU Octave with the same syntax but dramatically faster performance. Jupyter kernel, BLAS/LAPACK, and JIT compilation.',
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
        <link rel="manifest" href="/site.webmanifest" />
        <link rel="canonical" href="https://runmat.org" />
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              "@context": "https://schema.org",
              "@type": "WebSite",
              "name": "RunMat",
              "description": "Fast, free, modern MATLAB runtime with Jupyter kernel, BLAS/LAPACK, beautiful plotting, and JIT compilation",
              "url": "https://runmat.org",
              "potentialAction": {
                "@type": "SearchAction",
                "target": {
                  "@type": "EntryPoint",
                  "urlTemplate": "https://runmat.org/search?q={search_term_string}"
                },
                "query-input": "required name=search_term_string"
              },
              "mainEntityOfPage": {
                "@type": "WebPage",
                "@id": "https://runmat.org"
              },
              "publisher": {
                "@type": "Organization",
                "name": "Dystr Inc.",
                "url": "https://dystr.com"
              }
            })
          }}
        />
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              "@context": "https://schema.org",
              "@type": "SoftwareApplication",
              "name": "RunMat",
              "description": "Fast, free, modern MATLAB runtime with Jupyter kernel, BLAS/LAPACK, beautiful plotting, and JIT compilation",
              "url": "https://runmat.org",
              "downloadUrl": "https://runmat.org/download",
              "operatingSystem": ["Windows", "macOS", "Linux"],
              "applicationCategory": "Scientific Computing Software",
              "offers": {
                "@type": "Offer",
                "price": "0",
                "priceCurrency": "USD"
              },
              "author": {
                "@type": "Organization",
                "name": "Dystr Inc.",
                "url": "https://dystr.com"
              }
            })
          }}
        />
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              "@context": "https://schema.org",
              "@type": "ItemList",
              "name": "RunMat Navigation",
              "itemListElement": [
                {
                  "@type": "SiteNavigationElement",
                  "position": 1,
                  "name": "Download",
                  "description": "Download RunMat for your platform",
                  "url": "https://runmat.org/download"
                },
                {
                  "@type": "SiteNavigationElement",
                  "position": 2,
                  "name": "Documentation",
                  "description": "Complete guides and reference for RunMat",
                  "url": "https://runmat.org/docs"
                },
                {
                  "@type": "SiteNavigationElement",
                  "position": 3,
                  "name": "Getting Started",
                  "description": "Quick start guide to using RunMat",
                  "url": "https://runmat.org/docs/getting-started"
                },
                {
                  "@type": "SiteNavigationElement",
                  "position": 4,
                  "name": "Architecture",
                  "description": "Deep dive into RunMat's V8-inspired design",
                  "url": "https://runmat.org/docs/architecture"
                },
                {
                  "@type": "SiteNavigationElement",
                  "position": 5,
                  "name": "CLI Reference",
                  "description": "Command-line interface guide",
                  "url": "https://runmat.org/docs/cli"
                },
                {
                  "@type": "SiteNavigationElement",
                  "position": 6,
                  "name": "Configuration",
                  "description": "Configuration system guide",
                  "url": "https://runmat.org/docs/configuration"
                }
              ]
            })
          }}
        />
        <GoogleAnalytics />
      </head>
      <body className={`${inter.className} antialiased`}>
        <GoogleTagManager />
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