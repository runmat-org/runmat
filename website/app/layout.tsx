import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Navigation from "@/components/Navigation";
import Footer from "@/components/Footer";
import { ThemeProvider } from "@/components/ThemeProvider";

const inter = Inter({ 
  subsets: ["latin"],
  display: 'swap',
});

export const metadata: Metadata = {
  metadataBase: new URL('https://rustmat.com'),
  title: {
    default: "RustMat - High-Performance MATLAB/Octave Runtime in Rust",
    template: "%s | RustMat"
  },
  description: "A modern, blazing-fast, open-source runtime for MATLAB and GNU Octave code. Built in Rust with a V8-inspired JIT compiler, advanced garbage collection, and GPU-accelerated plotting. Free forever.",
  keywords: [
    "MATLAB", "Octave", "Rust", "JIT compiler", "scientific computing", 
    "numerical computing", "open source", "high performance", "plotting",
    "mathematics", "engineering", "simulation", "dystr"
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
    url: 'https://rustmat.com',
    siteName: 'RustMat',
    title: 'RustMat - High-Performance MATLAB/Octave Runtime in Rust',
    description: 'A modern, blazing-fast, open-source runtime for MATLAB and GNU Octave code. Built in Rust with a V8-inspired JIT compiler, advanced garbage collection, and GPU-accelerated plotting. Free forever.',
    images: [
      {
        url: '/og-image.png',
        width: 1200,
        height: 630,
        alt: 'RustMat - High-Performance MATLAB/Octave Runtime',
      }
    ]
  },
  twitter: {
    card: 'summary_large_image',
    title: 'RustMat - High-Performance MATLAB/Octave Runtime in Rust',
    description: 'A modern, blazing-fast, open-source runtime for MATLAB and GNU Octave code. Built in Rust with a V8-inspired JIT compiler.',
    images: ['/og-image.png'],
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
  verification: {
    google: 'your-google-verification-code',
  }
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
        <link rel="canonical" href="https://rustmat.com" />
      </head>
      <body className={`${inter.className} antialiased`}>
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