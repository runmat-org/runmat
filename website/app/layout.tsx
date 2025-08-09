import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Navigation from "@/components/Navigation";
import Footer from "@/components/Footer";
import { ThemeProvider } from "@/components/ThemeProvider";
import { GoogleAnalytics } from "@/components/GoogleAnalytics";

const inter = Inter({ 
  subsets: ["latin"],
  display: 'swap',
});

export const metadata: Metadata = {
  metadataBase: new URL('https://runmat.org'),
  title: {
    default: "RunMat - High-Performance MATLAB/Octave Runtime",
    template: "%s | RunMat"
  },
  description: "Drop-in replacement for MATLAB and GNU Octave with the same syntax but dramatically faster performance. Built in Rust with JIT compilation, generational garbage collection, and GPU-accelerated plotting. Open source and free.",
  keywords: [
    "MATLAB", "Octave", "Rust", "JIT compiler", "scientific computing", 
    "numerical computing", "open source", "high performance", "plotting",
    "mathematics", "engineering", "simulation", "drop-in replacement", "dystr"
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
    description: 'Drop-in replacement for MATLAB and GNU Octave with the same syntax but dramatically faster performance. Built in Rust with JIT compilation, generational garbage collection, and GPU-accelerated plotting.',
    images: [
      {
        url: '/og-image.png',
        width: 1200,
        height: 630,
        alt: 'RunMat - High-Performance MATLAB/Octave Runtime',
      }
    ]
  },
  twitter: {
    card: 'summary_large_image',
    title: 'RunMat - High-Performance MATLAB/Octave Runtime',
    description: 'Drop-in replacement for MATLAB and GNU Octave with the same syntax but dramatically faster performance. Built in Rust with JIT compilation.',
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
        <GoogleAnalytics />
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