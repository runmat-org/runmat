"use client";

import Script from 'next/script';

// Google Analytics Measurement ID
// You'll need to replace this with your actual GA4 Measurement ID
const GA_MEASUREMENT_ID = process.env.NEXT_PUBLIC_GA_MEASUREMENT_ID || 'G-XXXXXXXXXX';

export function GoogleAnalytics() {
  // Only load analytics in production
  if (process.env.NODE_ENV !== 'production') {
    return null;
  }

  return (
    <>
      <Script
        src={`https://www.googletagmanager.com/gtag/js?id=${GA_MEASUREMENT_ID}`}
        strategy="afterInteractive"
      />
      <Script id="google-analytics" strategy="afterInteractive">
        {`
          window.dataLayer = window.dataLayer || [];
          function gtag(){dataLayer.push(arguments);}
          gtag('js', new Date());
          gtag('config', '${GA_MEASUREMENT_ID}', {
            page_title: document.title,
            page_location: window.location.href,
          });
          try {
            gtag('get', '${GA_MEASUREMENT_ID}', 'client_id', function (clientId) {
              if (clientId) {
                window.dispatchEvent(new CustomEvent('ga_client_id', { detail: clientId }));
              }
            });
          } catch (e) {}
        `}
      </Script>
    </>
  );
}

// Helper function to track custom events
export const trackEvent = (action: string, category: string, label?: string, value?: number) => {
  if (typeof window !== 'undefined' && window.gtag) {
    window.gtag('event', action, {
      event_category: category,
      event_label: label,
      value: value,
    });
  }
};

// Helper function to track page views (for SPA navigation)
export const trackPageView = (url: string, title?: string) => {
  if (typeof window !== 'undefined' && window.gtag) {
    window.gtag('config', GA_MEASUREMENT_ID, {
      page_title: title || document.title,
      page_location: url,
    });
  }
};

// Types for Google Analytics
type GTagEvent = {
  event_category?: string;
  event_label?: string;
  value?: number;
};

type GTagConfig = {
  page_title?: string;
  page_location?: string;
};

// Extend the Window interface to include gtag
declare global {
  interface Window {
    gtag: (command: 'config' | 'event' | 'js', target: string | Date, config?: GTagConfig | GTagEvent) => void;
    dataLayer: unknown[];
  }
}