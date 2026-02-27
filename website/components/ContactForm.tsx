"use client";

import { useState, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import { trackEvent } from "@/components/GoogleAnalytics";

const inputClass =
  "w-full h-11 rounded-lg border border-border bg-muted/20 px-4 text-base text-foreground placeholder:text-muted-foreground shadow-sm focus:outline-none focus:ring-2 focus:ring-ring/40 sm:h-10 sm:text-sm disabled:opacity-60";

export default function ContactForm() {
  const searchParams = useSearchParams();
  const inquiryType = searchParams.get("type");

  const [firstname, setFirstname] = useState("");
  const [lastname, setLastname] = useState("");
  const [email, setEmail] = useState("");
  const [company, setCompany] = useState("");
  const [message, setMessage] = useState("");
  const [status, setStatus] = useState<"idle" | "loading" | "success" | "error">("idle");

  useEffect(() => {
    if (inquiryType === "enterprise" && !message) {
      setMessage("I'm interested in RunMat Enterprise.");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inquiryType]);

  const disabled = status === "loading" || status === "success";

  return (
    <div className="w-full max-w-lg">
      <form
        className="flex flex-col gap-4"
        onSubmit={async (e) => {
          e.preventDefault();
          if (!firstname || !lastname || !email || !message) return;
          setStatus("loading");
          const resp = await fetch("/api/contact", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              firstname,
              lastname,
              email,
              company,
              message,
              pageUri: window.location.href,
              pageName: document.title,
            }),
          }).catch(() => undefined);

          if (resp && resp.ok) {
            setStatus("success");
            trackEvent("contact_form_submit", "contact", inquiryType || "general");
          } else {
            setStatus("error");
          }
        }}
      >
        <div className="grid gap-4 sm:grid-cols-2">
          <div>
            <label htmlFor="contact-firstname" className="mb-1.5 block text-sm font-medium text-foreground">
              First name <span className="text-destructive">*</span>
            </label>
            <input
              id="contact-firstname"
              type="text"
              value={firstname}
              onChange={(e) => setFirstname(e.target.value)}
              placeholder="Jane"
              className={inputClass}
              autoComplete="given-name"
              required
              disabled={disabled}
            />
          </div>
          <div>
            <label htmlFor="contact-lastname" className="mb-1.5 block text-sm font-medium text-foreground">
              Last name <span className="text-destructive">*</span>
            </label>
            <input
              id="contact-lastname"
              type="text"
              value={lastname}
              onChange={(e) => setLastname(e.target.value)}
              placeholder="Doe"
              className={inputClass}
              autoComplete="family-name"
              required
              disabled={disabled}
            />
          </div>
        </div>

        <div>
          <label htmlFor="contact-email" className="mb-1.5 block text-sm font-medium text-foreground">
            Work email <span className="text-destructive">*</span>
          </label>
          <input
            id="contact-email"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="jane@company.com"
            className={inputClass}
            autoComplete="email"
            inputMode="email"
            required
            disabled={disabled}
          />
        </div>

        <div>
          <label htmlFor="contact-company" className="mb-1.5 block text-sm font-medium text-foreground">
            Company
          </label>
          <input
            id="contact-company"
            type="text"
            value={company}
            onChange={(e) => setCompany(e.target.value)}
            placeholder="Acme Inc."
            className={inputClass}
            autoComplete="organization"
            disabled={disabled}
          />
        </div>

        <div>
          <label htmlFor="contact-message" className="mb-1.5 block text-sm font-medium text-foreground">
            Message <span className="text-destructive">*</span>
          </label>
          <textarea
            id="contact-message"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="Tell us about your use case or what you're looking for..."
            rows={4}
            className="w-full rounded-lg border border-border bg-muted/20 px-4 py-3 text-base text-foreground placeholder:text-muted-foreground shadow-sm focus:outline-none focus:ring-2 focus:ring-ring/40 sm:text-sm resize-y disabled:opacity-60"
            required
            disabled={disabled}
          />
        </div>

        <button
          type="submit"
          className="h-12 w-full px-8 text-base font-semibold rounded-lg bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow hover:from-blue-600 hover:to-purple-700 transition-colors disabled:opacity-60 sm:h-10 sm:text-sm"
          disabled={disabled}
          data-ph-capture-attribute-destination="contact"
          data-ph-capture-attribute-source="contact-page"
          data-ph-capture-attribute-cta="submit-contact-form"
        >
          {status === "loading" ? "Sending..." : status === "success" ? "Sent!" : "Send Message"}
        </button>
      </form>

      {status === "error" && (
        <p className="mt-3 text-sm text-destructive">Something went wrong. Please try again.</p>
      )}
      {status === "success" && (
        <p className="mt-3 text-sm text-muted-foreground">
          Thanks for reaching out! We&apos;ll get back to you soon.
        </p>
      )}
    </div>
  );
}
