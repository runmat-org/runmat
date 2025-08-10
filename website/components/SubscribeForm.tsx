"use client";

import { useState } from "react";

export default function SubscribeForm() {
  const [email, setEmail] = useState("");
  const [status, setStatus] = useState<"idle" | "loading" | "success" | "error">("idle");

  return (
    <div className="w-full max-w-xl">
      <form
        className="flex w-full flex-col gap-2 sm:flex-row sm:items-center sm:gap-3"
        onSubmit={async (e) => {
          e.preventDefault();
          if (!email) return;
          setStatus("loading");
          const resp = await fetch("/api/subscribe", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ email, pageUri: window.location.href, pageName: document.title }),
          }).catch(() => undefined);
          if (resp && resp.ok) {
            setStatus("success");
            setEmail("");
          } else {
            setStatus("error");
          }
        }}
      >
        <label htmlFor="subscribe-email" className="sr-only">Email address</label>
        <input
          id="subscribe-email"
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="you@example.com"
          className="w-full h-11 rounded-lg border border-border bg-muted/20 px-4 text-base text-foreground placeholder:text-muted-foreground shadow-sm focus:outline-none focus:ring-2 focus:ring-ring/40 sm:h-10 sm:text-sm sm:flex-1 min-w-0"
          aria-label="Email address"
          autoComplete="email"
          inputMode="email"
          required
          disabled={status === "loading" || status === "success"}
        />
        <button
          type="submit"
          className="h-11 w-full sm:h-10 sm:w-auto flex-shrink-0 whitespace-nowrap rounded-lg bg-gradient-to-r from-blue-500 to-purple-600 px-5 sm:px-6 text-base sm:text-sm font-semibold text-white shadow hover:from-blue-600 hover:to-purple-700 transition-colors disabled:opacity-60"
          disabled={status === "loading" || status === "success"}
        >
          {status === "loading" ? "Subscribing..." : status === "success" ? "Subscribed" : "Stay in the loop"}
        </button>
      </form>

      {status === "error" && (
        <p className="mt-2 text-xs text-destructive">Something went wrong. Please try again.</p>
      )}
      {status === "success" && (
        <p className="mt-2 text-xs text-muted-foreground">You&apos;re subscribed! We&apos;ll be in touch soon.</p>
      )}
    </div>
  );
}

