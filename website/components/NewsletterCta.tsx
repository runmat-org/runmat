"use client";

import React from "react";
import SubscribeForm from "@/components/SubscribeForm";

type NewsletterCtaProps = {
  title?: string;
  description?: string;
  align?: "left" | "center";
  className?: string;
};

export default function NewsletterCta({
  title,
  description,
  align = "left",
  className = "",
}: NewsletterCtaProps) {
  const textAlignClass = align === "center" ? "text-center" : "text-left";
  const formWrapClass = align === "center" ? "flex justify-center" : "";

  return (
    <section className={`not-prose ${className}`} aria-label="Newsletter signup">
      {(title || description) && (
        <div className={`${textAlignClass} mb-4`}>
          {title && (
            <h3 className="text-lg font-semibold">{title}</h3>
          )}
          {description && (
            <p className="text-sm text-muted-foreground mt-1">{description}</p>
          )}
        </div>
      )}
      <div className={formWrapClass}>
        <SubscribeForm />
      </div>
    </section>
  );
}

