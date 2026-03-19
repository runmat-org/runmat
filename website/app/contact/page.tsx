import type { Metadata } from "next";
import { Suspense } from "react";
import ContactForm from "@/components/ContactForm";

export const metadata: Metadata = {
  title: "Contact Us",
  description:
    "Get in touch with the RunMat team. Reach out about Enterprise deployment, partnerships, or general inquiries.",
  alternates: { canonical: "https://runmat.com/contact" },
  openGraph: {
    type: "website",
    url: "https://runmat.com/contact",
    title: "Contact Us | RunMat",
    description:
      "Get in touch with the RunMat team. Reach out about Enterprise deployment, partnerships, or general inquiries.",
  },
  twitter: {
    card: "summary_large_image",
    title: "Contact Us | RunMat",
    description:
      "Get in touch with the RunMat team. Reach out about Enterprise deployment, partnerships, or general inquiries.",
  },
};

const jsonLd = {
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "ContactPage",
      "@id": "https://runmat.com/contact#webpage",
      url: "https://runmat.com/contact",
      name: "Contact Us | RunMat",
      description:
        "Get in touch with the RunMat team. Reach out about Enterprise deployment, partnerships, or general inquiries.",
      inLanguage: "en",
      isPartOf: { "@id": "https://runmat.com/#website" },
      breadcrumb: { "@id": "https://runmat.com/contact#breadcrumb" },
      mainEntity: { "@id": "https://runmat.com/#organization" },
    },
    {
      "@type": "BreadcrumbList",
      "@id": "https://runmat.com/contact#breadcrumb",
      itemListElement: [
        { "@type": "ListItem", position: 1, name: "Home", item: "https://runmat.com" },
        { "@type": "ListItem", position: 2, name: "Contact", item: "https://runmat.com/contact" },
      ],
    },
  ],
};

export default function ContactPage() {
  return (
    <div className="min-h-screen bg-background">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: JSON.stringify(jsonLd).replace(/<\//g, "<\\/"),
        }}
      />
      <div className="container mx-auto px-4 md:px-6 lg:px-8 pt-12 md:pt-16 lg:pt-20 pb-16 md:pb-24 lg:pb-32">
        <section className="mx-auto max-w-lg">
          <div className="mb-8 space-y-3 text-center">
            <h1 className="text-3xl font-bold text-foreground md:text-4xl">
              Get in touch
            </h1>
            <p className="text-lg text-muted-foreground">
              Have a question about RunMat Enterprise, partnerships, or anything else? Fill out the
              form below and we&apos;ll get back to you.
            </p>
          </div>

          <Suspense>
            <ContactForm />
          </Suspense>

          <p className="mt-8 text-center text-sm text-muted-foreground">
            You can also email us directly at{" "}
            <a
              href="mailto:team@runmat.com"
              className="underline hover:text-foreground"
            >
              team@runmat.com
            </a>
          </p>
        </section>
      </div>
    </div>
  );
}
