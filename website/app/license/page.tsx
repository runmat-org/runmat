import { Metadata } from "next";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { CheckCircle, Code, Sparkles, Mail } from "lucide-react";
import Link from "next/link";

export const metadata: Metadata = {
  title: "RunMat License — MIT, Open Source",
  description:
    "The RunMat runtime is open source under the MIT License — the same license used by Julia, VS Code, TypeScript, and Bun. Free to use, modify, and redistribute for any purpose.",
  alternates: { canonical: "https://runmat.com/license" },
  openGraph: {
    url: "https://runmat.com/license",
  },
};

const jsonLd = {
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "WebPage",
      "@id": "https://runmat.com/license#webpage",
      url: "https://runmat.com/license",
      name: "RunMat License — MIT, Open Source",
      description:
        "The RunMat runtime is open source under the MIT License. Free for everyone — individuals, academics, and companies of any size — with no attribution requirements or field-of-use restrictions.",
      inLanguage: "en",
      isPartOf: { "@id": "https://runmat.com/#website" },
      author: { "@id": "https://runmat.com/#organization" },
      publisher: { "@id": "https://runmat.com/#organization" },
      mainEntity: { "@id": "https://runmat.com/license#faq" },
    },
    {
      "@type": "FAQPage",
      "@id": "https://runmat.com/license#faq",
      mainEntityOfPage: { "@id": "https://runmat.com/license#webpage" },
      publisher: { "@id": "https://runmat.com/#organization" },
      mainEntity: [
        {
          "@type": "Question",
          name: "Is RunMat open source?",
          acceptedAnswer: {
            "@type": "Answer",
            text: "Yes. The RunMat runtime is licensed under the MIT License, which is approved by the Open Source Initiative (OSI). It is the same license used by Julia, VS Code, TypeScript, Bun, and Node.js.",
          },
        },
        {
          "@type": "Question",
          name: "Can I use RunMat for free?",
          acceptedAnswer: {
            "@type": "Answer",
            text: "Yes. The MIT License lets anyone use RunMat for any purpose — individuals, researchers, students, startups, large enterprises, and government organizations. There are no fees, no usage limits, and no field-of-use restrictions.",
          },
        },
        {
          "@type": "Question",
          name: "Can I use RunMat in a commercial product?",
          acceptedAnswer: {
            "@type": "Answer",
            text: "Yes. The MIT License explicitly permits commercial use. You can embed RunMat in proprietary software, ship it inside paid products, integrate it into internal tools, or use it as a dependency in any commercial context. Your own code that uses RunMat remains your own.",
          },
        },
        {
          "@type": "Question",
          name: "Can I fork RunMat or modify it?",
          acceptedAnswer: {
            "@type": "Answer",
            text: "Yes. You can fork, modify, redistribute, and sublicense RunMat. The only requirement is that you preserve the MIT copyright notice in source files. There are no attribution requirements in startup messages, documentation, or user interfaces.",
          },
        },
        {
          "@type": "Question",
          name: "What about the Dystr AI assistant — is that open source?",
          acceptedAnswer: {
            "@type": "Answer",
            text: "No. The Dystr AI assistant and Dystr Cloud are commercial products built on top of the open-source RunMat runtime. This is a standard open-core model, similar to VS Code (open source) plus Pylance and GitHub Copilot (commercial). See runmat.com/pricing for details.",
          },
        },
        {
          "@type": "Question",
          name: "Do I need to credit RunMat?",
          acceptedAnswer: {
            "@type": "Answer",
            text: "Beyond preserving the copyright notice in source files (a standard MIT requirement), no. We appreciate attribution in your README or about page, but it is not legally required.",
          },
        },
        {
          "@type": "Question",
          name: "Is the MIT License OSI-approved?",
          acceptedAnswer: {
            "@type": "Answer",
            text: "Yes. The MIT License is one of the original licenses approved by the Open Source Initiative and is on every major enterprise legal team's pre-approved license list. This means RunMat can be adopted at organizations that require formally OSI-approved open-source software.",
          },
        },
        {
          "@type": "Question",
          name: "Will the license ever change?",
          acceptedAnswer: {
            "@type": "Answer",
            text: "We have no plans to change the license of the RunMat runtime away from MIT. Past releases will always remain available under the MIT terms they were originally published under, regardless of any future changes.",
          },
        },
        {
          "@type": "Question",
          name: "Is 'RunMat' a trademark?",
          acceptedAnswer: {
            "@type": "Answer",
            text: "Yes. 'RunMat' is a trademark of Dystr, Inc. The MIT License covers the source code but does not grant rights to the RunMat name or logo. You can fork the code freely; please don't call your fork 'RunMat'. For trademark questions, contact legal@dystr.com.",
          },
        },
      ],
    },
  ],
};

export default function LicensePage() {
  return (
    <div className="min-h-screen bg-background">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: JSON.stringify(jsonLd).replace(/<\//g, "<\\/"),
        }}
      />
      <div className="container mx-auto max-w-4xl px-4 md:px-6 py-16 md:py-24 lg:py-32">

        {/* Header */}
        <div className="mb-12">
          <Badge variant="secondary" className="mb-4">Legal</Badge>
          <h1 className="text-2xl sm:text-3xl font-bold mb-6 text-foreground">
            RunMat License
          </h1>
          <p className="text-[0.938rem] text-foreground leading-relaxed">
            The RunMat runtime is open source under the <strong>MIT License</strong> —
            the same license used by Julia, VS Code, TypeScript, and Bun.
            Free to use, modify, and redistribute for any purpose.
          </p>
        </div>

        {/* Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          <Card className="border-green-300 dark:border-green-800 bg-green-50 dark:bg-green-950/50 shadow-sm">
            <CardHeader className="text-center pb-3">
              <CheckCircle className="h-8 w-8 mx-auto mb-2 text-green-600 dark:text-green-400" />
              <CardTitle className="text-lg text-foreground dark:text-green-100 font-semibold">Open Source (MIT)</CardTitle>
            </CardHeader>
            <CardContent className="text-center">
              <p className="text-sm text-foreground dark:text-green-200 font-medium">
                Free for everyone — individuals, academics, and companies of any size
              </p>
            </CardContent>
          </Card>

          <Card className="bg-card border-border">
            <CardHeader className="text-center pb-3">
              <Code className="h-8 w-8 mx-auto mb-2 text-[hsl(var(--brand))]" />
              <CardTitle className="text-lg text-foreground font-semibold">No Restrictions</CardTitle>
            </CardHeader>
            <CardContent className="text-center">
              <p className="text-sm text-foreground font-medium">
                Use commercially, modify, redistribute. No attribution display required.
              </p>
            </CardContent>
          </Card>

          <Card className="bg-card border-border">
            <CardHeader className="text-center pb-3">
              <Sparkles className="h-8 w-8 mx-auto mb-2 text-[hsl(var(--brand))]" />
              <CardTitle className="text-lg text-foreground font-semibold">AI Assistant Separate</CardTitle>
            </CardHeader>
            <CardContent className="text-center">
              <p className="text-sm text-foreground font-medium">
                The Dystr AI assistant is a commercial product — like VS Code + Copilot
              </p>
            </CardContent>
          </Card>
        </div>

        {/* FAQ Section */}
        <section className="mb-12">
          <h2 className="text-xl sm:text-2xl font-bold mb-8 text-foreground">
            Frequently Asked Questions
          </h2>

          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg font-semibold text-foreground">
                  Is RunMat open source?
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-foreground">
                  <strong>Yes.</strong> The RunMat runtime is licensed under the MIT License,
                  which is approved by the{" "}
                  <Link
                    href="https://opensource.org/license/mit"
                    className="underline"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Open Source Initiative
                  </Link>
                  . It is the same license used by Julia, VS Code, TypeScript, Bun, and Node.js.
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg font-semibold text-foreground">
                  Can I use RunMat for free?
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-foreground mb-3">
                  <strong>Yes.</strong> The MIT License lets anyone use RunMat for any purpose:
                </p>
                <ul className="list-disc list-inside space-y-1 text-foreground/80 ml-4">
                  <li>Individual researchers, scientists, and engineers</li>
                  <li>Academic institutions and educational organizations</li>
                  <li>Students for learning and coursework</li>
                  <li>Companies of any size, including direct competitors</li>
                  <li>Open source projects and their maintainers</li>
                  <li>Government agencies and non-profits</li>
                </ul>
                <p className="text-foreground/90 mt-3">
                  There are no fees, no usage limits, and no field-of-use restrictions.
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg font-semibold text-foreground">
                  Can I use RunMat in a commercial product?
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-foreground">
                  <strong>Yes.</strong> The MIT License explicitly permits commercial use.
                  You can embed RunMat in proprietary software, ship it inside paid products,
                  integrate it into internal tools, or use it as a dependency in any commercial
                  context. Your own code that uses RunMat remains your own — RunMat does not
                  impose copyleft on derivative works.
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg font-semibold text-foreground">
                  Can I fork RunMat or modify it?
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-foreground">
                  <strong>Yes.</strong> You can fork, modify, redistribute, and sublicense RunMat.
                  The only requirement is that you preserve the MIT copyright notice in source files.
                  There are no attribution requirements in startup messages, documentation, or user
                  interfaces.
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg font-semibold text-foreground">
                  What about the Dystr AI assistant — is that open source?
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-foreground mb-3">
                  <strong>No.</strong> The Dystr AI assistant and Dystr Cloud are commercial
                  products built on top of the open-source RunMat runtime.
                </p>
                <p className="text-foreground mb-3">
                  This is a standard open-core model, similar to VS Code (open source under MIT)
                  paired with Pylance and GitHub Copilot (commercial). The runtime is free forever;
                  the AI assistant is a paid Dystr product.
                </p>
                <p className="text-foreground/90">
                  See{" "}
                  <Link href="/pricing" className="underline">
                    runmat.com/pricing
                  </Link>{" "}
                  for details on the AI assistant.
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg font-semibold text-foreground">
                  Do I need to credit RunMat?
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-foreground">
                  Beyond preserving the copyright notice in source files (a standard MIT requirement),
                  <strong> no</strong>. We appreciate attribution in your README or about page,
                  but it is not legally required.
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg font-semibold text-foreground">
                  Is the MIT License OSI-approved?
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-foreground">
                  <strong>Yes.</strong> The MIT License is one of the original licenses approved by
                  the Open Source Initiative and is on every major enterprise legal team&apos;s
                  pre-approved license list. RunMat can be adopted at organizations that require
                  formally OSI-approved open-source software.
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg font-semibold text-foreground">
                  Will the license ever change?
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-foreground">
                  We have no plans to change the license of the RunMat runtime away from MIT.
                  Past releases will always remain available under the MIT terms they were
                  originally published under, regardless of any future changes.
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg font-semibold text-foreground">
                  Is &quot;RunMat&quot; a trademark?
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-foreground mb-3">
                  Yes. <strong>&quot;RunMat&quot;</strong> is a trademark of Dystr, Inc.
                  The MIT License covers the source code but does not grant rights to the RunMat
                  name or logo.
                </p>
                <p className="text-foreground/90">
                  You can fork the code freely; please don&apos;t call your fork &quot;RunMat&quot;.
                  For trademark questions, contact{" "}
                  <Link href="mailto:legal@dystr.com" className="underline">
                    legal@dystr.com
                  </Link>
                  .
                </p>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* Full License Text */}
        <section className="mb-12">
          <h2 className="text-xl sm:text-2xl font-bold mb-8 text-foreground">
            Full License Text
          </h2>

          <Card className="p-0">
            <CardContent className="p-6">
              <div className="font-mono text-sm text-foreground/90 whitespace-pre-line leading-relaxed">
{`MIT License

Copyright (c) 2024-2026 Dystr, Inc. and RunMat contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.`}
              </div>
            </CardContent>
          </Card>
        </section>

        {/* Contact Section */}
        <section>
          <Card className="bg-card border-border">
            <CardContent className="p-6 text-center">
              <h3 className="text-lg font-semibold mb-3 text-foreground">
                Questions about commercial products or trademarks?
              </h3>
              <p className="text-[0.938rem] text-foreground mb-4">
                The MIT License covers the runtime. For questions about the Dystr AI assistant,
                Dystr Cloud, enterprise agreements, or trademark use, get in touch.
              </p>
              <p className="text-xs text-foreground mb-4">
                New to MATLAB? Read the primer{" "}
                <Link href="/blog/what-is-matlab" className="underline">
                  What is MATLAB?
                </Link>
                .
              </p>
              <div className="flex items-center justify-center space-x-2">
                <Mail className="h-4 w-4 text-[hsl(var(--brand))]" />
                <Link
                  href="mailto:legal@dystr.com"
                  className="text-[hsl(var(--brand))] hover:underline font-medium"
                >
                  legal@dystr.com
                </Link>
              </div>
            </CardContent>
          </Card>
        </section>
      </div>
    </div>
  );
}
