import { buildPageMetadata } from "@/lib/seo";
import { EigenvalueExplorerClient } from "@/components/guides/EigenvalueExplorerClient";
import Link from "next/link";

const EIG_SOURCE_URL =
  "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/factor/eig.rs";

export const metadata = buildPageMetadata({
  title: "Interactive Eigenvalue Explorer | RunMat",
  description:
    "Eigenvalue calculator and interactive 2x2 matrix eigenvalue visualization. See eigenvalues on the complex plane in real time. Try eig(A) in RunMat.",
  canonicalPath: "/guides/eigenvalue-explorer",
  ogType: "article",
});

const webApplicationSchema = {
  "@context": "https://schema.org",
  "@type": "WebApplication",
  name: "Eigenvalue Explorer",
  description: "Interactive 2x2 matrix eigenvalue visualization on the complex plane",
  url: "https://runmat.com/guides/eigenvalue-explorer",
  applicationCategory: "EducationalApplication",
};

export default function EigenvalueExplorerPage() {
  return (
    <div className="min-h-screen bg-background">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(webApplicationSchema) }}
      />
      <div className="container mx-auto px-4 md:px-0 py-8 md:py-12">
        <header className="mb-8 md:mb-10">
          <h1 className="text-3xl md:text-4xl font-bold text-foreground">
            What happens when you call eig(A)?
          </h1>
          <p className="mt-2 text-muted-foreground max-w-2xl">
            Change the 2×2 matrix below and watch its eigenvalues move on the complex plane. No
            backend—eigenvalues are computed instantly with the quadratic formula.
          </p>
        </header>

        <EigenvalueExplorerClient />

        <section className="mt-12 max-w-3xl space-y-6 text-muted-foreground">
          <h2 className="text-xl font-semibold text-foreground">What are eigenvalues?</h2>
          <p>
            Eigenvalues tell you the fundamental modes of a linear system. For a matrix A, the
            eigenvalues are the scalars λ where <code className="rounded bg-muted px-1">A·v = λ·v</code> for
            some non-zero vector v.
          </p>
          <h2 className="text-xl font-semibold text-foreground">Why do they matter?</h2>
          <ul className="list-disc pl-6 space-y-1">
            <li>
              <strong>Stability analysis:</strong> If all eigenvalues have negative real parts, the
              system is stable.
            </li>
            <li>
              <strong>Vibration frequencies:</strong> Eigenvalues of a mass-spring system give
              natural frequencies.
            </li>
            <li>
              <strong>Principal Component Analysis:</strong> Eigenvalues of the covariance matrix
              rank the principal components.
            </li>
          </ul>
          <h2 className="text-xl font-semibold text-foreground">What&apos;s happening under the hood?</h2>
          <p>
            For a 2×2 matrix, eigenvalues come from the quadratic formula applied to the
            characteristic polynomial. For larger matrices, RunMat uses Schur decomposition.{" "}
            <Link href={EIG_SOURCE_URL} className="text-primary hover:underline" target="_blank" rel="noopener noreferrer">
              Read the eig source on GitHub
            </Link>
            .
          </p>
        </section>
      </div>
    </div>
  );
}
