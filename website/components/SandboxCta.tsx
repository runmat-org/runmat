import Link from "next/link";
import { Button } from "@/components/ui/button";

interface SandboxCtaProps {
  source?: string;
  secondaryLabel?: string;
  secondaryHref?: string;
}

export function SandboxCta({
  source = "sandbox-cta",
  secondaryLabel = "Other download options",
  secondaryHref = "/download",
}: SandboxCtaProps) {
  return (
    <div className="mx-auto max-w-2xl rounded-lg border border-border bg-card px-6 py-8 sm:px-8 sm:py-10 text-center space-y-4">
      <h3 className="text-lg sm:text-xl font-semibold text-foreground">
        Try RunMat for free
      </h3>
      <p className="text-foreground text-[0.938rem] max-w-md mx-auto">
        Open the sandbox and start running MATLAB code in seconds. No account required.
      </p>
      <div className="flex flex-col sm:flex-row items-center justify-center gap-3 pt-1">
        <Button
          size="lg"
          asChild
          className="h-11 px-7 text-sm font-semibold rounded-none bg-[hsl(var(--brand))] text-white border-0 shadow-none hover:bg-[hsl(var(--brand))]/90"
        >
          <Link
            href="/sandbox"
            data-ph-capture-attribute-destination="sandbox"
            data-ph-capture-attribute-source={source}
            data-ph-capture-attribute-cta="launch-sandbox"
          >
            Launch the sandbox
          </Link>
        </Button>
        <Button variant="ghost" size="lg" asChild className="h-11 px-7 text-sm text-muted-foreground hover:text-foreground">
          <Link href={secondaryHref}>{secondaryLabel}</Link>
        </Button>
      </div>
    </div>
  );
}
