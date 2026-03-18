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
    <div className="rounded-2xl border border-purple-500/20 bg-gradient-to-br from-purple-500/5 via-transparent to-blue-500/5 p-[1px]">
      <div className="rounded-2xl bg-background/80 backdrop-blur-sm px-4 py-8 sm:px-8 sm:py-10 text-center space-y-5">
        <h3 className="text-xl sm:text-2xl font-semibold text-foreground">
          Try RunMat — free, no sign-up
        </h3>
        <p className="text-muted-foreground text-sm sm:text-base max-w-md mx-auto">
          Start running MATLAB code immediately in your browser.
        </p>
        <div className="flex flex-col sm:flex-row items-center justify-center gap-3 pt-1">
          <Button
            size="lg"
            asChild
            className="h-11 px-7 text-sm font-semibold bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-lg border-0 transition-all duration-200 hover:shadow-purple-500/25 hover:scale-[1.02]"
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
    </div>
  );
}
