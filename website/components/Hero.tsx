import Link from "next/link";
import { Button } from "@/components/ui/button";
import LazyVideo from "@/components/LazyVideo";

export default function Hero() {
  return (
    <section className="w-full py-16 md:py-24 lg:py-32" id="hero">
      <div className="container mx-auto px-4 md:px-6">
        <div className="grid grid-cols-1 gap-10 lg:grid-cols-2 lg:items-center">
          <div className="flex flex-col space-y-6 text-left items-start">
            <p className="font-bold text-left leading-tight tracking-tight text-[clamp(2.6rem,4.8vw,4.25rem)] sm:text-[clamp(3rem,4vw,5rem)] lg:text-[clamp(3.25rem,3.6vw,5.25rem)]" role="heading" aria-level={2}>
              Run math blazing fast
            </p>
            <p className="max-w-[42rem] leading-relaxed text-foreground text-[0.938rem] sm:text-base">
              MATLAB-syntax math on GPU, in your browser or from the CLI. With the built-in agent, sweep fifty parameter variations in the time it used to take for one.
            </p>
            <div className="flex flex-col sm:flex-row gap-3 w-full sm:w-auto">
              <Button
                size="lg"
                asChild
                className="h-12 px-8 text-base font-semibold rounded-none bg-[hsl(var(--brand))] hover:bg-[hsl(var(--brand))]/90 text-white border-0 shadow-none"
              >
                <Link href="/sandbox">Open Sandbox</Link>
              </Button>
              <Button variant="outline" size="lg" asChild className="h-12 px-8 text-base rounded-none">
                <Link href="/download">Download</Link>
              </Button>
            </div>
          </div>

          <Link
            href="/sandbox"
            className="group relative rounded-lg border border-border overflow-hidden min-h-[360px] flex items-center justify-center focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
            data-ph-capture-attribute-destination="sandbox"
            data-ph-capture-attribute-source="home-hero-video"
            data-ph-capture-attribute-cta="try-runmat-agent"
            aria-label="Open the RunMat sandbox"
          >
            <LazyVideo
              className="w-full h-auto min-h-[360px] object-cover"
              muted
              loop
              playsInline
              poster="https://web.runmatstatic.com/video/posters/clamp-agent-runmat.webp"
              aria-label="RunMat agent extending a clamped plate vibration simulation"
            >
              <source src="https://web.runmatstatic.com/video/clamp-agent-runmat.mp4" type="video/mp4" />
            </LazyVideo>
            <span className="hidden sm:flex absolute bottom-2 right-2 items-center gap-1 rounded-md bg-black/40 backdrop-blur-sm px-2 py-1 text-[10px] font-medium text-white/70 transition-colors group-hover:text-white group-hover:bg-black/60">
              <svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="m18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" /><polyline points="15 3 21 3 21 9" /><line x1="10" x2="21" y1="14" y2="3" /></svg>
              Open sandbox
            </span>
          </Link>
        </div>
      </div>
    </section>
  );
}
