import Link from "next/link";
import { Button } from "@/components/ui/button";

export default function Hero() {
  return (
    <section className="w-full py-16 md:py-24 lg:py-24" id="hero">
      <div className="container mx-auto px-4 md:px-6">
        <div className="grid grid-cols-1 gap-10 lg:grid-cols-2 lg:items-center">
          <div className="flex flex-col space-y-6 text-left items-start">
            <h1 className="font-heading text-left leading-tight tracking-tight text-[clamp(2.6rem,4.8vw,4.25rem)] sm:text-[clamp(3rem,4vw,5rem)] lg:text-[clamp(3.25rem,3.6vw,5.25rem)]">
              Run math blazing fast
            </h1>
            <p className="max-w-[42rem] leading-relaxed text-muted-foreground text-base sm:text-lg">
              Write MATLAB-syntax code and run it with GPU acceleration, in your browser, on the desktop, or from the CLI. No license required.
            </p>
            <div className="flex flex-col sm:flex-row gap-3 w-full sm:w-auto">
              <Button
                size="lg"
                asChild
                className="h-12 px-8 text-base font-semibold bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-xl border-0 transition-all duration-200 hover:from-blue-600 hover:to-purple-700"
              >
                <Link href="/sandbox">Open Sandbox</Link>
              </Button>
              <Button variant="outline" size="lg" asChild className="h-12 px-8 text-base">
                <Link href="/download">Download</Link>
              </Button>
            </div>
          </div>

          <Link
            href="/sandbox"
            className="rounded-2xl border border-border overflow-hidden min-h-[360px] flex items-center justify-center bg-muted/40 focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
          >
            <video
              className="w-full h-auto min-h-[360px] object-cover"
              autoPlay
              muted
              loop
              playsInline
              aria-label="RunMat wave simulation demo"
            >
              <source src="https://web.runmatstatic.com/video/runmat-wave-simulation.mp4" type="video/mp4" />
            </video>
          </Link>
        </div>
      </div>
    </section>
  );
}
