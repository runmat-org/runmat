import Link from 'next/link';
import { Badge } from '@/components/ui/badge';
import { BuiltinMetadata } from '@/lib/builtins';
import { Layers, Zap } from 'lucide-react';

interface BuiltinMetadataChipsProps {
  metadata: BuiltinMetadata;
  categoryAnchor?: string;
  /** Same-page anchor for the GPU section (only set when the heading exists) */
  gpuSectionAnchor?: string;
}

export function BuiltinMetadataChips({ metadata, categoryAnchor, gpuSectionAnchor }: BuiltinMetadataChipsProps) {
  const otherBadges = metadata.badges.filter((badge) => badge !== 'GPU');
  const categoryHref = categoryAnchor
    ? `/docs/matlab-function-reference#${categoryAnchor}`
    : '/docs/matlab-function-reference';

  return (
    <div className="flex flex-wrap items-center gap-3 mb-6">
      <Link href={categoryHref}>
        <Badge variant="outline" className="px-3 py-1.5 h-auto border-border/50 bg-muted cursor-pointer hover:bg-accent hover:text-accent-foreground transition-colors">
          <div className="flex items-center gap-2">
            <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Category</span>
            <Layers className="h-3.5 w-3.5 text-muted-foreground" />
            <span className="font-medium text-foreground">{metadata.category}</span>
          </div>
        </Badge>
      </Link>

      {metadata.gpuSupport && (
        <div className="group relative">
          <Link href={gpuSectionAnchor ?? '/docs/accelerate/fusion-intro'}>
            <Badge
              variant="outline"
              className="px-3 py-1.5 h-auto border-border/50 bg-muted cursor-pointer hover:bg-accent hover:text-accent-foreground transition-colors"
            >
              <div className="flex items-center gap-2">
                <Zap className="h-3.5 w-3.5 text-muted-foreground" />
                <span className="font-medium text-foreground">Auto GPU</span>
              </div>
            </Badge>
          </Link>
          <div className="absolute left-1/2 -translate-x-1/2 top-full mt-2 w-64 rounded-md border border-border bg-popover p-3 text-xs text-popover-foreground shadow-md opacity-0 pointer-events-none group-hover:opacity-100 group-hover:pointer-events-auto transition-opacity z-50">
            <p className="mb-1.5 font-medium">RunMat automatically offloads this function to the GPU when it estimates a speedup, without requiring explicit gpuArray inputs.</p>
            <Link href="/docs/accelerate/fusion-intro" className="text-primary hover:underline">
              Learn more about Auto GPU &rarr;
            </Link>
          </div>
        </div>
      )}

      {otherBadges.map((badge) => (
        <div key={badge} className="group/badge relative">
          <Badge
            variant="outline"
            className="px-3 py-1.5 h-auto border-border/50 bg-muted font-medium"
          >
            {badge}
          </Badge>
          {badge === 'BLAS/LAPACK' && (
            <div className="absolute left-1/2 -translate-x-1/2 top-full mt-2 w-64 rounded-md border border-border bg-popover p-3 text-xs text-popover-foreground shadow-md opacity-0 pointer-events-none group-hover/badge:opacity-100 group-hover/badge:pointer-events-auto transition-opacity z-50">
              <p className="font-medium">This function is backed by optimized BLAS/LAPACK routines (via Apple Accelerate or OpenBLAS) for high-performance linear algebra.</p>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
