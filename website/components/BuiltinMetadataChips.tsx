import Link from 'next/link';
import { Badge } from '@/components/ui/badge';
import { BuiltinMetadata } from '@/lib/builtins';
import { Layers, Zap } from 'lucide-react';

interface BuiltinMetadataChipsProps {
  metadata: BuiltinMetadata;
  categoryAnchor?: string;
}

export function BuiltinMetadataChips({ metadata, categoryAnchor }: BuiltinMetadataChipsProps) {
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
        <Badge
          variant="outline"
          className="px-3 py-1.5 h-auto border-border/50 bg-muted"
        >
          <div className="flex items-center gap-2">
            <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">GPU</span>
            <Zap className="h-3.5 w-3.5 text-muted-foreground" />
            <span className="font-medium text-foreground">Yes</span>
          </div>
        </Badge>
      )}

      {otherBadges.map((badge) => (
        <Badge
          key={badge}
          variant="outline"
          className="px-3 py-1.5 h-auto border-border/50 bg-muted font-medium"
        >
          {badge}
        </Badge>
      ))}
    </div>
  );
}
