import { Badge } from '@/components/ui/badge';
import { BuiltinMetadata } from '@/lib/builtins';
import { Layers, Zap, Flame } from 'lucide-react';

interface BuiltinMetadataChipsProps {
  metadata: BuiltinMetadata;
}

export function BuiltinMetadataChips({ metadata }: BuiltinMetadataChipsProps) {
  return (
    <div className="flex flex-wrap items-center gap-3 mb-6">
      <Badge variant="outline" className="px-3 py-1.5 h-auto border-border/50 bg-muted/30">
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Category</span>
          <Layers className="h-3.5 w-3.5 text-muted-foreground" />
          <span className="font-medium text-foreground">{metadata.category}</span>
        </div>
      </Badge>
      
      {metadata.gpuSupport && (
        <Badge 
          variant="default" 
          className="px-3 py-1.5 h-auto"
        >
          <div className="flex items-center gap-2">
            <span className="text-xs font-medium text-muted-foreground/80 uppercase tracking-wide">GPU</span>
            <Zap className="h-3.5 w-3.5" />
            <span className="font-medium text-green-600 dark:text-green-400">Yes</span>
          </div>
        </Badge>
      )}
      
      <Badge variant="outline" className="px-3 py-1.5 h-auto border-border/50 bg-muted/30">
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Fusion</span>
          <Flame className="h-3.5 w-3.5 text-muted-foreground" />
          <span className="font-medium text-foreground">{metadata.fusion}</span>
        </div>
      </Badge>
    </div>
  );
}

