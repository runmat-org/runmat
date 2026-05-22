import { cn } from "@/lib/utils";

export type MediaTone = "muted" | "brand" | "surface";

interface MediaProps {
  label: string;
  note?: string;
  tone?: MediaTone;
  className?: string;
}

const toneClasses: Record<MediaTone, string> = {
  muted: "border-border bg-muted text-foreground",
  brand: "border-brand/30 bg-brand/15 text-foreground",
  surface: "border-border bg-card text-card-foreground",
};

export default function Media({
  label,
  note,
  tone = "muted",
  className,
}: MediaProps) {
  return (
    <div
      role="img"
      aria-label={label}
      className={cn(
        "flex min-h-[320px] w-full items-end overflow-hidden rounded-2xl border p-5",
        toneClasses[tone],
        className,
      )}
    >
      <div className="max-w-sm">
        <p className="text-sm font-semibold">{label}</p>
        {note ? <p className="mt-1 text-xs opacity-70">{note}</p> : null}
      </div>
    </div>
  );
}
