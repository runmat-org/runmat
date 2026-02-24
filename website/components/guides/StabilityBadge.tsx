"use client";

import { useEffect, useRef } from "react";
import anime from "animejs";
import { cn } from "@/lib/utils";
import type { StabilityStatus } from "./EigenSolver";

const STATUS_CONFIG: Record<
  StabilityStatus,
  { label: string; bg: string }
> = {
  stable: { label: "STABLE", bg: "#22c55e" },
  unstable: { label: "UNSTABLE", bg: "#ef4444" },
  marginal: { label: "MARGINALLY STABLE", bg: "#eab308" },
};

const ANIM_DURATION = 300;
const EASING = "easeOutQuad";

export interface StabilityBadgeProps {
  status: StabilityStatus;
  className?: string;
}

export function StabilityBadge({ status, className }: StabilityBadgeProps) {
  const elRef = useRef<HTMLSpanElement>(null);
  const { label, bg } = STATUS_CONFIG[status];

  useEffect(() => {
    const el = elRef.current;
    if (!el) return;
    const anim = anime({
      targets: el,
      backgroundColor: bg,
      duration: ANIM_DURATION,
      easing: EASING,
    });
    return () => anim.pause();
  }, [bg]);

  return (
    <span
      ref={elRef}
      className={cn(
        "inline-flex items-center rounded-md border border-transparent px-2 py-1 text-xs font-medium text-white transition-[color,box-shadow]",
        className
      )}
      style={{ backgroundColor: bg }}
      role="status"
      aria-live="polite"
    >
      {label}
    </span>
  );
}
