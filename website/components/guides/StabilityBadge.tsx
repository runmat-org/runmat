"use client";

import { motion } from "motion/react";
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

export interface StabilityBadgeProps {
  status: StabilityStatus;
  className?: string;
}

export function StabilityBadge({ status, className }: StabilityBadgeProps) {
  const { label, bg } = STATUS_CONFIG[status];

  return (
    <motion.span
      className={cn(
        "inline-flex items-center rounded-md border border-transparent px-2 py-1 text-xs font-medium text-white",
        className
      )}
      animate={{ backgroundColor: bg }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      role="status"
      aria-live="polite"
    >
      {label}
    </motion.span>
  );
}
