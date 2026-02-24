"use client";

import { useMemo } from "react";
import { motion } from "motion/react";
import { cn } from "@/lib/utils";
import type { ComplexEigenvalue } from "./EigenSolver";

const VIEW_MIN = -5;
const VIEW_MAX = 5;
const VIEW_RANGE = VIEW_MAX - VIEW_MIN;
const PAD = 0.6;
const DOT_R = 0.2;

function clampView(v: number): number {
  return Math.max(VIEW_MIN, Math.min(VIEW_MAX, v));
}

function toSvg(re: number, im: number): { x: number; y: number } {
  return {
    x: clampView(re),
    y: clampView(-im),
  };
}

export interface ComplexPlaneMotionProps {
  lambda1: ComplexEigenvalue;
  lambda2: ComplexEigenvalue;
  isComplexPair: boolean;
  className?: string;
}

const TRANSITION = { type: "tween" as const, duration: 0.2, ease: "easeOut" as const };

export function ComplexPlaneMotion({ lambda1, lambda2, isComplexPair, className }: ComplexPlaneMotionProps) {
  const pos1 = useMemo(() => toSvg(lambda1.re, lambda1.im), [lambda1.re, lambda1.im]);
  const repeated = Math.abs(lambda1.re - lambda2.re) < 1e-10 && Math.abs(lambda1.im - lambda2.im) < 1e-10;
  const pos2 = useMemo(() => {
    const p = toSvg(lambda2.re, lambda2.im);
    if (repeated) return { x: p.x + 0.15, y: p.y + 0.15 };
    return p;
  }, [lambda2.re, lambda2.im, repeated]);

  const dotColor = isComplexPair ? "#a78bfa" : "#3b82f6";

  const vbMin = VIEW_MIN - PAD;
  const vbSize = VIEW_RANGE + 2 * PAD;

  return (
    <div className={cn("flex flex-col items-center", className)}>
      <svg
        viewBox={`${vbMin} ${vbMin} ${vbSize} ${vbSize}`}
        className="w-full max-w-[500px] aspect-square border border-border rounded-lg bg-background"
        style={{ overflow: "hidden" }}
        aria-label="Complex plane with eigenvalue positions (Motion)"
      >
        <defs>
          <linearGradient id="plane-left-m" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="rgb(34 197 94 / 0.08)" />
            <stop offset="100%" stopColor="transparent" />
          </linearGradient>
          <linearGradient id="plane-right-m" x1="100%" y1="0%" x2="0%" y2="0%">
            <stop offset="0%" stopColor="rgb(239 68 68 / 0.08)" />
            <stop offset="100%" stopColor="transparent" />
          </linearGradient>
        </defs>
        {Array.from({ length: VIEW_RANGE + 1 }, (_, i) => VIEW_MIN + i).map((v) => (
          <g key={`grid-m-${v}`}>
            <line x1={v} y1={VIEW_MIN} x2={v} y2={VIEW_MAX} stroke="currentColor" strokeOpacity="0.12" strokeWidth="0.02" />
            <line x1={VIEW_MIN} y1={v} x2={VIEW_MAX} y2={v} stroke="currentColor" strokeOpacity="0.12" strokeWidth="0.02" />
          </g>
        ))}
        <rect x={VIEW_MIN} y={VIEW_MIN} width={0 - VIEW_MIN} height={VIEW_RANGE} fill="url(#plane-left-m)" />
        <rect x={0} y={VIEW_MIN} width={VIEW_MAX} height={VIEW_RANGE} fill="url(#plane-right-m)" />
        <line x1={0} y1={VIEW_MIN} x2={0} y2={VIEW_MAX} stroke="currentColor" strokeOpacity="0.4" strokeWidth="0.04" />
        <line x1={VIEW_MIN} y1={0} x2={VIEW_MAX} y2={0} stroke="currentColor" strokeOpacity="0.4" strokeWidth="0.04" />
        <text x={VIEW_MAX + 0.15} y={0.25} fontSize="0.4" fill="currentColor" opacity={0.8}>Re</text>
        <text x={0.15} y={VIEW_MIN - 0.15} fontSize="0.4" fill="currentColor" opacity={0.8}>Im</text>

        <motion.circle
          r={DOT_R}
          fill={dotColor}
          stroke="currentColor"
          strokeOpacity="0.3"
          strokeWidth="0.03"
          animate={{ cx: pos1.x, cy: pos1.y }}
          transition={TRANSITION}
          style={{ filter: "drop-shadow(0 0 2px rgba(0,0,0,0.3))" }}
        />
        <motion.circle
          r={DOT_R}
          fill={dotColor}
          stroke="currentColor"
          strokeOpacity="0.3"
          strokeWidth="0.03"
          animate={{ cx: pos2.x, cy: pos2.y }}
          transition={TRANSITION}
          style={{ filter: "drop-shadow(0 0 2px rgba(0,0,0,0.3))" }}
        />
      </svg>
      <p className="mt-1 text-xs text-muted-foreground">Complex plane (Motion)</p>
    </div>
  );
}
