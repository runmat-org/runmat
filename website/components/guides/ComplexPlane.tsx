"use client";

import { useEffect, useRef, useMemo } from "react";
import anime from "animejs";
import { cn } from "@/lib/utils";
import type { ComplexEigenvalue } from "./EigenSolver";

const VIEW_MIN = -5;
const VIEW_MAX = 5;
const VIEW_RANGE = VIEW_MAX - VIEW_MIN;
const PAD = 0.6;
const DOT_R = 0.2;
const ANIM_DURATION = 200;
const EASING = "easeOutQuad";

function clampView(v: number): number {
  return Math.max(VIEW_MIN, Math.min(VIEW_MAX, v));
}

/** Map (re, im) in math to (x, y) in SVG viewBox. Origin center, Im up. */
function toSvg(re: number, im: number): { x: number; y: number } {
  return {
    x: clampView(re),
    y: clampView(-im),
  };
}

export interface ComplexPlaneProps {
  lambda1: ComplexEigenvalue;
  lambda2: ComplexEigenvalue;
  /** True if complex conjugate pair (show purple), else blue */
  isComplexPair: boolean;
  className?: string;
}

export function ComplexPlane({ lambda1, lambda2, isComplexPair, className }: ComplexPlaneProps) {
  const circle1Ref = useRef<SVGCircleElement>(null);
  const circle2Ref = useRef<SVGCircleElement>(null);

  const pos1 = useMemo(() => toSvg(lambda1.re, lambda1.im), [lambda1.re, lambda1.im]);
  const repeated = Math.abs(lambda1.re - lambda2.re) < 1e-10 && Math.abs(lambda1.im - lambda2.im) < 1e-10;
  const pos2 = useMemo(() => {
    const p = toSvg(lambda2.re, lambda2.im);
    if (repeated) return { x: p.x + 0.15, y: p.y + 0.15 };
    return p;
  }, [lambda2.re, lambda2.im, repeated]);

  useEffect(() => {
    const targets = [
      { el: circle1Ref.current, ...pos1 },
      { el: circle2Ref.current, ...pos2 },
    ].filter((t) => t.el) as { el: SVGCircleElement; x: number; y: number }[];
    if (targets.length === 0) return;

    const anims = targets.map(({ el, x, y }) =>
      anime({
        targets: el,
        cx: x,
        cy: y,
        duration: ANIM_DURATION,
        easing: EASING,
      })
    );
    return () => anims.forEach((a) => a.pause());
  }, [pos1, pos2]);

  const dotColor = isComplexPair ? "#a78bfa" : "#3b82f6";

  const vbMin = VIEW_MIN - PAD;
  const vbSize = VIEW_RANGE + 2 * PAD;

  return (
    <div className={cn("flex flex-col items-center", className)}>
      <svg
        viewBox={`${vbMin} ${vbMin} ${vbSize} ${vbSize}`}
        className="w-full max-w-[500px] aspect-square border border-border rounded-lg bg-background"
        style={{ overflow: "hidden" }}
        aria-label="Complex plane with eigenvalue positions"
      >
        <defs>
          <linearGradient id="plane-left" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="rgb(34 197 94 / 0.08)" />
            <stop offset="100%" stopColor="transparent" />
          </linearGradient>
          <linearGradient id="plane-right" x1="100%" y1="0%" x2="0%" y2="0%">
            <stop offset="0%" stopColor="rgb(239 68 68 / 0.08)" />
            <stop offset="100%" stopColor="transparent" />
          </linearGradient>
        </defs>
        {/* Grid */}
        {Array.from({ length: VIEW_RANGE + 1 }, (_, i) => VIEW_MIN + i).map((v) => (
          <g key={`grid-${v}`}>
            <line
              x1={v}
              y1={VIEW_MIN}
              x2={v}
              y2={VIEW_MAX}
              stroke="currentColor"
              strokeOpacity="0.12"
              strokeWidth="0.02"
            />
            <line
              x1={VIEW_MIN}
              y1={v}
              x2={VIEW_MAX}
              y2={v}
              stroke="currentColor"
              strokeOpacity="0.12"
              strokeWidth="0.02"
            />
          </g>
        ))}
        {/* Half-plane tints */}
        <rect
          x={VIEW_MIN}
          y={VIEW_MIN}
          width={0 - VIEW_MIN}
          height={VIEW_RANGE}
          fill="url(#plane-left)"
        />
        <rect x={0} y={VIEW_MIN} width={VIEW_MAX - 0} height={VIEW_RANGE} fill="url(#plane-right)" />
        {/* Im axis */}
        <line
          x1={0}
          y1={VIEW_MIN}
          x2={0}
          y2={VIEW_MAX}
          stroke="currentColor"
          strokeOpacity="0.4"
          strokeWidth="0.04"
        />
        {/* Re axis */}
        <line
          x1={VIEW_MIN}
          y1={0}
          x2={VIEW_MAX}
          y2={0}
          stroke="currentColor"
          strokeOpacity="0.4"
          strokeWidth="0.04"
        />
        {/* Axis labels — placed in padding so they are not clipped */}
        <text x={VIEW_MAX + 0.15} y={0.25} fontSize="0.4" fill="currentColor" opacity={0.8}>
          Re
        </text>
        <text x={0.15} y={VIEW_MIN - 0.15} fontSize="0.4" fill="currentColor" opacity={0.8}>
          Im
        </text>
        {/* Eigenvalue dots */}
        <circle
          ref={circle1Ref}
          r={DOT_R}
          fill={dotColor}
          stroke="currentColor"
          strokeOpacity="0.3"
          strokeWidth="0.03"
          cx={pos1.x}
          cy={pos1.y}
          style={{ filter: "drop-shadow(0 0 2px rgba(0,0,0,0.3))" }}
        />
        <circle
          ref={circle2Ref}
          r={DOT_R}
          fill={dotColor}
          stroke="currentColor"
          strokeOpacity="0.3"
          strokeWidth="0.03"
          cx={pos2.x}
          cy={pos2.y}
          style={{ filter: "drop-shadow(0 0 2px rgba(0,0,0,0.3))" }}
        />
      </svg>
      <p className="mt-1 text-xs text-muted-foreground">Complex plane</p>
    </div>
  );
}
