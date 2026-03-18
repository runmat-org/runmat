"use client";

import { useCallback, useMemo, useState } from "react";
import useMeasure from "react-use-measure";
import { scaleLinear } from "@visx/scale";
import { Group } from "@visx/group";
import { LinePath, AreaClosed } from "@visx/shape";
import { AxisBottom, AxisLeft } from "@visx/axis";
import { GridRows } from "@visx/grid";
import { curveLinear } from "@visx/curve";
import { localPoint } from "@visx/event";

interface TimeDomainChartProps {
  signal: number[];
  sampleRate: number;
  height?: number;
}

const AXIS_COLOR = "rgba(255,255,255,0.25)";
const LABEL_COLOR = "rgba(255,255,255,0.9)";
const GRID_COLOR = "rgba(255,255,255,0.08)";
const LINE_COLOR = "#7dd3fc";
const FILL_COLOR_TOP = "rgba(125,211,252,0.25)";
const FILL_COLOR_BOT = "rgba(125,211,252,0.02)";
const CROSSHAIR_COLOR = "rgba(255,255,255,0.35)";

interface HoverState {
  x: number;
  time: number;
  amplitude: number;
}

export function TimeDomainChart({ signal, sampleRate, height = 300 }: TimeDomainChartProps) {
  const [ref, bounds] = useMeasure();
  const width = bounds.width;
  const isCompact = width > 0 && width < 480;
  const margin = isCompact
    ? { top: 16, right: 8, bottom: 36, left: 44 }
    : { top: 16, right: 12, bottom: 40, left: 52 };
  const innerWidth = Math.max(0, width - margin.left - margin.right);
  const innerHeight = Math.max(0, height - margin.top - margin.bottom);

  const displayCount = Math.min(signal.length, 512);
  const displaySignal = useMemo(
    () => signal.slice(0, displayCount),
    [signal, displayCount]
  );

  const xDomain = useMemo(
    () => [0, (displayCount - 1) / sampleRate],
    [displayCount, sampleRate]
  );

  const yDomain = useMemo(() => {
    if (!displaySignal.length) return [-1, 1];
    let min = Infinity;
    let max = -Infinity;
    for (const v of displaySignal) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const pad = Math.max((max - min) * 0.1, 0.1);
    return [min - pad, max + pad];
  }, [displaySignal]);

  const xScale = useMemo(
    () => scaleLinear({ domain: xDomain, range: [0, innerWidth] }),
    [xDomain, innerWidth]
  );
  const yScale = useMemo(
    () => scaleLinear({ domain: yDomain, range: [innerHeight, 0], nice: true }),
    [yDomain, innerHeight]
  );

  const points = useMemo(
    () => displaySignal.map((v, i) => ({ t: i / sampleRate, v })),
    [displaySignal, sampleRate]
  );

  const [hover, setHover] = useState<HoverState | null>(null);
  const [hasHovered, setHasHovered] = useState(false);

  const gradientId = "td-area-grad";

  const handleMouseMove = useCallback(
    (event: React.MouseEvent<SVGRectElement>) => {
      if (!innerWidth) return;
      const point = localPoint(event);
      if (!point) return;
      const px = point.x - margin.left;
      const time = xScale.invert(px);
      const idx = Math.round(time * sampleRate);
      if (idx < 0 || idx >= displaySignal.length) {
        setHover(null);
        return;
      }
      setHover({ x: px, time: idx / sampleRate, amplitude: displaySignal[idx] });
    },
    [innerWidth, margin.left, xScale, sampleRate, displaySignal]
  );

  const handleMouseLeave = useCallback(() => setHover(null), []);

  return (
    <div className="rounded-lg border border-border bg-card p-2">
      <h2 className="mb-1 pl-1 text-xs font-medium text-muted-foreground">
        Time domain
      </h2>
      <div ref={ref} className="w-full">
        {width === 0 ? (
          <div style={{ height }} />
        ) : (
          <svg width={width} height={height} role="img" aria-label="Time-domain waveform">
            <defs>
              <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={FILL_COLOR_TOP} />
                <stop offset="100%" stopColor={FILL_COLOR_BOT} />
              </linearGradient>
            </defs>
            <Group left={margin.left} top={margin.top}>
              <GridRows
                scale={yScale}
                width={innerWidth}
                stroke={GRID_COLOR}
                numTicks={5}
                pointerEvents="none"
              />
              <AreaClosed
                data={points}
                x={(d) => xScale(d.t) ?? 0}
                y={(d) => yScale(d.v) ?? 0}
                yScale={yScale}
                curve={curveLinear}
                fill={`url(#${gradientId})`}
                pointerEvents="none"
              />
              <LinePath
                data={points}
                x={(d) => xScale(d.t) ?? 0}
                y={(d) => yScale(d.v) ?? 0}
                curve={curveLinear}
                stroke={LINE_COLOR}
                strokeWidth={1.5}
              />
              {hover && (
                <>
                  <line
                    x1={hover.x}
                    x2={hover.x}
                    y1={0}
                    y2={innerHeight}
                    stroke={CROSSHAIR_COLOR}
                    strokeWidth={1}
                    strokeDasharray="3 3"
                    pointerEvents="none"
                  />
                  <circle
                    cx={hover.x}
                    cy={yScale(hover.amplitude) ?? 0}
                    r={4}
                    fill={LINE_COLOR}
                    stroke="rgba(0,0,0,0.5)"
                    strokeWidth={1.5}
                    pointerEvents="none"
                  />
                  <rect
                    x={Math.min(hover.x + 8, innerWidth - 100)}
                    y={4}
                    width={92}
                    height={34}
                    rx={4}
                    fill="rgba(0,0,0,0.75)"
                    pointerEvents="none"
                  />
                  <text
                    x={Math.min(hover.x + 14, innerWidth - 94)}
                    y={18}
                    fill={LABEL_COLOR}
                    fontSize={10}
                    fontFamily="monospace"
                  >
                    {`t = ${(hover.time * 1000).toFixed(1)} ms`}
                  </text>
                  <text
                    x={Math.min(hover.x + 14, innerWidth - 94)}
                    y={32}
                    fill={LINE_COLOR}
                    fontSize={10}
                    fontFamily="monospace"
                  >
                    {`A = ${hover.amplitude.toFixed(3)}`}
                  </text>
                </>
              )}
              {!hasHovered && !hover && innerWidth > 200 && (
                <text
                  x={innerWidth - 4}
                  y={12}
                  textAnchor="end"
                  fill="rgba(255,255,255,0.4)"
                  fontSize={10}
                  pointerEvents="none"
                >
                  Hover to inspect
                </text>
              )}
              <rect
                x={0}
                y={0}
                width={innerWidth}
                height={innerHeight}
                fill="transparent"
                className="cursor-crosshair"
                onMouseMove={(e) => { if (!hasHovered) setHasHovered(true); handleMouseMove(e); }}
                onMouseLeave={handleMouseLeave}
              />
              <AxisLeft
                scale={yScale}
                stroke={AXIS_COLOR}
                tickStroke={AXIS_COLOR}
                numTicks={5}
                tickLabelProps={() => ({
                  fill: LABEL_COLOR,
                  fontSize: isCompact ? 10 : 11,
                  dx: isCompact ? -20 : -28,
                  dy: 4,
                })}
              />
              <AxisBottom
                top={innerHeight}
                scale={xScale}
                stroke={AXIS_COLOR}
                tickStroke={AXIS_COLOR}
                numTicks={isCompact ? 4 : 6}
                tickFormat={(v) => `${(Number(v) * 1000).toFixed(0)}`}
                tickLabelProps={() => ({
                  fill: LABEL_COLOR,
                  fontSize: isCompact ? 10 : 11,
                  dy: 4,
                  textAnchor: "middle",
                })}
              />
              <text
                x={innerWidth / 2}
                y={innerHeight + (isCompact ? 30 : 34)}
                textAnchor="middle"
                fill={LABEL_COLOR}
                fontSize={isCompact ? 10 : 11}
              >
                Time (ms)
              </text>
            </Group>
          </svg>
        )}
      </div>
    </div>
  );
}
