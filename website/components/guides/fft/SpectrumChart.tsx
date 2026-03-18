"use client";

import { useCallback, useMemo, useState } from "react";
import useMeasure from "react-use-measure";
import { scaleLinear } from "@visx/scale";
import { Group } from "@visx/group";
import { LinePath, AreaClosed } from "@visx/shape";
import { AxisBottom, AxisLeft } from "@visx/axis";
import { GridRows } from "@visx/grid";
import { curveLinear } from "@visx/curve";
import { Text } from "@visx/text";
import { localPoint } from "@visx/event";

interface SpectrumChartProps {
  spectrum: number[];
  sampleRate: number;
  height?: number;
}

const AXIS_COLOR = "rgba(255,255,255,0.25)";
const LABEL_COLOR = "rgba(255,255,255,0.9)";
const GRID_COLOR = "rgba(255,255,255,0.08)";
const LINE_COLOR = "#a78bfa";
const PEAK_COLOR = "#f472b6";
const FILL_COLOR_TOP = "rgba(167,139,250,0.22)";
const FILL_COLOR_BOT = "rgba(167,139,250,0.02)";
const CROSSHAIR_COLOR = "rgba(255,255,255,0.35)";
const PEAK_THRESHOLD_RATIO = 0.15;
const MAX_PEAKS = 6;

interface HoverState {
  x: number;
  freq: number;
  mag: number;
}

function findPeaks(spectrum: number[], sampleRate: number): { freq: number; mag: number }[] {
  const n = spectrum.length;
  const nyquist = Math.floor(n / 2);
  if (nyquist < 3) return [];

  const maxMag = Math.max(...spectrum.slice(1, nyquist));
  const threshold = maxMag * PEAK_THRESHOLD_RATIO;
  const peaks: { freq: number; mag: number; idx: number }[] = [];

  for (let i = 2; i < nyquist - 1; i++) {
    if (
      spectrum[i] > threshold &&
      spectrum[i] > spectrum[i - 1] &&
      spectrum[i] > spectrum[i + 1]
    ) {
      peaks.push({
        freq: (i * sampleRate) / n,
        mag: spectrum[i],
        idx: i,
      });
    }
  }

  peaks.sort((a, b) => b.mag - a.mag);
  return peaks.slice(0, MAX_PEAKS);
}

export function SpectrumChart({ spectrum, sampleRate, height = 300 }: SpectrumChartProps) {
  const [ref, bounds] = useMeasure();
  const width = bounds.width;
  const isCompact = width > 0 && width < 480;
  const margin = isCompact
    ? { top: 16, right: 8, bottom: 36, left: 44 }
    : { top: 16, right: 12, bottom: 40, left: 52 };
  const innerWidth = Math.max(0, width - margin.left - margin.right);
  const innerHeight = Math.max(0, height - margin.top - margin.bottom);

  const nyquist = Math.floor(spectrum.length / 2);
  const halfSpectrum = useMemo(
    () => spectrum.slice(0, nyquist),
    [spectrum, nyquist]
  );

  const freqStep = sampleRate / spectrum.length;

  const points = useMemo(
    () => halfSpectrum.map((mag, i) => ({ freq: i * freqStep, mag })),
    [halfSpectrum, freqStep]
  );

  const yMax = useMemo(() => {
    if (!halfSpectrum.length) return 1;
    const m = Math.max(...halfSpectrum.slice(1));
    return m > 0 ? m * 1.15 : 1;
  }, [halfSpectrum]);

  const xScale = useMemo(
    () => scaleLinear({ domain: [0, sampleRate / 2], range: [0, innerWidth] }),
    [sampleRate, innerWidth]
  );
  const yScale = useMemo(
    () => scaleLinear({ domain: [0, yMax], range: [innerHeight, 0], nice: true }),
    [yMax, innerHeight]
  );

  const peaks = useMemo(
    () => findPeaks(spectrum, sampleRate),
    [spectrum, sampleRate]
  );

  const [hover, setHover] = useState<HoverState | null>(null);
  const [hasHovered, setHasHovered] = useState(false);
  const gradientId = "sp-area-grad";

  const handleMouseMove = useCallback(
    (event: React.MouseEvent<SVGRectElement>) => {
      if (!innerWidth) return;
      const point = localPoint(event);
      if (!point) return;
      const px = point.x - margin.left;
      const freq = xScale.invert(px);
      const idx = Math.round(freq / freqStep);
      if (idx < 0 || idx >= halfSpectrum.length) {
        setHover(null);
        return;
      }
      setHover({ x: px, freq: idx * freqStep, mag: halfSpectrum[idx] });
    },
    [innerWidth, margin.left, xScale, freqStep, halfSpectrum]
  );

  const handleMouseLeave = useCallback(() => setHover(null), []);

  return (
    <div className="rounded-lg border border-border bg-card p-2">
      <h2 className="mb-1 pl-1 text-xs font-medium text-muted-foreground">
        Frequency spectrum
      </h2>
      <div ref={ref} className="w-full">
        {width === 0 ? (
          <div style={{ height }} />
        ) : (
          <svg width={width} height={height} role="img" aria-label="FFT magnitude spectrum">
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
                x={(d) => xScale(d.freq) ?? 0}
                y={(d) => yScale(d.mag) ?? 0}
                yScale={yScale}
                curve={curveLinear}
                fill={`url(#${gradientId})`}
                pointerEvents="none"
              />
              <LinePath
                data={points}
                x={(d) => xScale(d.freq) ?? 0}
                y={(d) => yScale(d.mag) ?? 0}
                curve={curveLinear}
                stroke={LINE_COLOR}
                strokeWidth={1.5}
              />
              {peaks.map((peak) => {
                const px = xScale(peak.freq) ?? 0;
                const py = yScale(peak.mag) ?? 0;
                return (
                  <g key={peak.freq}>
                    <circle cx={px} cy={py} r={3.5} fill={PEAK_COLOR} />
                    <Text
                      x={px}
                      y={py - 10}
                      textAnchor="middle"
                      fill={PEAK_COLOR}
                      fontSize={isCompact ? 9 : 10}
                      fontWeight={600}
                    >
                      {`${Math.round(peak.freq)} Hz`}
                    </Text>
                  </g>
                );
              })}
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
                    cy={yScale(hover.mag) ?? 0}
                    r={4}
                    fill={LINE_COLOR}
                    stroke="rgba(0,0,0,0.5)"
                    strokeWidth={1.5}
                    pointerEvents="none"
                  />
                  <rect
                    x={Math.min(hover.x + 8, innerWidth - 110)}
                    y={4}
                    width={102}
                    height={34}
                    rx={4}
                    fill="rgba(0,0,0,0.75)"
                    pointerEvents="none"
                  />
                  <text
                    x={Math.min(hover.x + 14, innerWidth - 104)}
                    y={18}
                    fill={LABEL_COLOR}
                    fontSize={10}
                    fontFamily="monospace"
                  >
                    {`f = ${hover.freq.toFixed(1)} Hz`}
                  </text>
                  <text
                    x={Math.min(hover.x + 14, innerWidth - 104)}
                    y={32}
                    fill={LINE_COLOR}
                    fontSize={10}
                    fontFamily="monospace"
                  >
                    {`|X| = ${hover.mag.toFixed(1)}`}
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
                tickFormat={(v) => {
                  const n = Number(v);
                  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
                  return n < 10 ? n.toFixed(1) : Math.round(n).toString();
                }}
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
                Frequency (Hz)
              </text>
            </Group>
          </svg>
        )}
      </div>
    </div>
  );
}
