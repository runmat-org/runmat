"use client";

import { useId } from "react";
import useMeasure from "react-use-measure";
import { scaleBand, scaleLinear } from "@visx/scale";
import { Group } from "@visx/group";
import { AxisBottom, AxisLeft } from "@visx/axis";
import { BarRounded } from "@visx/shape";
import { LinearGradient } from "@visx/gradient";
import { Text } from "@visx/text";

import { useTheme } from "next-themes";
import type { BenchmarkBarChartData } from "@/lib/marketing-benchmarks";

interface BenchmarkBarChartProps {
  data: BenchmarkBarChartData;
  height?: number;
}

export function BenchmarkBarChart({ data, height = 320 }: BenchmarkBarChartProps) {
  const gradientId = useId();
  const { resolvedTheme } = useTheme();
  const isDark = resolvedTheme === "dark";
  const fg = isDark ? "rgba(255,255,255,0.9)" : "rgba(0,0,0,0.8)";
  const axisStroke = isDark ? "rgba(255,255,255,0.35)" : "rgba(0,0,0,0.4)";
  const gridLine = isDark ? "rgba(255,255,255,0.15)" : "rgba(0,0,0,0.1)";
  const barMuted = isDark ? "rgba(255,255,255,0.28)" : "rgba(0,0,0,0.12)";
  const barHighlightStroke = isDark ? "rgba(255,255,255,0.35)" : "rgba(0,0,0,0.1)";
  const labelFill = isDark ? "white" : "#0f172a";
  const [ref, bounds] = useMeasure();
  const width = bounds.width;
  const isCompact = width > 0 && width < 460;
  const effectiveHeight = isCompact ? 300 : height;
  const margin = isCompact
    ? { top: 12, right: 8, bottom: 70, left: 48 }
    : { top: 12, right: 12, bottom: 58, left: 56 };

  const innerWidth = Math.max(0, width - margin.left - margin.right);
  const innerHeight = Math.max(0, effectiveHeight - margin.top - margin.bottom);

  const entries = data.entries;
  const maxSpeedup = Math.max(...entries.map((entry) => entry.speedup));
  const yDomainMax = maxSpeedup < 1 ? 1.05 : maxSpeedup * 1.15;

  const xScale = scaleBand<string>({
    domain: entries.map((entry) => entry.label),
    range: [0, innerWidth],
    padding: 0.35,
  });

  const yScale = scaleLinear<number>({
    domain: [0, yDomainMax],
    range: [innerHeight, 0],
    nice: true,
  });

  const axisLabel = "x faster than NumPy";

  const formatLabel = (label: string) => {
    if (!isCompact) {
      return label;
    }
    return label.replace(/^Python\s+/i, "");
  };

  return (
    <div ref={ref} className="w-full">
      {width === 0 ? (
        <div style={{ height: effectiveHeight }} />
      ) : (
          <svg width={width} height={effectiveHeight} role="img" aria-label="Benchmark bar chart">
          <defs>
            <LinearGradient id={gradientId} from="#5b8dff" to="#bb51ff" />
          </defs>
          <Group left={margin.left} top={margin.top}>
            {/* Baseline grid */}
            <line
              x1={0}
              x2={innerWidth}
              y1={yScale(1)}
              y2={yScale(1)}
              stroke={gridLine}
              strokeDasharray="4 6"
            />
            {entries.map((entry) => {
              const barWidth = xScale.bandwidth();
              const barHeight = innerHeight - yScale(entry.speedup);
              const x = xScale(entry.label) ?? 0;
              const y = yScale(entry.speedup);
              const isHighlight = entry.impl === data.highlightImpl;
              const fill = isHighlight ? `url(#${gradientId})` : barMuted;
              const stroke = isHighlight ? barHighlightStroke : "transparent";
              return (
                <Group key={entry.impl}>
                  <BarRounded
                    x={x}
                    y={y}
                    width={barWidth}
                    height={barHeight}
                    radius={14}
                    all
                    fill={fill}
                    stroke={stroke}
                    strokeWidth={isHighlight ? 1.5 : 0}
                    opacity={isHighlight ? 1 : 0.85}
                  />
                  {isHighlight && (
                    <Text
                      x={x + barWidth / 2}
                      y={y - 8}
                      textAnchor="middle"
                      className="text-sm font-semibold"
                      fill={labelFill}
                    >
                      {entry.speedup >= 9 ? `${entry.speedup.toFixed(0)}×` : `${entry.speedup.toFixed(1)}×`}
                    </Text>
                  )}
                </Group>
              );
            })}

            <AxisLeft
              scale={yScale}
              numTicks={3}
              tickFormat={(value) => `${Number(value).toFixed(0)}×`}
              tickStroke={axisStroke}
              stroke={axisStroke}
              tickLabelProps={() => ({
                fill: fg,
                fontSize: isCompact ? 12 : 14,
                dx: -10,
                dy: 4,
              })}
            />
            <Text
              x={-innerHeight / 2}
                y={isCompact ? -36 : -42}
              transform="rotate(-90)"
              textAnchor="middle"
                fill={fg}
                fontSize={isCompact ? 12 : 14}
            >
              {axisLabel}
            </Text>

            <AxisBottom
              top={innerHeight}
              scale={xScale}
              stroke={axisStroke}
              hideTicks
                tickFormat={(value) => formatLabel(String(value))}
              tickLabelProps={(value) => ({
                fill: fg,
                fontSize: isCompact ? 12 : 14,
                dy: isCompact ? 12 : 16,
                textAnchor: "middle",
                x: (xScale(value as string) ?? 0) + xScale.bandwidth() / 2,
              })}
            />
          </Group>
        </svg>
      )}
    </div>
  );
}

export default BenchmarkBarChart;


