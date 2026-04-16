"use client";

import { useId, useMemo } from "react";
import useMeasure from "react-use-measure";
import { scaleLinear, scaleLog } from "@visx/scale";
import { Group } from "@visx/group";
import { LinePath, AreaClosed } from "@visx/shape";
import { AxisBottom, AxisLeft } from "@visx/axis";
import { GridRows } from "@visx/grid";
import { curveLinear } from "@visx/curve";
import { Circle } from "@visx/shape";
import { Text } from "@visx/text";

import { useTheme } from "next-themes";
import type { BenchmarkLineChartData, BenchmarkLinePoint } from "@/lib/marketing-benchmarks";
import { formatNumber } from "@/lib/number-format";

interface BenchmarkLineChartProps {
  data: BenchmarkLineChartData;
  height?: number;
}

const SERIES_COLORS_DARK = ["#7dd3fc", "#f1f5f9", "#a5b4fc", "#f472b6"];
const SERIES_COLORS_LIGHT = ["#2563eb", "#374151", "#6366f1", "#db2777"];

export function BenchmarkLineChart({ data, height = 320 }: BenchmarkLineChartProps) {
  const gradientId = useId();
  const { resolvedTheme } = useTheme();
  const isDark = resolvedTheme === "dark";
  const SERIES_COLORS = isDark ? SERIES_COLORS_DARK : SERIES_COLORS_LIGHT;
  const fg = isDark ? "rgba(255,255,255,0.9)" : "rgba(0,0,0,0.8)";
  const fgMuted = isDark ? "rgba(255,255,255,0.6)" : "rgba(0,0,0,0.5)";
  const gridStroke = isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)";
  const axisStroke = isDark ? "rgba(255,255,255,0.25)" : "rgba(0,0,0,0.4)";
  const dotFill = isDark ? "#f8fafc" : "#1e293b";
  const nonHighlightStroke = isDark ? "rgba(255,255,255,0.65)" : "rgba(0,0,0,0.5)";
  const dotNonHighlight = isDark ? "rgba(255,255,255,0.7)" : "rgba(0,0,0,0.5)";
  const [ref, bounds] = useMeasure();
  const width = bounds.width;
  const isCompact = width > 0 && width < 520;
  const effectiveHeight = isCompact ? 300 : height;
  const margin = isCompact ? { top: 48, right: 8, bottom: 62, left: 56 } : { top: 48, right: 0, bottom: 68, left: 64 };
  const innerWidth = Math.max(0, width - margin.left - margin.right);
  const innerHeight = Math.max(0, effectiveHeight - margin.top - margin.bottom);

  const isLogX = data.logX ?? false;
  const logXMin = data.logXMin;

  const allPoints = useMemo(
    () => data.series.flatMap((series) => series.points),
    [data.series]
  );
  const uniqueParams = useMemo(() => {
    const values = Array.from(new Set(allPoints.map((pt) => pt.param)));
    return values.sort((a, b) => a - b);
  }, [allPoints]);

  const rawXDomain = useMemo(() => {
    if (!allPoints.length) {
      return [0, 1];
    }
    const values = allPoints.map((pt) => pt.param);
    return [Math.min(...values), Math.max(...values)];
  }, [allPoints]);

  const xDomain = useMemo(() => {
    if (!isLogX) {
      return rawXDomain;
    }
    const minCandidate = logXMin ?? rawXDomain[0];
    const safeMin = Math.max(minCandidate, 1e-6);
    return [safeMin, rawXDomain[1]];
  }, [isLogX, logXMin, rawXDomain]);

  const yDomain = useMemo(() => {
    if (!data.series.length) {
      return [0, 1];
    }
    const values = data.series.flatMap((series) =>
      series.points.map((pt) => (series.impl === data.baselineImpl ? 1 : pt.speedup ?? 0))
    );
    const max = Math.max(...values, 1);
    return [0, Math.max(1.1, max * 1.15)];
  }, [data.series, data.baselineImpl]);

  const xScale = useMemo(() => {
    if (isLogX) {
      return scaleLog({
        domain: xDomain as [number, number],
        range: [0, innerWidth],
        nice: true,
        clamp: true,
      });
    }
    return scaleLinear({
      domain: xDomain,
      range: [0, innerWidth],
      nice: true,
    });
  }, [isLogX, xDomain, innerWidth]);
  const xTickValues = isLogX ? uniqueParams : undefined;
  const yScale = useMemo(
    () =>
      scaleLinear({
        domain: yDomain,
        range: [innerHeight, 0],
        nice: true,
      }),
    [yDomain, innerHeight]
  );


  return (
    <div ref={ref} className="w-full">
      {width === 0 ? (
        <div style={{ height: effectiveHeight }} />
      ) : (
          <svg width={width} height={effectiveHeight} role="img" aria-label="Benchmark line chart">
          <Group left={margin.left} top={margin.top}>
              <foreignObject x={0} y={isCompact ? -32 : -40} width={innerWidth} height={isCompact ? 40 : 40}>
                <div
                  className={`flex w-full items-center justify-center ${isCompact ? "gap-3 text-[11px]" : "h-10 gap-8 text-sm"}`}
                >
                  {data.series.map((series, idx) => {
                    const color =
                      series.impl === data.highlightImpl ? "#a855f7" : SERIES_COLORS[idx % SERIES_COLORS.length];
                    const dash = series.impl === data.baselineImpl ? "4 4" : undefined;
                    const lineStyle =
                      dash !== undefined
                        ? { borderBottom: `1px dashed ${color}` }
                        : { borderBottom: `2px solid ${color}` };
                    return (
                      <div
                        key={series.impl}
                        className={`flex items-center gap-2 text-foreground/80 ${series.impl === data.highlightImpl ? "font-semibold text-foreground" : ""}`}
                      >
                        <span className="inline-block w-8 shrink-0" style={lineStyle} />
                        <span className={series.impl === data.highlightImpl ? "text-foreground font-semibold" : undefined}>
                          {series.label}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </foreignObject>
            <GridRows
              scale={yScale}
              width={innerWidth}
              stroke={gridStroke}
              pointerEvents="none"
                tickValues={yScale.ticks(3)}
              />
            {data.series.map((series, idx) => {
              const pointValue = (point: BenchmarkLinePoint) =>
                series.impl === data.baselineImpl ? 1 : point.speedup ?? 0;
              const color =
                series.impl === data.highlightImpl ? `url(#${gradientId})` : SERIES_COLORS[idx % SERIES_COLORS.length];
              const strokeColor = series.impl === data.highlightImpl ? "#a855f7" : nonHighlightStroke;

              return (
                <Group key={series.impl}>
                  {series.impl === data.highlightImpl && (
                    <>
                      <defs>
                        <linearGradient id={gradientId} x1="0" x2="0" y1="0" y2="1">
                          <stop offset="0%" stopColor="#a855f7" stopOpacity={0.45} />
                          <stop offset="100%" stopColor="#2563eb" stopOpacity={0.05} />
                        </linearGradient>
                      </defs>
                      <AreaClosed<BenchmarkLinePoint>
                        data={series.points}
                        x={(d) => xScale(d.param) ?? 0}
                        y={(d) => yScale(pointValue(d)) ?? 0}
                        yScale={yScale}
                        curve={curveLinear}
                        fill={color}
                        opacity={0.45}
                      />
                    </>
                  )}
                  <LinePath<BenchmarkLinePoint>
                    data={series.points}
                    x={(d) => xScale(d.param) ?? 0}
                    y={(d) => yScale(pointValue(d)) ?? 0}
                    curve={curveLinear}
                    stroke={strokeColor}
                    strokeWidth={series.impl === data.highlightImpl ? 3 : 2}
                    strokeOpacity={series.impl === data.highlightImpl ? 1 : 0.6}
                    strokeDasharray={series.impl === data.baselineImpl ? "6 6" : undefined}
                  />
                  {series.points.map((point) => (
                    <Circle
                      key={`${series.impl}-${point.param}`}
                      cx={xScale(point.param)}
                      cy={yScale(pointValue(point))}
                      r={series.impl === data.highlightImpl ? 5 : 4}
                      fill={series.impl === data.highlightImpl ? dotFill : dotNonHighlight}
                      stroke={series.impl === data.highlightImpl ? "#a855f7" : "transparent"}
                      strokeWidth={2}
                      opacity={series.impl === data.highlightImpl ? 1 : 0.75}
                    />
                  ))}
                </Group>
              );
            })}

            <AxisLeft
              scale={yScale}
              tickStroke={axisStroke}
              stroke={axisStroke}
              numTicks={3}
                tickFormat={(value) => `${Number(value).toFixed(0)}×`}
              tickLabelProps={() => ({
                fill: fg,
                fontSize: isCompact ? 12 : 14,
                dx: isCompact ? -24 : -35,
                dy: 5,
              })}
                left={isCompact ? 4 : 10}
            />
            <AxisBottom
              top={innerHeight}
              scale={xScale}
              tickStroke={axisStroke}
              stroke={axisStroke}
                numTicks={xTickValues ? undefined : isCompact ? 4 : 5}
                tickValues={xTickValues}
              tickFormat={(value) => formatNumber(Number(value))}
              tickLabelProps={() => ({
                fill: fg,
                fontSize: isCompact ? 12 : 14,
                dy: isCompact ? 4 : 5,
                dx: isCompact ? -6 : -20,
                textAnchor: "middle",
              })}
                tickTransform={isCompact ? "translate(-6, 0)" : "translate(-10, 0)"}
                hideTicks
                left={isCompact ? 4 : 10}
            />
            <Text
              x={-innerHeight / 2}
                y={isCompact ? -44 : -50}
              transform="rotate(-90)"
              textAnchor="middle"
                fill={fg}
                fontSize={isCompact ? 12 : 14}
            >
                x faster than NumPy
            </Text>
            <Text
              x={innerWidth / 2}
                y={innerHeight + (isCompact ? 54 : 60)}
              textAnchor="middle"
                fill={fg}
                fontSize={isCompact ? 12 : 14}
            >
              {data.paramLabel ?? data.paramKey}
              </Text>
          </Group>
        </svg>
      )}
    </div>
  );
}

export default BenchmarkLineChart;


