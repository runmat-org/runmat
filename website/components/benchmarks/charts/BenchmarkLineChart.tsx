"use client";

import { useId, useMemo } from "react";
import useMeasure from "react-use-measure";
import { scaleLinear } from "@visx/scale";
import { Group } from "@visx/group";
import { LinePath, AreaClosed } from "@visx/shape";
import { AxisBottom, AxisLeft } from "@visx/axis";
import { GridRows, GridColumns } from "@visx/grid";
import { curveMonotoneX } from "@visx/curve";
import { Circle } from "@visx/shape";
import { Text } from "@visx/text";

import type { BenchmarkLineChartData, BenchmarkLinePoint } from "@/lib/marketing-benchmarks";
import { formatNumber } from "@/lib/number-format";

interface BenchmarkLineChartProps {
  data: BenchmarkLineChartData;
  height?: number;
}

const SERIES_COLORS = ["#7dd3fc", "#f1f5f9", "#a5b4fc", "#f472b6"];

export function BenchmarkLineChart({ data, height = 320 }: BenchmarkLineChartProps) {
  const gradientId = useId();
  const [ref, bounds] = useMeasure();
  const width = bounds.width;
  const margin = { top: 16, right: 16, bottom: 68, left: 64 };
  const innerWidth = Math.max(0, width - margin.left - margin.right);
  const innerHeight = Math.max(0, height - margin.top - margin.bottom);

  const allPoints = useMemo(
    () => data.series.flatMap((series) => series.points),
    [data.series]
  );

  const xDomain = useMemo(() => {
    if (!allPoints.length) {
      return [0, 1];
    }
    const values = allPoints.map((pt) => pt.param);
    return [Math.min(...values), Math.max(...values)];
  }, [allPoints]);

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

  const xScale = useMemo(
    () =>
      scaleLinear({
        domain: xDomain,
        range: [0, innerWidth],
        nice: true,
      }),
    [xDomain, innerWidth]
  );
  const yScale = useMemo(
    () =>
      scaleLinear({
        domain: yDomain,
        range: [innerHeight, 0],
        nice: true,
      }),
    [yDomain, innerHeight]
  );

  const highlightSeries = data.series.find((series) => series.impl === data.highlightImpl);
  const highlightPoint = highlightSeries?.points[highlightSeries.points.length - 1];
  const highlightSpeed = highlightPoint?.speedup ?? 0;

  const annotationText =
    highlightPoint && typeof highlightSpeed === "number"
      ? `${highlightSpeed >= 9 ? highlightSpeed.toFixed(0) : highlightSpeed.toFixed(1)}× vs baseline`
      : undefined;

  return (
    <div ref={ref} className="w-full">
      {width === 0 ? (
        <div style={{ height }} />
      ) : (
        <svg width={width} height={height} role="img" aria-label="Benchmark line chart">
          <Group left={margin.left} top={margin.top}>
            <GridRows
              scale={yScale}
              width={innerWidth}
              stroke="rgba(255,255,255,0.08)"
              pointerEvents="none"
            />
            <GridColumns
              scale={xScale}
              height={innerHeight}
              stroke="rgba(255,255,255,0.04)"
              pointerEvents="none"
              tickValues={xScale.ticks(4)}
            />
            {data.series.map((series, idx) => {
              const pointValue = (point: BenchmarkLinePoint) =>
                series.impl === data.baselineImpl ? 1 : point.speedup ?? 0;
              const color =
                series.impl === data.highlightImpl ? `url(#${gradientId})` : SERIES_COLORS[idx % SERIES_COLORS.length];
              const strokeColor = series.impl === data.highlightImpl ? "#a855f7" : "rgba(255,255,255,0.65)";

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
                        curve={curveMonotoneX}
                        fill={color}
                        opacity={0.45}
                      />
                    </>
                  )}
                  <LinePath<BenchmarkLinePoint>
                    data={series.points}
                    x={(d) => xScale(d.param) ?? 0}
                    y={(d) => yScale(pointValue(d)) ?? 0}
                    curve={curveMonotoneX}
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
                      fill={series.impl === data.highlightImpl ? "#f8fafc" : "rgba(255,255,255,0.7)"}
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
              tickStroke="rgba(255,255,255,0.25)"
              stroke="rgba(255,255,255,0.25)"
              numTicks={3}
              tickFormat={(value) => `${Number(value).toFixed(1)}×`}
              tickLabelProps={() => ({
                fill: "rgba(255,255,255,0.95)",
                fontSize: 15,
                dx: -10,
                dy: 4,
              })}
            />
            <AxisBottom
              top={innerHeight}
              scale={xScale}
              tickStroke="rgba(255,255,255,0.2)"
              stroke="rgba(255,255,255,0.2)"
              numTicks={5}
              tickFormat={(value) => formatNumber(Number(value))}
              tickLabelProps={() => ({
                fill: "rgba(255,255,255,0.9)",
                fontSize: 13,
                dy: 12,
              })}
            />
            <Text
              x={-innerHeight / 2}
              y={-44}
              transform="rotate(-90)"
              textAnchor="middle"
              fill="rgba(255,255,255,0.6)"
              fontSize={16}
            >
              × faster than Python NumPy
            </Text>
            <Text
              x={innerWidth / 2}
              y={innerHeight + 54}
              textAnchor="middle"
              fill="rgba(255,255,255,0.6)"
              fontSize={16}
            >
              {data.paramLabel ?? data.paramKey}
            </Text>

            {highlightPoint && annotationText && (
              <Group>
                <line
                  x1={xScale(highlightPoint.param)}
                  y1={yScale(highlightSpeed)}
                  x2={(xScale(highlightPoint.param) ?? 0) + 80}
                  y2={(yScale(highlightSpeed) ?? 0) - 40}
                  stroke="#a855f7"
                  strokeWidth={1.5}
                  strokeDasharray="4 4"
                />
                <Circle
                  cx={(xScale(highlightPoint.param) ?? 0) + 80}
                  cy={(yScale(highlightSpeed) ?? 0) - 40}
                  r={4}
                  fill="#a855f7"
                />
                <foreignObject
                  x={(xScale(highlightPoint.param) ?? 0) + 88}
                  y={(yScale(highlightSpeed) ?? 0) - 64}
                  width={140}
                  height={64}
                >
                  <div className="rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs text-white shadow-lg backdrop-blur">
                    <p className="font-semibold">{annotationText}</p>
                    <p className="text-[11px] text-white/70">
                      {formatNumber(highlightPoint.rawParam)} {data.paramLabel ?? data.paramKey}
                    </p>
                  </div>
                </foreignObject>
              </Group>
            )}
          </Group>
        </svg>
      )}
    </div>
  );
}

export default BenchmarkLineChart;


