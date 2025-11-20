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
  const margin = { top: 48, right: 0, bottom: 68, left: 64 };
  const innerWidth = Math.max(0, width - margin.left - margin.right);
  const innerHeight = Math.max(0, height - margin.top - margin.bottom);

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
              <foreignObject x={0} y={-40} width={innerWidth} height={40}>
                <div className="flex h-10 w-full items-center justify-center gap-8">
                  {data.series.map((series, idx) => {
                    const color =
                      series.impl === data.highlightImpl ? "#a855f7" : SERIES_COLORS[idx % SERIES_COLORS.length];
                    const dash = series.impl === data.baselineImpl ? "4 4" : undefined;
                    const lineStyle =
                      dash !== undefined
                        ? { borderBottom: `1px dashed ${color}` }
                        : { borderBottom: `2px solid ${color}` };
                    return (
                      <div key={series.impl} className="flex items-center justify-center gap-2 text-sm text-white/80">
                        <span className="inline-block w-8" style={lineStyle} />
                        <span className={series.impl === data.highlightImpl ? "text-white font-semibold" : undefined}>
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
              stroke="rgba(255,255,255,0.08)"
              pointerEvents="none"
                tickValues={yScale.ticks(3)}
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
                tickFormat={(value) => `${Number(value).toFixed(0)}×`}
              tickLabelProps={() => ({
                fill: "rgba(255,255,255,0.95)",
                fontSize: 14,
                dx: -35,
                dy: 5,
              })}
                left={10}
            />
            <AxisBottom
              top={innerHeight}
              scale={xScale}
              tickStroke="rgba(255,255,255,0.2)"
              stroke="rgba(255,255,255,0.2)"
                numTicks={xTickValues ? undefined : 5}
                tickValues={xTickValues}
              tickFormat={(value) => formatNumber(Number(value))}
              tickLabelProps={() => ({
                fill: "rgba(255,255,255,0.9)",
                fontSize: 14,
                dy: 5,
                dx: -20,
              })}
                tickTransform="translate(-10, 0)"
                hideTicks
                left={10}
            />
            <Text
              x={-innerHeight / 2}
                y={-50}
              transform="rotate(-90)"
              textAnchor="middle"
                fill="rgba(255,255,255,0.9)"
                fontSize={14}
            >
                x faster than NumPy
            </Text>
            <Text
              x={innerWidth / 2}
                y={innerHeight + 60}
              textAnchor="middle"
                fill="rgba(255,255,255,0.9)"
                fontSize={14}
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


