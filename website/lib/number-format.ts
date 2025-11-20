export function formatNumber(value: number | string): string {
  if (value === 0) return "";
  const num = typeof value === "number" ? value : Number(value);
  if (!Number.isNaN(num)) {
    if (Math.abs(num) >= 1_000_000_000) {
      return `${(num / 1_000_000_000).toFixed(1).replace(/\.0$/, "")}B`;
    }
    if (Math.abs(num) >= 1_000_000) {
      return `${(num / 1_000_000).toFixed(1).replace(/\.0$/, "")}M`;
    }
    if (Math.abs(num) >= 1_000) {
      return `${(num / 1_000).toFixed(1).replace(/\.0$/, "")}K`;
    }
    return Intl.NumberFormat("en-US", { maximumFractionDigits: 2 }).format(num);
  }
  if (typeof value === "string") {
    return value;
  }
  return "";
}


