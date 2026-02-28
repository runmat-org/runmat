export function typeColor(type: string) {
  switch (type) {
    case "docs":
      return { text: "#a78bfa", bg: "rgba(167,139,250,0.1)" };
    case "guides":
      return { text: "#60a5fa", bg: "rgba(96,165,250,0.1)" };
    case "blogs":
      return { text: "#fb923c", bg: "rgba(251,146,60,0.1)" };
    case "case-studies":
      return { text: "#4ade80", bg: "rgba(74,222,128,0.1)" };
    case "webinars":
      return { text: "#f472b6", bg: "rgba(244,114,182,0.1)" };
    case "benchmarks":
      return { text: "#22d3ee", bg: "rgba(34,211,238,0.1)" };
    default:
      return { text: "#9ca3af", bg: "rgba(255,255,255,0.05)" };
  }
}
