import { readFileSync } from "fs";
import { join } from "path";

const SITE_URL = "https://runmat.com";

function escapeXml(str: string): string {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

interface FeedEntry {
  title: string;
  description: string;
  slug: string;
}

function parseChangelog(md: string): FeedEntry[] {
  const entries: FeedEntry[] = [];
  const sections = md.split(/^## /m).slice(1);

  for (const section of sections) {
    const firstLine = section.split("\n")[0].trim();

    const linkMatch = firstLine.match(/^\[([^\]]+)\]/);
    const title = linkMatch ? linkMatch[1] : firstLine.replace(/---+/g, "").trim();

    const slug = title
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-|-$/g, "");

    const body = section.split("\n").slice(1).join("\n").trim();
    const bullets = body
      .split("\n")
      .filter((l) => l.startsWith("- "))
      .slice(0, 5)
      .map((l) => l.replace(/^- /, "").replace(/\[([^\]]+)\]\([^)]+\)/g, "$1"))
      .join(". ");

    entries.push({
      title,
      description: bullets || title,
      slug,
    });
  }

  return entries;
}

export async function GET() {
  const mdPath = join(process.cwd(), "..", "docs", "CHANGELOG.md");
  const md = readFileSync(mdPath, "utf-8");
  const entries = parseChangelog(md);

  const items = entries
    .map(
      (entry) => `    <item>
      <title>${escapeXml(entry.title)}</title>
      <link>${SITE_URL}/docs/changelog#${entry.slug}</link>
      <guid isPermaLink="false">changelog-${entry.slug}</guid>
      <description>${escapeXml(entry.description)}</description>
    </item>`
    )
    .join("\n");

  const rss = `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
  <channel>
    <title>RunMat Changelog</title>
    <link>${SITE_URL}/docs/changelog</link>
    <description>What's new across the RunMat runtime, cloud, and sandbox.</description>
    <language>en</language>
    <lastBuildDate>${new Date().toUTCString()}</lastBuildDate>
    <atom:link href="${SITE_URL}/feed" rel="self" type="application/rss+xml" />
${items}
  </channel>
</rss>`;

  return new Response(rss, {
    headers: {
      "Content-Type": "application/rss+xml; charset=utf-8",
      "Cache-Control": "s-maxage=3600, stale-while-revalidate",
    },
  });
}
