import { request as httpsRequest } from "node:https";
import { request as httpRequest } from "node:http";

const DEFAULT_URLS = [
  "/docs",
  "/docs/getting-started",
  "/docs/how-it-works",
  "/docs/fusion-guide",
  "/docs/elements-of-matlab",
  "/docs/matlab-function-reference",
  "/docs/search?q=sin",
  "/docs/cli",
  "/docs/configuration",
  "/docs/architecture",
  "/docs/reference/builtins",
  "/docs/reference/builtins/sin",
  "/docs/reference/builtins/plot",
  "/docs/reference/builtins/gather",
];

function fetch(url, { userAgent = "facebookexternalhit/1.1" } = {}) {
  return new Promise((resolve, reject) => {
    const u = new URL(url);
    const requestFn = u.protocol === "http:" ? httpRequest : httpsRequest;
    const req = requestFn(
      {
        protocol: u.protocol,
        hostname: u.hostname,
        port: u.port || (u.protocol === "https:" ? 443 : 80),
        path: `${u.pathname}${u.search}`,
        method: "GET",
        headers: {
          "user-agent": userAgent,
        },
      },
      (res) => {
        const chunks = [];
        res.on("data", (c) => chunks.push(c));
        res.on("end", () => {
          const body = Buffer.concat(chunks);
          resolve({
            status: res.statusCode ?? 0,
            headers: res.headers,
            body,
          });
        });
      }
    );
    req.on("error", reject);
    req.end();
  });
}

function extractMeta(html) {
  const head = html.split("</head>", 1)[0] ?? "";
  const metaTags = head.match(/<meta[^>]+?>/gi) ?? [];
  const linkTags = head.match(/<link[^>]+?>/gi) ?? [];

  const getAttr = (tag, attr) => {
    const m = tag.match(new RegExp(`${attr}=(\"|')(.*?)(\\1)`, "i"));
    return m ? m[2] : null;
  };

  const metaBy = (key, val) => {
    for (const tag of metaTags) {
      const k = getAttr(tag, key);
      if (k && k.toLowerCase() === val.toLowerCase()) {
        return getAttr(tag, "content");
      }
    }
    return null;
  };

  const linkRel = (rel) => {
    for (const tag of linkTags) {
      const r = getAttr(tag, "rel");
      if (r && r.toLowerCase() === rel.toLowerCase()) {
        return getAttr(tag, "href");
      }
    }
    return null;
  };

  return {
    canonical: linkRel("canonical"),
    "og:title": metaBy("property", "og:title"),
    "og:description": metaBy("property", "og:description"),
    "og:url": metaBy("property", "og:url"),
    "og:type": metaBy("property", "og:type"),
    "og:image": metaBy("property", "og:image"),
    "twitter:card": metaBy("name", "twitter:card"),
    "twitter:title": metaBy("name", "twitter:title"),
    "twitter:description": metaBy("name", "twitter:description"),
    "twitter:image": metaBy("name", "twitter:image"),
  };
}

function missingFields(meta) {
  const required = [
    "og:title",
    "og:description",
    "og:url",
    "og:type",
    "og:image",
    "twitter:card",
    "twitter:title",
    "twitter:description",
    "twitter:image",
  ];
  return required.filter((k) => !meta[k]);
}

async function main() {
  const base = process.argv[2] ?? "https://runmat.org";
  const baseUrl = new URL(base);
  const baseOrigin = baseUrl.origin;

  let failures = 0;
  for (const p of DEFAULT_URLS) {
    const url = new URL(p, baseUrl).toString();
    // eslint-disable-next-line no-console
    console.log(`\\n== ${url} ==`);

    const res = await fetch(url);
    if (res.status < 200 || res.status >= 400) {
      failures++;
      // eslint-disable-next-line no-console
      console.log(`FETCH_ERROR status=${res.status}`);
      continue;
    }

    const ct = String(res.headers["content-type"] ?? "");
    if (!ct.includes("text/html")) {
      failures++;
      // eslint-disable-next-line no-console
      console.log(`UNEXPECTED_CONTENT_TYPE ${ct}`);
      continue;
    }

    const html = res.body.toString("utf-8");
    const meta = extractMeta(html);
    const missing = missingFields(meta);

    // eslint-disable-next-line no-console
    console.log("canonical:", meta.canonical ?? "NONE");
    // eslint-disable-next-line no-console
    console.log("og:title:", meta["og:title"] ?? "NONE");
    // eslint-disable-next-line no-console
    console.log("og:url:", meta["og:url"] ?? "NONE");
    // eslint-disable-next-line no-console
    console.log("og:image:", meta["og:image"] ?? "NONE");

    if (missing.length) {
      failures++;
      // eslint-disable-next-line no-console
      console.log("MISSING:", missing.join(", "));
    }

    if (meta["og:image"]) {
      const imgUrl = (() => {
        // When auditing a local server, rewrite absolute production URLs to local origin
        // so we validate the images we just built.
        if (baseOrigin !== "https://runmat.org" && meta["og:image"].startsWith("https://runmat.org/")) {
          return meta["og:image"].replace("https://runmat.org", baseOrigin);
        }
        return meta["og:image"];
      })();

      const imgRes = await fetch(imgUrl, { userAgent: "Twitterbot/1.0" });
      const imgCt = String(imgRes.headers["content-type"] ?? "");
      if (imgRes.status !== 200 || !imgCt.includes("image/png")) {
        failures++;
        // eslint-disable-next-line no-console
        console.log(`OG_IMAGE_BAD status=${imgRes.status} content-type=${imgCt} url=${imgUrl}`);
      }
    }
  }

  if (failures) {
    // eslint-disable-next-line no-console
    console.error(`\\nOG audit failed: ${failures} issue(s).`);
    process.exit(1);
  }

  // eslint-disable-next-line no-console
  console.log("\\nOG audit OK.");
}

main().catch((err) => {
  // eslint-disable-next-line no-console
  console.error(err);
  process.exit(1);
});

