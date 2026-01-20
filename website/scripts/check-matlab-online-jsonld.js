const fs = require("fs");
const path = require("path");

const filePath = path.join(__dirname, "..", "app", "run-matlab-online", "page.tsx");
const source = fs.readFileSync(filePath, "utf8");

const requiredSnippets = [
  "export const metadata",
  'canonical: "https://runmat.org/run-matlab-online"',
  '"@type": "BreadcrumbList"',
  '"@type": "SoftwareApplication"',
  '"@type": "FAQPage"',
  '"@type": "HowTo"',
  'type="application/ld+json"',
];

const missing = requiredSnippets.filter(snippet => !source.includes(snippet));

if (missing.length > 0) {
  console.error("Matlab-online JSON-LD/metadata check failed. Missing:");
  missing.forEach(snippet => console.error(`- ${snippet}`));
  process.exit(1);
}

console.log("Matlab-online JSON-LD/metadata check passed.");


