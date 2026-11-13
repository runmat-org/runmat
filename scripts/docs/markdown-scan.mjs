// Shared scanner for the FEA docs tree (docs/simulation). It reads raw Markdown the way the
// website and the agent do: front matter, fenced blocks with their info string, links and headings.
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";

/** Every Markdown file under `root`, as paths relative to `root`, sorted. `examples/` holds study packages, not pages. */
export function markdownFiles(root, { skip = ["examples", ".artifacts", "cases"] } = {}) {
    const files = [];
    (function walk(dir) {
        for (const name of readdirSync(dir).sort()) {
            const path = join(dir, name);
            if (statSync(path).isDirectory()) {
                if (!skip.includes(name)) walk(path);
            }
            else if (name.endsWith(".md")) files.push(relative(root, path).replaceAll("\\", "/"));
        }
    })(root);
    return files;
}

/**
 * Front matter in the YAML subset the docs use: `key: value`, `key: [a, b]` and block lists (`- item`).
 * Returns { data, body, bodyLine } where bodyLine is the 1-based line where the body starts, or data = null.
 */
export function parseFrontMatter(text) {
    if (!text.startsWith("---\n")) return { data: null, body: text, bodyLine: 1 };
    const end = text.indexOf("\n---\n", 4);
    if (end === -1) throw new Error("Front matter is not closed with ---");
    const lines = text.slice(4, end).split("\n");
    const data = {};
    let listKey = null;
    for (const [index, line] of lines.entries()) {
        if (!line.trim() || line.trimStart().startsWith("#")) continue;
        const item = line.match(/^\s+-\s+(.*)$/);
        if (item && listKey) { data[listKey].push(scalar(item[1])); continue; }
        const pair = line.match(/^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$/);
        if (!pair) throw new Error(`Front matter line ${index + 2} is not \`key: value\``);
        const [, key, raw] = pair;
        if (raw === "") { data[key] = []; listKey = key; continue; }
        listKey = null;
        data[key] = raw.startsWith("[") && raw.endsWith("]")
            ? raw.slice(1, -1).split(",").map(part => scalar(part.trim())).filter(part => part !== "")
            : scalar(raw);
    }
    const body = text.slice(end + 5);
    return { data, body, bodyLine: text.slice(0, end + 5).split("\n").length };
}

function scalar(raw) {
    const quoted = raw.match(/^"(.*)"$/) ?? raw.match(/^'(.*)'$/);
    return quoted ? quoted[1] : raw;
}

/** Fenced code blocks: { info, language, attributes, code, line } (line = 1-based fence line). */
export function fencedBlocks(text) {
    const blocks = [];
    const lines = text.split("\n");
    let open = null;
    for (const [index, line] of lines.entries()) {
        const fence = line.match(/^(\s*)(`{3,}|~{3,})(.*)$/);
        if (!open && fence) {
            open = { marker: fence[2], info: fence[3].trim(), line: index + 1, code: [] };
            continue;
        }
        if (open && fence && fence[2].startsWith(open.marker) && fence[3].trim() === "") {
            const [language = "", ...rest] = open.info.split(/\s+/).filter(Boolean);
            const attributes = {};
            for (const token of rest) {
                const [key, value] = token.split("=");
                attributes[key] = value ?? true;
            }
            blocks.push({ info: open.info, language, attributes, code: open.code.join("\n") + "\n", line: open.line });
            open = null;
            continue;
        }
        if (open) open.code.push(line);
    }
    if (open) throw new Error(`Unclosed code fence at line ${open.line}`);
    return blocks;
}

/** Lines outside fenced code, with their 1-based numbers; inline code spans are blanked. */
export function proseLines(text, { blankCode = true } = {}) {
    const out = [];
    let fence = null;
    for (const [index, line] of text.split("\n").entries()) {
        const marker = line.match(/^\s*(`{3,}|~{3,})/);
        if (marker) {
            if (!fence) fence = marker[1];
            else if (marker[1].startsWith(fence)) fence = null;
            continue;
        }
        if (!fence) out.push({ line: index + 1, text: blankCode ? line.replace(/`[^`]*`/g, match => " ".repeat(match.length)) : line });
    }
    return out;
}

/** Markdown links `[text](target)` in prose, excluding images. */
export function links(text) {
    const found = [];
    for (const { line, text: prose } of proseLines(text)) {
        for (const match of prose.matchAll(/(!?)\[((?:[^\[\]]|\[[^\]]*\])*)\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g)) {
            if (match[1] === "!") continue;
            found.push({ line, text: match[2], target: match[3] });
        }
    }
    return found;
}

/** ATX headings outside code, with the anchor the website gives them. */
export function headings(text) {
    return proseLines(text, { blankCode: false }).flatMap(({ line, text: prose }) => {
        const match = prose.match(/^(#{1,6})\s+(.*?)\s*#*\s*$/);
        return match ? [{ line, depth: match[1].length, text: match[2], anchor: headingAnchor(match[2]) }] : [];
    });
}

/** Same rule as website/lib/utils.ts slugifyHeading, after removing pending flags and link syntax. */
export function headingAnchor(markdown) {
    const text = markdown
        .replace(/\s*\[[^\]]*\]\(pending:[^)]+\)/g, "")
        .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
        .replace(/[`*]/g, "");
    return text.toLowerCase().replace(/[^a-z0-9\s-]/g, "-").trim().replace(/\s+/g, "-").replace(/-+/g, "-").replace(/^-|-$/g, "");
}

export const readText = path => readFileSync(path, "utf8");
