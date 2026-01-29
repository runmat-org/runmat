use runmat_builtins::{BuiltinDoc, BuiltinFunction};
use std::fmt::Write as _;

use crate::core::builtins_json;

/// Helpers for documentation linking/formatting.
const RUNMAT_DOC_BASE_URL: &str = "https://runmat.org/docs/reference/builtins/";

/// Build a documentation URL for a builtin slug (website uses lowercase).
pub fn builtin_doc_url(name: &str) -> String {
    format!("{RUNMAT_DOC_BASE_URL}{}", name.to_ascii_lowercase())
}

/// Lookup a BuiltinDoc by name.
pub fn find_builtin_doc(name: &str) -> Option<&'static BuiltinDoc> {
    // Note: builtin_docs() allocates a Vec; acceptable given small cardinality.
    runmat_builtins::builtin_docs()
        .into_iter()
        .find(|d| d.name == name)
}

fn see_also_url(url: &str) -> String {
    // builtins-json uses "./foo" for other builtin links.
    if let Some(rest) = url.strip_prefix("./") {
        builtin_doc_url(rest)
    } else {
        url.to_string()
    }
}

fn normalize_markdown(mut s: String) -> String {
    // Monaco doesn't treat these specially; keep the fence but normalize the info string.
    s = s.replace("```matlab:runnable", "```matlab");
    s = s.replace("```matlab:run", "```matlab");

    // Some builtins-json descriptions encode multiple list items in a single bullet line:
    // "\n\n- `foo` ... - `bar` ... - `baz` ..."
    // This renders as a run-on in narrow hovers. Split " - `X`" into real list items
    // only when the string already contains a list.
    if s.contains("\n\n- ") {
        // Split only when it looks like a new bullet for an inline-code leading token.
        s = s.replace(" - `", "\n- `");
    }

    s
}

fn looks_like_call_syntax(seg: &str, name: &str) -> bool {
    let seg_trim = seg.trim();
    if seg_trim.is_empty() {
        return false;
    }
    let name_lc = name.to_ascii_lowercase();
    let seg_lc = seg_trim.to_ascii_lowercase();

    // Find the function name occurrence.
    let Some(name_pos) = seg_lc.find(&name_lc) else {
        return false;
    };

    // If there's an '=' it must be before the function name (e.g. "y = sin(x)").
    if let Some(eq_pos) = seg_trim.find('=') {
        if eq_pos > name_pos {
            return false;
        }
    }

    // After the function name, allow whitespace, then require '('
    let mut idx = name_pos + name_lc.len();
    let bytes = seg_trim.as_bytes();
    while idx < bytes.len() && bytes[idx].is_ascii_whitespace() {
        idx += 1;
    }
    idx < bytes.len() && bytes[idx] == b'('
}

fn first_backticked_segment(s: &str) -> Option<&str> {
    let mut rest = s;
    let start = rest.find('`')?;
    rest = &rest[start + 1..];
    let end = rest.find('`')?;
    Some(&rest[..end])
}

fn extract_syntax_from_behaviors(name: &str, behaviors: &[String]) -> Vec<String> {
    let name_lc = name.to_ascii_lowercase();
    let mut out: Vec<String> = Vec::new();

    for line in behaviors {
        // Extract backtick-delimited segments and keep those that look like call patterns.
        let mut rest = line.as_str();
        while let Some(start) = rest.find('`') {
            let after_start = &rest[start + 1..];
            let Some(end) = after_start.find('`') else { break };
            let seg = &after_start[..end];
            rest = &after_start[end + 1..];

            let seg_lc = seg.to_ascii_lowercase();
            let mentions_name = seg_lc.contains(&name_lc);
            if mentions_name
                && looks_like_call_syntax(seg, name)
                && !out.iter().any(|s| s.eq_ignore_ascii_case(seg))
            {
                out.push(seg.to_string());
            }
        }
    }

    out
}

fn render_builtin_hover_from_json(func: &BuiltinFunction, doc: builtins_json::BuiltinDocJson) -> String {
    let mut out = String::new();

    // Header (no typed signature yet): reliably show just `name(...)`.
    let _ = writeln!(out, "```runmat\n{}(...)\n```", func.name);

    // Prefer the builtins-json "description" as the lede when it contains a call form,
    // because it typically already explains the function more concretely than summary/category/keywords.
    let description = doc.description.as_deref().filter(|s| !s.trim().is_empty());
    let use_description_as_lede = description
        .and_then(first_backticked_segment)
        .is_some_and(|seg| looks_like_call_syntax(seg, func.name));

    if use_description_as_lede {
        let normalized = normalize_markdown(description.unwrap().to_string());
        let _ = writeln!(out, "{normalized}\n");
    } else {
        let title = doc.title.clone().unwrap_or_else(|| func.name.to_string());
        let summary = doc
            .summary
            .clone()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| func.description.to_string());

        if !summary.trim().is_empty() {
            let _ = writeln!(out, "**{title}** — {summary}\n");
        } else {
            let _ = writeln!(out, "**{title}**\n");
        }

        if let Some(category) = doc.category.as_deref().filter(|s| !s.trim().is_empty()) {
            let _ = writeln!(out, "**Category:** {category}\n");
        }
        if let Some(keywords) = doc.keywords.as_ref().filter(|k| !k.is_empty()) {
            let joined = keywords.join(", ");
            let _ = writeln!(out, "**Keywords:** {joined}\n");
        }

        if let Some(description) = description {
            let normalized = normalize_markdown(description.to_string());
            let _ = writeln!(out, "{normalized}\n");
        }
    }

    // Syntax (derived from behaviors or from explicit syntax field).
    let mut syntax: Vec<String> = Vec::new();
    if let Some(behaviors) = doc.behaviors.as_ref() {
        syntax = extract_syntax_from_behaviors(func.name, behaviors);
    }
    if !syntax.is_empty() {
        out.push_str("**Syntax**\n\n");
        for s in syntax {
            let _ = writeln!(out, "- `{s}`");
        }
        out.push('\n');
    }

    // Behaviors: include full list (no artificial filtering).
    if let Some(behaviors) = doc.behaviors.as_ref().filter(|b| !b.is_empty()) {
        out.push_str("**Key behaviors**\n\n");
        for b in behaviors {
            let normalized = normalize_markdown(b.clone());
            // If the behavior contains hard newlines, keep it as a paragraph to avoid
            // broken list indentation in Markdown renderers.
            if normalized.contains('\n') {
                let _ = writeln!(out, "- {}", normalized.lines().next().unwrap_or("").trim_end());
                for cont in normalized.lines().skip(1) {
                    let _ = writeln!(out, "  {}", cont.trim_end());
                }
            } else {
                let _ = writeln!(out, "- {normalized}");
            }
        }
        out.push('\n');
    }

    // Options (generic)
    if let Some(options) = doc.options.as_ref().filter(|o| !o.is_empty()) {
        out.push_str("**Options**\n\n");
        for o in options {
            let _ = writeln!(out, "- {o}");
        }
        out.push('\n');
    }
    // jsonencode_options (structured)
    if let Some(opts) = doc.jsonencode_options.as_ref().filter(|o| !o.is_empty()) {
        out.push_str("**Options**\n\n");
        for o in opts {
            let _ = writeln!(
                out,
                "- **{}** ({}, default: `{}`): {}",
                o.name, o.type_name, o.default, o.description
            );
        }
        out.push('\n');
    }

    // GPU section.
    if doc.gpu_support.is_some() || doc.gpu_residency.is_some() || doc.gpu_behavior.is_some() {
        out.push_str("**GPU**\n\n");
        if let Some(gpu) = doc.gpu_support.as_ref() {
            if let Some(notes) = gpu.notes.as_deref().filter(|s| !s.trim().is_empty()) {
                let notes = normalize_markdown(notes.to_string());
                let _ = writeln!(out, "- **Support**: {notes}");
            }
        }
        let gpu_residency_norm = doc
            .gpu_residency
            .as_deref()
            .filter(|s| !s.trim().is_empty())
            .map(|s| normalize_markdown(s.to_string()));

        if let Some(points) = doc.gpu_behavior.as_ref().filter(|p| !p.is_empty()) {
            for p in points {
                let p_norm = normalize_markdown(p.clone());
                // Avoid obvious duplication (gpu_behavior often repeats gpu_residency verbatim).
                if gpu_residency_norm.as_deref().is_some_and(|r| r.trim() == p_norm.trim()) {
                    continue;
                }
                let _ = writeln!(out, "- {p_norm}");
            }
        }
        if let Some(res) = gpu_residency_norm.as_deref() {
            let _ = writeln!(out, "\n{res}");
        }
        out.push('\n');
    }

    // Examples: include full set (no artificial filtering).
    if let Some(examples) = doc.examples.as_ref().filter(|e| !e.is_empty()) {
        out.push_str("**Examples**\n");
        for ex in examples {
            let _ = writeln!(out, "\n**{}**", ex.description.trim());
            out.push_str("```matlab\n");
            out.push_str(&normalize_markdown(ex.input.trim_end().to_string()));
            out.push_str("\n```\n");
            if let Some(output) = ex.output.as_deref().filter(|s| !s.trim().is_empty()) {
                out.push_str("\nOutput:\n```text\n");
                out.push_str(&normalize_markdown(output.trim_end().to_string()));
                out.push_str("\n```\n");
            }
        }
        out.push('\n');
    }

    // See also
    if let Some(links) = doc.links.as_ref().filter(|l| !l.is_empty()) {
        let mut rendered: Vec<String> = Vec::new();
        for l in links {
            let url = see_also_url(&l.url);
            rendered.push(format!("[{}]({})", l.label, url));
        }
        if !rendered.is_empty() {
            let _ = writeln!(out, "**See also**: {}", rendered.join(", "));
        }
    }

    // Docs link
    let slug = doc.title.unwrap_or_else(|| func.name.to_string());
    let _ = writeln!(out, "Docs: {}", builtin_doc_url(&slug));
    out
}

/// Render a Markdown hover string for a builtin function, enriched with metadata.
pub fn build_builtin_hover(func: &BuiltinFunction) -> String {
    if let Some(doc) = builtins_json::builtin_doc(func.name) {
        return render_builtin_hover_from_json(func, doc);
    }

    // Fallback: no builtins-json entry available.
    let mut out = String::new();
    let _ = writeln!(out, "```runmat\n{}(...)\n```", func.name);

    if !func.description.trim().is_empty() {
        let _ = writeln!(out, "**{}** — {}\n", func.name, func.description);
    }

    if !func.doc.is_empty() {
        out.push_str(func.doc);
        out.push_str("\n\n");
    }

    if let Some(meta) = find_builtin_doc(func.name) {
        if let Some(summary) = meta.summary {
            let _ = writeln!(out, "{summary}\n");
        }
        if let Some(category) = meta.category {
            let _ = writeln!(out, "**Category:** {category}");
        }
        if let Some(keywords) = meta.keywords {
            let _ = writeln!(out, "**Keywords:** {keywords}");
        }
        if let Some(errors) = meta.errors {
            let _ = writeln!(out, "**Errors:** {errors}");
        }
        if let Some(related) = meta.related {
            let _ = writeln!(out, "**Related:** {related}");
        }
        if let Some(status) = meta.status {
            let _ = writeln!(out, "**Status:** {status}");
        }
        if let Some(introduced) = meta.introduced {
            let _ = writeln!(out, "**Since:** {introduced}");
        }
        if let Some(examples) = meta.examples {
            out.push_str("\n**Examples:**\n");
            out.push_str(examples);
            out.push('\n');
        }
        out.push('\n');
    }

    let _ = writeln!(out, "Docs: {}", builtin_doc_url(func.name));
    out
}
