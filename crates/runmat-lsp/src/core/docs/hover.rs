//! Compact editor presentation; complete reference prose stays in the catalog export.
use std::{collections::BTreeSet, fmt::Write as _};

use runmat_builtins::{
    BuiltinCatalogEntry, BuiltinDocumentationAuthority, BuiltinDocumentationLinkTarget,
    BuiltinFunction,
};

use super::{builtin_doc_url, builtins_json};

const MAX_EXAMPLE_LINES: usize = 12;
const MAX_EXAMPLE_BYTES: usize = 800;

fn paragraph(text: &str) -> String {
    text.split("\n\n")
        .find(|part| !part.trim().is_empty())
        .unwrap_or("")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn short_example(program: &str) -> bool {
    !program.trim().is_empty()
        && program.len() <= MAX_EXAMPLE_BYTES
        && program.lines().count() <= MAX_EXAMPLE_LINES
}

#[derive(Default)]
struct Related {
    names: BTreeSet<String>,
    links: Vec<String>,
}

impl Related {
    fn builtin(&mut self, name: &str, label: &str) {
        if self.names.insert(name.to_ascii_lowercase()) {
            self.links
                .push(format!("[{label}]({})", builtin_doc_url(name)));
        }
    }

    fn legacy(&mut self, link: &builtins_json::BuiltinDocLink) {
        // Relative builtin references are part of the legacy presentation schema.
        // Other link kinds belong on the reference page, not in See also.
        if let Some(name) = link.url.strip_prefix("./") {
            self.builtin(name, &link.label);
        }
    }
}

fn render(
    signature: &str,
    remaining: usize,
    summary: &str,
    example: Option<&str>,
    related: Related,
    slug: &str,
) -> String {
    let mut out = format!("```runmat\n{signature}\n```\n\n");
    if remaining > 0 {
        let suffix = if remaining == 1 { "" } else { "s" };
        let _ = writeln!(out, "+{remaining} more overload{suffix}\n");
    }
    let summary = paragraph(summary);
    if !summary.is_empty() {
        let _ = writeln!(out, "{summary}\n");
    }
    if let Some(program) = example {
        let _ = writeln!(out, "```runmat\n{}\n```\n", program.trim());
    }
    if !related.links.is_empty() {
        let _ = writeln!(out, "See also: {}\n", related.links.join(", "));
    }
    let _ = writeln!(out, "Docs: {}", builtin_doc_url(slug));
    out
}

pub(super) fn catalog(entry: &BuiltinCatalogEntry, selected: &[usize]) -> String {
    let signatures = entry.descriptor.signatures;
    let mut labels = selected
        .iter()
        .filter_map(|index| signatures.get(*index).map(|signature| signature.label))
        .collect::<Vec<_>>();
    if labels.is_empty() {
        if let Some(first) = signatures.first() {
            labels.push(first.label);
        }
    }
    let header = if labels.is_empty() {
        format!("{}(...)", entry.identity.name)
    } else {
        labels.join("\n")
    };
    let remaining = signatures.len().saturating_sub(labels.len());
    if entry.documentation.authority == BuiltinDocumentationAuthority::LegacySidecar {
        if let Some(document) = builtins_json::builtin_doc(entry.identity.name) {
            return super::render_builtin_hover_from_json(
                entry.identity.name,
                &header,
                entry.documentation.summary,
                document,
                remaining,
            );
        }
    }
    let documentation = &entry.documentation;
    let summary = if documentation.summary.trim().is_empty() {
        documentation.description
    } else {
        documentation.summary
    };
    let mut related = Related::default();
    for name in documentation.related {
        related.builtin(name, name);
    }
    for link in documentation.links {
        if let BuiltinDocumentationLinkTarget::Builtin(name) = link.target {
            related.builtin(name, link.label);
        }
    }
    render(
        &header,
        remaining,
        summary,
        documentation
            .examples
            .iter()
            .map(|example| example.program)
            .find(|program| short_example(program)),
        related,
        documentation.slug.unwrap_or(entry.identity.name),
    )
}

pub(super) fn legacy(
    name: &str,
    signature: &str,
    remaining: usize,
    fallback: &str,
    document: &builtins_json::BuiltinDocJson,
) -> String {
    let summary = document
        .summary
        .as_deref()
        .filter(|text| !text.trim().is_empty())
        .or_else(|| {
            document
                .description
                .as_deref()
                .filter(|text| !text.trim().is_empty())
        })
        .unwrap_or(fallback);
    let example = document
        .examples
        .iter()
        .flatten()
        .filter_map(|example| example.input.as_deref())
        .find(|program| short_example(program));
    let mut related = Related::default();
    for item in document.related.iter().flatten() {
        match item {
            builtins_json::BuiltinDocRelated::Name(name) => related.builtin(name, name),
            builtins_json::BuiltinDocRelated::Link(link) => related.legacy(link),
        }
    }
    for link in document.links.iter().flatten() {
        related.legacy(link);
    }
    render(
        signature,
        remaining,
        summary,
        example,
        related,
        document.slug.as_deref().unwrap_or(name),
    )
}

pub(super) fn builtin(function: &BuiltinFunction) -> String {
    let labels = super::signature_labels(function).unwrap_or_default();
    let header = labels
        .first()
        .cloned()
        .unwrap_or_else(|| super::signature_header(function));
    let remaining = labels.len().saturating_sub(1);
    if let Some(document) = builtins_json::builtin_doc(function.name) {
        return super::render_builtin_hover_from_json(
            function.name,
            &header,
            function.description,
            document,
            remaining,
        );
    }
    let summary = super::find_builtin_doc(function.name)
        .and_then(|metadata| metadata.summary)
        .filter(|text| !text.trim().is_empty())
        .unwrap_or(function.description);
    render(
        &header,
        remaining,
        summary,
        None,
        Related::default(),
        function.name,
    )
}
