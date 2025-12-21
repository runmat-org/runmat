use runmat_builtins::{BuiltinDoc, BuiltinFunction};

/// Helpers for documentation linking/formatting.
const RUNMAT_DOC_BASE_URL: &str = "https://runmat.dev/docs/builtins/";

/// Build a documentation URL for a builtin name.
pub fn builtin_doc_url(name: &str) -> String {
    format!("{RUNMAT_DOC_BASE_URL}{name}")
}

/// Lookup a BuiltinDoc by name.
pub fn find_builtin_doc(name: &str) -> Option<&'static BuiltinDoc> {
    // Note: builtin_docs() allocates a Vec; acceptable given small cardinality.
    runmat_builtins::builtin_docs()
        .into_iter()
        .find(|d| d.name == name)
}

/// Render a Markdown hover string for a builtin function, enriched with metadata.
pub fn build_builtin_hover(func: &BuiltinFunction) -> String {
    let signature = if !func.param_types.is_empty() {
        let params: Vec<String> = func.param_types.iter().map(|t| format!("{t:?}")).collect();
        format!("({}) -> {:?}", params.join(", "), func.return_type)
    } else {
        format!("builtin {}", func.name)
    };

    let mut out = String::new();
    out.push_str("```runmat\n");
    out.push_str(&signature);
    out.push_str("\n```\n");

    if !func.doc.is_empty() {
        out.push_str(func.doc);
        out.push_str("\n\n");
    }

    if let Some(meta) = find_builtin_doc(func.name) {
        if let Some(summary) = meta.summary {
            out.push_str(summary);
            out.push('\n');
        }
        if let Some(category) = meta.category {
            out.push_str(&format!("\n**Category:** {category}"));
        }
        if let Some(keywords) = meta.keywords {
            out.push_str(&format!("\n**Keywords:** {keywords}"));
        }
        if let Some(errors) = meta.errors {
            out.push_str(&format!("\n**Errors:** {errors}"));
        }
        if let Some(related) = meta.related {
            out.push_str(&format!("\n**Related:** {related}"));
        }
        if let Some(status) = meta.status {
            out.push_str(&format!("\n**Status:** {status}"));
        }
        if let Some(introduced) = meta.introduced {
            out.push_str(&format!("\n**Since:** {introduced}"));
        }
        if let Some(examples) = meta.examples {
            out.push_str("\n\n**Examples:**\n");
            out.push_str(examples);
            out.push('\n');
        }
        out.push('\n');
    }

    out.push_str(&format!("Docs: {}", builtin_doc_url(func.name)));
    out
}
