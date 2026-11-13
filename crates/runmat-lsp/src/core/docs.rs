use runmat_builtins::{
    BuiltinCatalogEntry, BuiltinDoc, BuiltinFunction, BuiltinParamType, BuiltinSignatureDescriptor,
};

use crate::core::builtins_json;

mod hover;

#[cfg(test)]
mod tests;

/// Helpers for documentation linking/formatting.
const RUNMAT_DOC_BASE_URL: &str = "https://runmat.com/docs/reference/builtins/";

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

fn render_builtin_hover_from_json(
    name: &str,
    signature: &str,
    fallback_description: &str,
    doc: builtins_json::BuiltinDocJson,
    remaining: usize,
) -> String {
    hover::legacy(name, signature, remaining, fallback_description, &doc)
}

/// Render a Markdown hover string for a builtin function, enriched with metadata.
pub fn build_builtin_hover(func: &BuiltinFunction) -> String {
    hover::builtin(func)
}

/// Build hover text from the executor-neutral catalog. Catalog-backed builtins
/// must remain fully discoverable without linking their runtime implementation.
pub fn build_catalog_hover(entry: &BuiltinCatalogEntry) -> String {
    build_catalog_hover_with_signatures(entry, &[0])
}

pub fn build_catalog_hover_with_signatures(
    entry: &BuiltinCatalogEntry,
    signatures: &[usize],
) -> String {
    hover::catalog(entry, signatures)
}

pub fn completion_signature_label(func: &BuiltinFunction) -> Option<String> {
    let signature = first_signature(func)?;
    let inputs = signature
        .inputs
        .iter()
        .map(format_param_for_completion)
        .collect::<Vec<_>>()
        .join(", ");
    let outputs = signature
        .outputs
        .iter()
        .map(|param| param.name.to_string())
        .collect::<Vec<_>>();
    if outputs.is_empty() {
        Some(format!("{}({inputs})", func.name))
    } else if outputs.len() == 1 {
        Some(format!("{} = {}({inputs})", outputs[0], func.name))
    } else {
        Some(format!(
            "[{}] = {}({inputs})",
            outputs.join(", "),
            func.name
        ))
    }
}

pub fn catalog_completion_signature_label(entry: &BuiltinCatalogEntry) -> Option<String> {
    completion_signature_label_for(entry.identity.name, entry.descriptor.signatures.first()?)
}

pub fn signature_labels(func: &BuiltinFunction) -> Option<Vec<String>> {
    let descriptor = func.descriptor?;
    let labels = descriptor
        .signatures
        .iter()
        .map(|sig| sig.label.to_string())
        .collect::<Vec<_>>();
    if labels.is_empty() {
        None
    } else {
        Some(labels)
    }
}

fn completion_signature_label_for(
    name: &str,
    signature: &BuiltinSignatureDescriptor,
) -> Option<String> {
    let inputs = signature
        .inputs
        .iter()
        .map(format_param_for_completion)
        .collect::<Vec<_>>()
        .join(", ");
    let outputs = signature
        .outputs
        .iter()
        .map(|param| param.name.to_string())
        .collect::<Vec<_>>();
    match outputs.as_slice() {
        [] => Some(format!("{name}({inputs})")),
        [output] => Some(format!("{output} = {name}({inputs})")),
        _ => Some(format!("[{}] = {name}({inputs})", outputs.join(", "))),
    }
}

fn signature_header(func: &BuiltinFunction) -> String {
    let Some(labels) = signature_labels(func) else {
        return format!("{}(...)", func.name);
    };
    labels.join("\n")
}

fn first_signature(func: &BuiltinFunction) -> Option<&BuiltinSignatureDescriptor> {
    let descriptor = func.descriptor?;
    descriptor.signatures.first()
}

fn format_param_for_completion(param: &runmat_builtins::BuiltinParamDescriptor) -> String {
    let ty = match param.ty {
        BuiltinParamType::Any => "any",
        BuiltinParamType::NumericScalar => "num",
        BuiltinParamType::IntegerScalar => "int",
        BuiltinParamType::StringScalar => "string",
        BuiltinParamType::NumericArray => "numeric array",
        BuiltinParamType::LogicalArray => "logical array",
        BuiltinParamType::SizeArg => "size",
        BuiltinParamType::LikePrototype => "prototype",
        BuiltinParamType::AxesHandle => "axes",
        BuiltinParamType::StyleSpec => "style",
        BuiltinParamType::PropertyName => "property",
        BuiltinParamType::PropertyValue => "value",
        BuiltinParamType::Callable => "function",
    };
    format!("{}: {}", param.name, ty)
}
