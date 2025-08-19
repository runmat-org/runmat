use clap::Parser;
use runmat_builtins::{builtin_functions, builtin_docs, BuiltinFunction, Type, Value};
use serde::Serialize;
use std::fs;
use std::path::PathBuf;
use runmat_runtime as _; // ensure lib is linked

#[derive(Parser, Debug)]
#[command(name = "export-builtins", about = "Export builtin function metadata to JSON")] 
struct Args {
    /// Output path for builtins.json
    #[arg(long = "out")]
    out: PathBuf,
}

#[derive(Serialize)]
struct BuiltinSignatureOut {
    r#in: Vec<String>,
    #[serde(rename = "inTypes")] in_types: Vec<String>,
    out: Vec<String>,
    #[serde(rename = "outTypes")] out_types: Vec<String>,
    nargin: Range,
    nargout: Range,
}

#[derive(Serialize)]
struct Range { min: usize, max: usize }

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct BuiltinOut {
    name: String,
    slug: String,
    category: Vec<String>,
    summary: String,
    status: String,
    signatures: Vec<BuiltinSignatureOut>,
    #[serde(skip_serializing_if = "Option::is_none")] errors: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")] examples: Option<Vec<ExampleOut>>,
    #[serde(skip_serializing_if = "Option::is_none")] keywords: Option<Vec<String>>,
}

#[derive(Serialize)]
struct ExampleOut { title: Option<String>, code: String }

#[allow(dead_code)]
fn format_type(t: &Type) -> String {
    match t {
        Type::Int => "int".to_string(),
        Type::Num => "double".to_string(),
        Type::Bool => "logical".to_string(),
        Type::Logical => "LogicalArray".to_string(),
        Type::String => "string".to_string(),
        Type::Tensor { shape } => {
            if let Some(dims) = shape {
                let dims_str = dims.iter().map(|d| d.map(|n| n.to_string()).unwrap_or_else(|| "?".to_string())).collect::<Vec<_>>().join(",");
                format!("Tensor[{}]", dims_str)
            } else {
                "Tensor".to_string()
            }
        }
        Type::Cell { .. } => "cell".to_string(),
        Type::Function { .. } => "function".to_string(),
        Type::Void => "void".to_string(),
        Type::Unknown => "any".to_string(),
        Type::Union(_) => "any".to_string(),
        Type::Struct { .. } => "struct".to_string(),
    }
}

fn slugify(name: &str) -> String {
    name.to_lowercase().replace(' ', "-")
}

fn to_signature_out(b: &BuiltinFunction) -> BuiltinSignatureOut {
    let inputs: Vec<String> = b.param_types.iter().enumerate().map(|(i, _t)| {
        // Prefer symbolic names x,y,z,a,b,c then argN
        const SYMBOLS: &[&str] = &["x","y","z","a","b","c","m","n","dim","idx"];
        let base = SYMBOLS.get(i).copied().unwrap_or("arg");
        let name = if SYMBOLS.get(i).is_some() { base.to_string() } else { format!("{base}{i}") };
        name
    }).collect();

    let outputs: Vec<String> = if matches!(b.return_type, Type::Void) { vec![] } else { vec!["out".to_string()] };
    let in_types: Vec<String> = b.param_types.iter().map(|t| format_type(t)).collect();
    let out_types: Vec<String> = if matches!(b.return_type, Type::Void) { vec![] } else { vec![format_type(&b.return_type)] };
    BuiltinSignatureOut {
        r#in: inputs,
        in_types,
        out: outputs,
        out_types,
        nargin: Range { min: 0, max: b.param_types.len() },
        nargout: Range { min: if matches!(b.return_type, Type::Void) { 0 } else { 0 }, max: if matches!(b.return_type, Type::Void) { 0 } else { 1 } },
    }
}

fn main() {
    // Touch a symbol from the lib to force link so inventory items are pulled in
    let _touch: fn(&str, &[Value]) -> Result<Value, String> = runmat_runtime::call_builtin;
    let args = Args::parse();
    let mut out: Vec<BuiltinOut> = Vec::new();
    for b in builtin_functions() {
        let category = if b.category.is_empty() { vec!["general".to_string()] } else { b.category.split('/').map(|s| s.trim().to_string()).collect() };
        let signatures = vec![to_signature_out(b)];
        // Merge optional docs
        let doc = builtin_docs().into_iter().find(|d| d.name == b.name);
        let summary = doc.and_then(|d| d.summary.map(|s| s.to_string())).unwrap_or_else(|| if b.description.is_empty() { "".to_string() } else { b.description.to_string() });
        let status = doc.and_then(|d| d.status.map(|s| s.to_string())).unwrap_or_else(|| "implemented".to_string());
        let category = if let Some(d) = doc.and_then(|d| d.category.map(|s| s.to_string())) { vec![d] } else { category };
        out.push(BuiltinOut {
            name: b.name.to_string(),
            slug: slugify(b.name),
            category,
            summary,
            status,
            signatures,
            errors: doc.and_then(|d| d.errors.map(|s| s.split(',').map(|x| x.trim().to_string()).collect())),
            examples: doc.and_then(|d| d.examples.map(|s| vec![ExampleOut { title: None, code: s.to_string() } ])),
            keywords: doc.and_then(|d| d.keywords.map(|s| s.split(',').map(|x| x.trim().to_string()).collect())),
        });
    }
    // Sort stable for diffs
    out.sort_by(|a, b| a.name.cmp(&b.name));
    let json = serde_json::to_string_pretty(&out).expect("serialize");
    if let Some(parent) = args.out.parent() { fs::create_dir_all(parent).ok(); }
    fs::write(&args.out, json).expect("write output");
}


