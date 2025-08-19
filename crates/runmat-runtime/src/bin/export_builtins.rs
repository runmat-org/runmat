use clap::Parser;
use runmat_builtins::{builtin_docs, builtin_functions, BuiltinFunction, Type, Value};
use runmat_runtime as _;
use serde::Serialize;
use std::fs;
use std::path::PathBuf; // ensure lib is linked

#[derive(Parser, Debug)]
#[command(
    name = "export-builtins",
    about = "Export builtin function metadata to JSON"
)]
struct Args {
    /// Output path for builtins.json
    #[arg(long = "out")]
    out: PathBuf,
}

#[derive(Serialize, Clone)]
struct BuiltinSignatureOut {
    r#in: Vec<String>,
    #[serde(rename = "inTypes")]
    in_types: Vec<String>,
    out: Vec<String>,
    #[serde(rename = "outTypes")]
    out_types: Vec<String>,
    nargin: Range,
    nargout: Range,
}

#[derive(Serialize, Clone)]
struct Range {
    min: usize,
    max: usize,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct BuiltinOut {
    name: String,
    slug: String,
    category: Vec<String>,
    summary: String,
    status: String,
    signatures: Vec<BuiltinSignatureOut>,
    #[serde(skip_serializing_if = "Option::is_none")]
    errors: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    examples: Option<Vec<ExampleOut>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    keywords: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    internal: Option<bool>,
}

#[derive(Serialize, Clone)]
struct ExampleOut {
    title: Option<String>,
    code: String,
}

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
                let dims_str = dims
                    .iter()
                    .map(|d| d.map(|n| n.to_string()).unwrap_or_else(|| "?".to_string()))
                    .collect::<Vec<_>>()
                    .join(",");
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
    let inputs: Vec<String> = b
        .param_types
        .iter()
        .enumerate()
        .map(|(i, _t)| {
            // Prefer symbolic names x,y,z,a,b,c then argN
            const SYMBOLS: &[&str] = &["x", "y", "z", "a", "b", "c", "m", "n", "dim", "idx"];
            let base = SYMBOLS.get(i).copied().unwrap_or("arg");
            let name = if SYMBOLS.get(i).is_some() {
                base.to_string()
            } else {
                format!("{base}{i}")
            };
            name
        })
        .collect();

    let outputs: Vec<String> = if matches!(b.return_type, Type::Void) {
        vec![]
    } else {
        vec!["out".to_string()]
    };
    let in_types: Vec<String> = b.param_types.iter().map(|t| format_type(t)).collect();
    let out_types: Vec<String> = if matches!(b.return_type, Type::Void) {
        vec![]
    } else {
        vec![format_type(&b.return_type)]
    };
    BuiltinSignatureOut {
        r#in: inputs,
        in_types,
        out: outputs,
        out_types,
        nargin: Range {
            min: 0,
            max: b.param_types.len(),
        },
        nargout: Range {
            min: if matches!(b.return_type, Type::Void) {
                0
            } else {
                0
            },
            max: if matches!(b.return_type, Type::Void) {
                0
            } else {
                1
            },
        },
    }
}

fn main() {
    // Touch a symbol from the lib to force link so inventory items are pulled in
    let _touch: fn(&str, &[Value]) -> Result<Value, String> = runmat_runtime::call_builtin;
    let args = Args::parse();
    use std::collections::{BTreeMap, BTreeSet};
    let mut grouped: BTreeMap<String, BuiltinOut> = BTreeMap::new();
    for b in builtin_functions() {
        let mut category: Vec<String> = if b.category.is_empty() {
            vec!["general".to_string()]
        } else {
            b.category
                .split('/')
                .map(|s| s.trim().to_string())
                .collect()
        };
        let sig = to_signature_out(b);
        let name = b.name.to_string();
        let slug = slugify(&name);
        let internal_flag = name.starts_with("__");

        // Merge optional docs (note: multiple BuiltinDoc entries may exist; prefer first with summary)
        let doc_opt = builtin_docs().into_iter().find(|d| d.name == b.name);
        let mut summary = if let Some(doc) = &doc_opt {
            doc.summary.map(|s| s.to_string())
        } else {
            None
        };
        if summary.is_none() {
            summary = if b.description.is_empty() {
                None
            } else {
                Some(b.description.to_string())
            };
        }
        let status = doc_opt
            .as_ref()
            .and_then(|d| d.status.map(|s| s.to_string()))
            .unwrap_or_else(|| "implemented".to_string());
        if internal_flag && status == "implemented" { /* keep */ }
        if let Some(doc_cat) = doc_opt
            .as_ref()
            .and_then(|d| d.category.map(|s| s.to_string()))
        {
            category = vec![doc_cat];
        }
        let errors = doc_opt.as_ref().and_then(|d| {
            d.errors
                .map(|s| s.split(',').map(|x| x.trim().to_string()).collect())
        });
        let examples = doc_opt.as_ref().and_then(|d| {
            d.examples.map(|s| {
                vec![ExampleOut {
                    title: None,
                    code: s.to_string(),
                }]
            })
        });
        let keywords = doc_opt.as_ref().and_then(|d| {
            d.keywords
                .map(|s| s.split(',').map(|x| x.trim().to_string()).collect())
        });

        let entry = grouped.entry(slug.clone()).or_insert_with(|| BuiltinOut {
            name: name.clone(),
            slug: slug.clone(),
            category: Vec::new(),
            summary: summary.clone().unwrap_or_default(),
            status: status.clone(),
            signatures: Vec::new(),
            errors: errors.clone(),
            examples: examples.clone(),
            keywords: keywords.clone(),
            internal: Some(internal_flag),
        });
        // Merge fields
        entry.signatures.push(sig);
        entry.status = status; // last wins; all are implemented anyway
        if entry.summary.is_empty() {
            if let Some(s) = summary {
                entry.summary = s;
            }
        }
        if entry.errors.is_none() {
            entry.errors = errors;
        }
        if entry.examples.is_none() {
            entry.examples = examples;
        }
        if entry.keywords.is_none() {
            entry.keywords = keywords;
        }
        // Categories: set-union and sorted
        let mut set: BTreeSet<String> = entry.category.iter().cloned().collect();
        for c in category {
            set.insert(c);
        }
        entry.category = set.into_iter().collect();
        // Internal: if any impl says internal
        if let Some(true) = entry.internal { /* keep */
        } else {
            entry.internal = Some(internal_flag);
        }
    }
    let mut out_vec: Vec<BuiltinOut> = grouped.into_values().collect();
    out_vec.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    let json = serde_json::to_string_pretty(&out_vec).expect("serialize");
    if let Some(parent) = args.out.parent() {
        fs::create_dir_all(parent).ok();
    }
    fs::write(&args.out, json).expect("write output");
}
