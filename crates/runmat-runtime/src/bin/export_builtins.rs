use clap::Parser;
use runmat_builtins::{builtin_docs, builtin_functions, BuiltinFunction, Type, Value};
use runmat_runtime as _;
use serde::Serialize;
use serde_yaml::Value as YamlValue;
use std::collections::HashMap;
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
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    mdx_path: Option<String>,
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
                format!("Tensor[{dims_str}]")
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
    let in_types: Vec<String> = b.param_types.iter().map(format_type).collect();
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
            min: 0,
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
    // Parse DOC_MD frontmatter from inventory texts if present
    let mut doc_text_frontmatter: HashMap<String, YamlValue> = HashMap::new();
    let mut doc_text_body: HashMap<String, String> = HashMap::new();
    // Pull through the re-export to ensure inventory is linked
    for entry in runmat_runtime::builtins::common::builtin_doc_texts() {
        if let Some((front, body)) = parse_frontmatter_and_body(entry.text) {
            doc_text_frontmatter.insert(entry.name.to_string(), front);
            doc_text_body.insert(entry.name.to_string(), body);
        }
    }
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
        let front = doc_text_frontmatter.get(b.name);
        let mut summary = if let Some(doc) = &doc_opt {
            doc.summary.map(|s| s.to_string())
        } else {
            None
        };
        if summary.is_none() {
            if let Some(fm) = front {
                if let Some(s) = yaml_get_string(fm, "summary") {
                    summary = Some(s);
                }
            }
        }
        if summary.is_none() {
            summary = if b.description.is_empty() {
                None
            } else {
                Some(b.description.to_string())
            };
        }
        let mut status = doc_opt
            .as_ref()
            .and_then(|d| d.status.map(|s| s.to_string()))
            .unwrap_or_else(|| "implemented".to_string());
        if let Some(fm) = front {
            if let Some(s) = yaml_get_string(fm, "status") {
                status = s;
            }
        }
        if internal_flag && status == "implemented" { /* keep */ }
        if let Some(doc_cat) = doc_opt
            .as_ref()
            .and_then(|d| d.category.map(|s| s.to_string()))
        {
            category = vec![doc_cat];
        }
        if let Some(fm) = front {
            if let Some(cat) = yaml_get_string(fm, "category") {
                category = vec![cat];
            }
        }
        let errors = doc_opt.as_ref().and_then(|d| {
            d.errors
                .map(|s| s.split(',').map(|x| x.trim().to_string()).collect())
        });
        let mut examples = doc_opt.as_ref().and_then(|d| {
            d.examples.map(|s| {
                vec![ExampleOut {
                    title: None,
                    code: s.to_string(),
                }]
            })
        });
        if examples.is_none() {
            if let Some(fm) = front {
                if let Some(arr) = yaml_get_string_array(fm, "examples") {
                    let v = arr
                        .into_iter()
                        .map(|code| ExampleOut { title: None, code })
                        .collect();
                    examples = Some(v);
                }
            }
        }
        let mut keywords = doc_opt.as_ref().and_then(|d| {
            d.keywords
                .map(|s| s.split(',').map(|x| x.trim().to_string()).collect())
        });
        if keywords.is_none() {
            if let Some(fm) = front {
                if let Some(arr) = yaml_get_string_array(fm, "keywords") {
                    keywords = Some(arr);
                }
            }
        }

        // Derive description from frontmatter override or first sentence(s) of first paragraph
        let description = {
            let override_desc = front.and_then(|fm| yaml_get_string(fm, "description"));
            if let Some(d) = override_desc {
                Some(d)
            } else {
                doc_text_body
                    .get(&name)
                    .and_then(|body| extract_description_from_body(body))
            }
        };

        let entry = grouped.entry(slug.clone()).or_insert_with(|| BuiltinOut {
            name: name.clone(),
            slug: slug.clone(),
            category: Vec::new(),
            summary: summary.clone().unwrap_or_default(),
            description: description.clone(),
            status: status.clone(),
            signatures: Vec::new(),
            errors: errors.clone(),
            examples: examples.clone(),
            keywords: keywords.clone(),
            internal: Some(internal_flag),
            mdx_path: None,
        });
        // Merge fields
        entry.signatures.push(sig);
        entry.status = status; // last wins; all are implemented anyway
        if entry.summary.is_empty() {
            if let Some(s) = summary {
                entry.summary = s;
            }
        }
        if entry.description.is_none() {
            entry.description = description.clone();
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

        // If we have long-form body, emit MDX file alongside JSON (sibling /builtins-mdx/<slug>.mdx)
        if let Some(body) = doc_text_body.get(&name) {
            if let Some(parent) = args.out.parent() {
                let mdx_dir = parent.join("builtins-mdx");
                let _ = fs::create_dir_all(&mdx_dir);
                let mdx_rel = format!("builtins-mdx/{}.mdx", slug);
                let mdx_abs = mdx_dir.join(format!("{}.mdx", slug));
                // Write file if absent or contents differ
                let write = match fs::read_to_string(&mdx_abs) {
                    Ok(existing) => existing != *body,
                    Err(_) => true,
                };
                if write {
                    let _ = fs::write(&mdx_abs, body);
                }
                entry.mdx_path = Some(mdx_rel);
            }
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

fn parse_frontmatter_and_body(doc: &str) -> Option<(YamlValue, String)> {
    let mut lines = doc.lines();
    if !matches!(lines.next(), Some(l) if l.trim_start().starts_with("---")) {
        return None;
    }
    let mut front = String::new();
    for line in lines {
        if line.trim_start().starts_with("---") {
            break;
        }
        front.push_str(line);
        front.push('\n');
    }
    let front_v: YamlValue = match serde_yaml::from_str::<YamlValue>(&front) {
        Ok(v) => v,
        Err(_) => return None,
    };
    // The remaining lines after the second --- constitute the body
    let body_start = doc.find("---").and_then(|first| {
        doc[first + 3..]
            .find("---")
            .map(|second| first + 3 + second + 3)
    });
    let body = body_start
        .map(|idx| doc[idx..].to_string())
        .unwrap_or_default();
    Some((front_v, body))
}

fn extract_description_from_body(body: &str) -> Option<String> {
    // Find first non-empty paragraph that is not a heading, code fence, list or blockquote
    let mut in_fence = false;
    let mut para_lines: Vec<String> = Vec::new();
    for line in body.lines() {
        let t = line.trim();
        if t.starts_with("```") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            continue;
        }
        if t.is_empty() {
            if !para_lines.is_empty() {
                break;
            }
            continue;
        }
        let is_heading = t.starts_with('#');
        let is_blockquote = t.starts_with('>');
        let is_list = t.starts_with("- ")
            || t.starts_with("* ")
            || t.chars().take_while(|c| c.is_ascii_digit()).count() > 0 && t.contains('.');
        if is_heading || is_blockquote || is_list {
            continue;
        }
        para_lines.push(t.to_string());
    }
    if para_lines.is_empty() {
        return None;
    }
    let paragraph = para_lines.join(" ");
    let sentences = split_sentences(&paragraph);
    if sentences.is_empty() {
        return None;
    }
    // Prefer 1–2 sentences up to ~200 chars
    let mut out = String::new();
    for s in sentences {
        if out.is_empty() {
            out.push_str(&s);
        } else if out.len() + 1 + s.len() <= 200 {
            out.push(' ');
            out.push_str(&s);
        } else {
            break;
        }
        if out.len() >= 120 {
            break;
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let ch = bytes[i] as char;
        current.push(ch);
        let is_end = ch == '.' || ch == '!' || ch == '?';
        if is_end {
            // Lookahead
            let next = bytes.get(i + 1).map(|b| *b as char);
            // Abbreviation/decimal heuristics
            let prev = bytes
                .get(i.saturating_sub(1))
                .map(|b| *b as char)
                .unwrap_or(' ');
            let next_is_space = matches!(next, Some(' ') | Some('\n') | None);
            let prev_is_digit = prev.is_ascii_digit();
            let next_is_digit = bytes
                .get(i + 1)
                .map(|b| (*b as char).is_ascii_digit())
                .unwrap_or(false);
            let tail = current.trim_end();
            let abbrev = tail.ends_with("e.g.")
                || tail.ends_with("i.e.")
                || tail.ends_with("etc.")
                || tail.ends_with("vs.")
                || tail.ends_with("Mr.")
                || tail.ends_with("Dr.")
                || tail.ends_with("Prof.")
                || tail.ends_with("No.")
                || tail.ends_with("Fig.")
                || tail.ends_with("Eq.")
                || tail.ends_with("al.")
                || tail.ends_with("approx.");
            let decimal = prev_is_digit && next_is_digit;
            if next_is_space && !abbrev && !decimal {
                out.push(current.trim().to_string());
                current.clear();
                // skip following space
                if matches!(next, Some(' ')) {
                    i += 1;
                }
            }
        }
        i += 1;
    }
    if !current.trim().is_empty() {
        out.push(current.trim().to_string());
    }
    out
}

fn yaml_get_string(v: &YamlValue, key: &str) -> Option<String> {
    v.get(key).and_then(|x| x.as_str()).map(|s| s.to_string())
}

fn yaml_get_string_array(v: &YamlValue, key: &str) -> Option<Vec<String>> {
    match v.get(key) {
        Some(YamlValue::Sequence(seq)) => Some(
            seq.iter()
                .filter_map(|e| e.as_str().map(|s| s.to_string()))
                .collect(),
        ),
        Some(YamlValue::String(s)) => Some(s.split(',').map(|x| x.trim().to_string()).collect()),
        _ => None,
    }
}
