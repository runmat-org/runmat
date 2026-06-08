use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn is_success_status(code: &openapiv3::StatusCode) -> bool {
    match code {
        openapiv3::StatusCode::Code(value) => (200..=299).contains(value),
        openapiv3::StatusCode::Range(value) => *value == 2,
    }
}

fn normalize_responses_for_progenitor(responses: &mut openapiv3::Responses) {
    let mut success_keys: Vec<openapiv3::StatusCode> = responses
        .responses
        .keys()
        .filter(|code| is_success_status(code))
        .cloned()
        .collect();
    if success_keys.len() <= 1 {
        if let Some(code) = success_keys.pop() {
            if let Some(openapiv3::ReferenceOr::Item(response)) = responses.responses.get_mut(&code)
            {
                if response.content.len() > 1 {
                    if let Some(mt) = response.content.shift_remove("application/json") {
                        response.content.clear();
                        response.content.insert("application/json".to_string(), mt);
                    } else if let Some(first_key) = response.content.keys().next().cloned() {
                        if let Some(mt) = response.content.shift_remove(&first_key) {
                            response.content.clear();
                            response.content.insert(first_key, mt);
                        }
                    }
                }
            }
        }
        return;
    }

    success_keys.sort_by(|a, b| match (a, b) {
        (openapiv3::StatusCode::Code(x), openapiv3::StatusCode::Code(y)) => x.cmp(y),
        (openapiv3::StatusCode::Code(_), openapiv3::StatusCode::Range(_)) => {
            std::cmp::Ordering::Less
        }
        (openapiv3::StatusCode::Range(_), openapiv3::StatusCode::Code(_)) => {
            std::cmp::Ordering::Greater
        }
        (openapiv3::StatusCode::Range(x), openapiv3::StatusCode::Range(y)) => x.cmp(y),
    });

    let keep = success_keys
        .iter()
        .find(|code| matches!(code, openapiv3::StatusCode::Code(200)))
        .cloned()
        .unwrap_or_else(|| success_keys[0].clone());

    let drop_keys: Vec<openapiv3::StatusCode> = responses
        .responses
        .keys()
        .filter(|code| is_success_status(code) && **code != keep)
        .cloned()
        .collect();
    for key in drop_keys {
        let _ = responses.responses.shift_remove(&key);
    }

    if let Some(openapiv3::ReferenceOr::Item(response)) = responses.responses.get_mut(&keep) {
        if response.content.len() > 1 {
            if let Some(mt) = response.content.shift_remove("application/json") {
                response.content.clear();
                response.content.insert("application/json".to_string(), mt);
            } else if let Some(first_key) = response.content.keys().next().cloned() {
                if let Some(mt) = response.content.shift_remove(&first_key) {
                    response.content.clear();
                    response.content.insert(first_key, mt);
                }
            }
        }
    }
}

fn operation_suffix(method: &str, path: &str) -> String {
    let normalized = path
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect::<String>();
    format!("{}_{}", method, normalized)
}

fn normalize_openapi_for_progenitor(spec: &mut openapiv3::OpenAPI) {
    let mut seen_operation_ids = HashSet::new();
    for (path, item) in spec.paths.paths.iter_mut() {
        let openapiv3::ReferenceOr::Item(item) = item else {
            continue;
        };
        for (method, op) in [
            ("get", item.get.as_mut()),
            ("put", item.put.as_mut()),
            ("post", item.post.as_mut()),
            ("delete", item.delete.as_mut()),
            ("options", item.options.as_mut()),
            ("head", item.head.as_mut()),
            ("patch", item.patch.as_mut()),
            ("trace", item.trace.as_mut()),
        ] {
            let Some(op) = op else {
                continue;
            };
            if let Some(operation_id) = op.operation_id.clone() {
                if !seen_operation_ids.insert(operation_id.clone()) {
                    let replacement =
                        format!("{}_{}", operation_id, operation_suffix(method, path));
                    op.operation_id = Some(replacement.clone());
                    seen_operation_ids.insert(replacement);
                }
            }
            normalize_responses_for_progenitor(&mut op.responses);
        }
    }
}

fn generate_spec(input_path: PathBuf, output_name: &str) {
    println!("cargo:rerun-if-changed={}", input_path.display());
    let spec_text = fs::read_to_string(&input_path).expect("read openapi spec");
    let mut spec: openapiv3::OpenAPI =
        serde_yaml::from_str(&spec_text).expect("parse openapi spec");
    normalize_openapi_for_progenitor(&mut spec);

    let settings = progenitor::GenerationSettings::default();
    let mut generator = progenitor::Generator::new(&settings);
    let tokens = generator
        .generate_tokens(&spec)
        .expect("generate client tokens");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("out dir"));
    let out_file = out_dir.join(output_name);
    fs::write(&out_file, tokens.to_string()).expect("write openapi client");
}

fn resolve_spec_path(manifest_dir: &Path) -> PathBuf {
    // Published crates cannot read files outside the package root, so keep a
    // crate-local copy for packaging and fall back to the workspace path while
    // developing if that copy is missing.
    let packaged_spec = manifest_dir.join("openapi/runmat-public.yaml");
    if packaged_spec.is_file() {
        return packaged_spec;
    }

    let workspace_spec = manifest_dir.join("../../openapi/runmat-public.yaml");
    if workspace_spec.is_file() {
        return workspace_spec;
    }

    panic!(
        "could not find runmat-public.yaml at {} or {}",
        packaged_spec.display(),
        workspace_spec.display()
    );
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("manifest"));
    generate_spec(resolve_spec_path(&manifest_dir), "runmat_public.rs");
}
