use std::env;
use std::fs;
use std::path::PathBuf;

fn is_success_status(code: &openapiv3::StatusCode) -> bool {
    match code {
        openapiv3::StatusCode::Code(value) => (200..=299).contains(value),
        openapiv3::StatusCode::Range(value) => *value == 2,
    }
}

fn normalize_responses_for_progenitor(responses: &mut openapiv3::Responses) {
    // Progenitor currently expects at most one "success" response type per operation.
    // Some specs include multiple 2xx responses (or multiple content types), which makes
    // the generator panic. Prefer 200, otherwise keep the lowest 2xx code/range.
    let mut success_keys: Vec<openapiv3::StatusCode> = responses
        .responses
        .keys()
        .filter(|code| is_success_status(code))
        .cloned()
        .collect();
    if success_keys.len() <= 1 {
        // Still normalize the single response content types if possible.
        if let Some(code) = success_keys.pop() {
            if let Some(openapiv3::ReferenceOr::Item(response)) = responses.responses.get_mut(&code) {
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
        (openapiv3::StatusCode::Code(_), openapiv3::StatusCode::Range(_)) => std::cmp::Ordering::Less,
        (openapiv3::StatusCode::Range(_), openapiv3::StatusCode::Code(_)) => std::cmp::Ordering::Greater,
        (openapiv3::StatusCode::Range(x), openapiv3::StatusCode::Range(y)) => x.cmp(y),
    });

    let keep = success_keys
        .iter()
        .find(|code| matches!(code, openapiv3::StatusCode::Code(200)))
        .cloned()
        .unwrap_or_else(|| success_keys[0].clone());

    // Drop all other success responses.
    let drop_keys: Vec<openapiv3::StatusCode> = responses
        .responses
        .keys()
        .filter(|code| is_success_status(code) && **code != keep)
        .cloned()
        .collect();
    for key in drop_keys {
        let _ = responses.responses.shift_remove(&key);
    }

    // Normalize kept response content types if needed.
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

fn normalize_openapi_for_progenitor(spec: &mut openapiv3::OpenAPI) {
    for (_path, item) in spec.paths.paths.iter_mut() {
        let openapiv3::ReferenceOr::Item(item) = item else {
            continue;
        };
        for op in [
            item.get.as_mut(),
            item.put.as_mut(),
            item.post.as_mut(),
            item.delete.as_mut(),
            item.options.as_mut(),
            item.head.as_mut(),
            item.patch.as_mut(),
            item.trace.as_mut(),
        ]
        .into_iter()
        .flatten()
        {
            normalize_responses_for_progenitor(&mut op.responses);
        }
    }
}

fn main() {
    let spec_path = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("manifest"))
        .join("../../openapi/runmat-public.yaml");
    println!("cargo:rerun-if-changed={}", spec_path.display());

    let spec_text = fs::read_to_string(&spec_path).expect("read openapi spec");
    let mut spec: openapiv3::OpenAPI = serde_yaml::from_str(&spec_text).expect("parse openapi spec");
    normalize_openapi_for_progenitor(&mut spec);

    let settings = progenitor::GenerationSettings::default();
    let mut generator = progenitor::Generator::new(&settings);
    let tokens = generator
        .generate_tokens(&spec)
        .expect("generate client tokens");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("out dir"));
    let out_file = out_dir.join("runmat_public.rs");
    fs::write(&out_file, tokens.to_string()).expect("write openapi client");
}
