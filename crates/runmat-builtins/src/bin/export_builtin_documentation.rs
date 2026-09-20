use runmat_builtins::{
    builtin_catalog_entries, validate_builtin_catalog, BuiltinDocumentationAuthority,
    BuiltinDocumentationLinkTarget,
};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const EXPORT_SCHEMA_VERSION: u32 = 2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut output = None;
    let mut check_only = false;
    let mut transition = false;
    let mut arguments = env::args_os().skip(1);
    while let Some(argument) = arguments.next() {
        if argument == "--check" {
            check_only = true;
        } else if argument == "--transition" {
            transition = true;
        } else if argument == "--output" {
            let path = arguments.next().ok_or("--output requires a path")?;
            output = Some(PathBuf::from(path));
        } else {
            return Err(
                format!("unexpected argument: {}", PathBuf::from(argument).display()).into(),
            );
        }
    }

    if check_only && output.is_some() {
        return Err("--check and --output cannot be combined".into());
    }

    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let legacy_directory = repository.join("docs/builtins/reference");
    let export = build_export(&legacy_directory, transition)?;
    if check_only {
        return Ok(());
    }

    let encoded = format!("{}\n", serde_json::to_string_pretty(&export)?);
    if let Some(path) = output {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, encoded)?;
    } else {
        print!("{encoded}");
    }
    Ok(())
}

fn build_export(
    legacy_directory: &Path,
    transition: bool,
) -> Result<Value, Box<dyn std::error::Error>> {
    let mut legacy = read_legacy_documents(legacy_directory)?;
    let legacy_sidecar_count = legacy.len();
    let mut catalog_entries = builtin_catalog_entries().to_vec();
    catalog_entries.sort_by(|left, right| {
        left.identity
            .name
            .to_ascii_lowercase()
            .cmp(&right.identity.name.to_ascii_lowercase())
    });

    let catalog_names = catalog_entries
        .iter()
        .map(|entry| entry.identity.name.to_ascii_lowercase())
        .collect::<BTreeSet<_>>();
    let complete_names = catalog_names
        .iter()
        .cloned()
        .chain(legacy.keys().cloned())
        .collect::<BTreeSet<_>>();
    let mut documents = BTreeMap::new();
    let mut slugs = BTreeMap::new();
    let mut missing_legacy = Vec::new();

    for entry in catalog_entries {
        let key = entry.identity.name.to_ascii_lowercase();
        match entry.documentation.authority {
            BuiltinDocumentationAuthority::Catalog => {
                if legacy.contains_key(&key) {
                    return Err(format!(
                        "{} has canonical catalog documentation and a legacy sidecar",
                        entry.identity.name
                    )
                    .into());
                }
                validate_catalog_documentation(entry, &complete_names)?;
                let document = catalog_document(entry);
                record_document(&mut documents, &mut slugs, key, document)?;
            }
            BuiltinDocumentationAuthority::LegacySidecar => {
                let Some(mut document) = legacy.remove(&key) else {
                    missing_legacy.push(entry.identity.name);
                    continue;
                };
                annotate_legacy_document(&key, &mut document)?;
                record_document(&mut documents, &mut slugs, key, document)?;
            }
        }
    }

    if !missing_legacy.is_empty() && !transition {
        return Err(format!(
            "catalog identities declare legacy documentation but have no sidecar: {}",
            missing_legacy.join(", ")
        )
        .into());
    }

    for (key, mut document) in legacy {
        annotate_legacy_document(&key, &mut document)?;
        record_document(&mut documents, &mut slugs, key, document)?;
    }

    Ok(json!({
        "schema_version": EXPORT_SCHEMA_VERSION,
        "inventory": {
            "documents": documents.len(),
            "catalog_identities": catalog_names.len(),
            "legacy_sidecars": legacy_sidecar_count,
            "missing_catalog_documentation": missing_legacy,
        },
        "builtins": documents.into_values().collect::<Vec<_>>(),
    }))
}

fn read_legacy_documents(
    directory: &Path,
) -> Result<BTreeMap<String, Value>, Box<dyn std::error::Error>> {
    let mut documents = BTreeMap::new();
    for item in fs::read_dir(directory)? {
        let item = item?;
        let path = item.path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("json") {
            continue;
        }
        let stem = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .ok_or_else(|| format!("non-UTF-8 builtin sidecar path: {}", path.display()))?;
        let key = stem.to_ascii_lowercase();
        let mut document: Value = serde_json::from_slice(&fs::read(&path)?)?;
        document
            .as_object_mut()
            .ok_or_else(|| format!("builtin sidecar is not an object: {}", path.display()))?
            .insert("module_stem".into(), Value::String(stem.into()));
        if documents.insert(key.clone(), document).is_some() {
            return Err(format!("duplicate legacy documentation identity: {key}").into());
        }
    }
    Ok(documents)
}

fn annotate_legacy_document(key: &str, document: &mut Value) -> Result<(), String> {
    let object = document
        .as_object_mut()
        .ok_or_else(|| format!("legacy documentation for {key} is not an object"))?;
    object.insert("key".into(), Value::String(key.into()));
    object.insert("authority".into(), Value::String("legacy_sidecar".into()));
    if let Some(examples) = object.get_mut("examples").and_then(Value::as_array_mut) {
        for example in examples {
            match example {
                Value::Object(example) => annotate_legacy_example(example),
                Value::String(description) if !description.trim().is_empty() => {
                    let description = description.clone();
                    *example = json!({
                        "id": legacy_presentation_example_id(key, &description),
                        "description": description,
                        "fixture": "None",
                        "requirements": no_example_requirements(),
                    });
                }
                _ => {
                    return Err(format!(
                        "legacy documentation for {key} has an invalid example"
                    ))
                }
            }
        }
    }
    Ok(())
}

fn annotate_legacy_example(example: &mut Map<String, Value>) {
    example
        .entry("fixture")
        .or_insert_with(|| Value::String("None".into()));
    example
        .entry("requirements")
        .or_insert_with(no_example_requirements);
}

fn no_example_requirements() -> Value {
    json!({
        "host": "Any",
        "engine": "Default",
        "compiler": [],
        "runtime": [],
        "toolchain": []
    })
}

fn legacy_presentation_example_id(key: &str, description: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(b"runmat.builtin-example.legacy-presentation.v1\0");
    digest.update(key.as_bytes());
    digest.update(b"\0");
    digest.update(description.as_bytes());
    format!("legacy-presentation-{digest:x}")
}

fn record_document(
    documents: &mut BTreeMap<String, Value>,
    slugs: &mut BTreeMap<String, String>,
    key: String,
    document: Value,
) -> Result<(), String> {
    let slug = document
        .get("slug")
        .and_then(Value::as_str)
        .or_else(|| document.get("title").and_then(Value::as_str))
        .unwrap_or(&key)
        .to_ascii_lowercase();
    if let Some(existing) = slugs.insert(slug.clone(), key.clone()) {
        return Err(format!(
            "duplicate builtin documentation slug {slug:?}: {existing} and {key}"
        ));
    }
    if documents.insert(key.clone(), document).is_some() {
        return Err(format!("duplicate builtin documentation identity: {key}"));
    }
    Ok(())
}

fn validate_catalog_documentation(
    entry: &'static runmat_builtins::BuiltinCatalogEntry,
    complete_names: &BTreeSet<String>,
) -> Result<(), String> {
    let documentation = &entry.documentation;
    let name = entry.identity.name;
    let errors = validate_builtin_catalog(&[entry]);
    if !errors.is_empty() {
        return Err(format!(
            "{name} has invalid canonical catalog data: {errors:#?}"
        ));
    }
    for related in documentation.related {
        if !complete_names.contains(&related.to_ascii_lowercase()) {
            return Err(format!(
                "{name} references unknown related builtin {related}"
            ));
        }
    }
    for link in documentation.links {
        if let BuiltinDocumentationLinkTarget::Builtin(target) = link.target {
            if !complete_names.contains(&target.to_ascii_lowercase()) {
                return Err(format!("{name} links to unknown builtin {target}"));
            }
        }
    }
    Ok(())
}

fn catalog_document(entry: &runmat_builtins::BuiltinCatalogEntry) -> Value {
    let documentation = &entry.documentation;
    let title = documentation.title.unwrap_or(entry.identity.name);
    let slug = documentation
        .slug
        .unwrap_or(entry.identity.name)
        .to_ascii_lowercase();
    let examples = documentation
        .examples
        .iter()
        .map(|example| {
            json!({
                "id": example.id,
                "description": example.title,
                "input": example.program,
                "output": example.display_output,
                "compatibility": example.compatibility,
                "harness": example.harness,
                "fixture": example.fixture,
                "requirements": example.requirements,
                "verification": example.verification,
            })
        })
        .collect::<Vec<_>>();
    let related = documentation
        .related
        .iter()
        .map(|name| json!({ "label": name, "url": format!("./{name}") }))
        .collect::<Vec<_>>();
    let mut object = Map::new();
    object.insert(
        "key".into(),
        Value::String(entry.identity.name.to_ascii_lowercase()),
    );
    object.insert(
        "module_stem".into(),
        Value::String(entry.identity.name.into()),
    );
    object.insert("authority".into(), Value::String("catalog".into()));
    object.insert("title".into(), Value::String(title.into()));
    object.insert("slug".into(), Value::String(slug));
    object.insert("category".into(), Value::String(entry.category.into()));
    object.insert(
        "summary".into(),
        Value::String(documentation.summary.into()),
    );
    object.insert(
        "description".into(),
        Value::String(documentation.description.into()),
    );
    object.insert("keywords".into(), json!(documentation.keywords));
    object.insert("sections".into(), json!(documentation.sections));
    object.insert("examples".into(), Value::Array(examples));
    object.insert(
        "example_exemption".into(),
        json!(documentation.example_exemption),
    );
    object.insert("faqs".into(), json!(documentation.faqs));
    object.insert(
        "links".into(),
        Value::Array(
            documentation
                .links
                .iter()
                .map(|link| {
                    let url = match link.target {
                        BuiltinDocumentationLinkTarget::Builtin(name) => format!("./{name}"),
                        BuiltinDocumentationLinkTarget::Documentation(url)
                        | BuiltinDocumentationLinkTarget::Source(url)
                        | BuiltinDocumentationLinkTarget::External(url) => url.to_string(),
                    };
                    json!({ "label": link.label, "url": url })
                })
                .collect(),
        ),
    );
    object.insert("related".into(), Value::Array(related));
    object.insert("media".into(), json!(documentation.media));
    object.insert("evidence".into(), json!(documentation.evidence));
    if let Some(source) = documentation
        .evidence
        .implementation
        .iter()
        .find_map(|link| {
            let BuiltinDocumentationLinkTarget::Source(url) = link.target else {
                return None;
            };
            Some(json!({ "label": link.label, "url": url }))
        })
    {
        object.insert("source".into(), source);
    }
    object.insert("introduced".into(), json!(documentation.introduced));
    object.insert("status".into(), json!(documentation.status));
    object.insert(
        "catalog".into(),
        json!({
            "descriptor": entry.descriptor,
            "contract": entry.contract,
            "placement": entry.placement,
            "link": entry.link,
            "bindings": entry.bindings,
            "extensions": entry.extensions,
            "integer_capabilities": entry.integer_capabilities,
            "integer_audit": entry.integer_audit,
        }),
    );
    Value::Object(object)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_transition_export_is_complete_and_deterministic() {
        let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let directory = repository.join("docs/builtins/reference");
        let first = build_export(&directory, true).expect("first export");
        let second = build_export(&directory, true).expect("second export");
        assert_eq!(first, second);
        let expected_documents = read_legacy_documents(&directory)
            .expect("legacy documents")
            .into_keys()
            .chain(
                builtin_catalog_entries()
                    .iter()
                    .filter(|entry| {
                        entry.documentation.authority == BuiltinDocumentationAuthority::Catalog
                    })
                    .map(|entry| entry.identity.name.to_ascii_lowercase()),
            )
            .collect::<BTreeSet<_>>()
            .len();
        assert_eq!(
            first["builtins"].as_array().map(Vec::len),
            Some(expected_documents),
            "the transition must preserve the complete current documentation inventory"
        );
        assert_eq!(first["schema_version"], EXPORT_SCHEMA_VERSION);
        for document in first["builtins"]
            .as_array()
            .expect("documentation export rows")
        {
            for example in document["examples"]
                .as_array()
                .expect("documentation examples")
            {
                assert!(example.is_object());
                assert!(example.get("fixture").is_some());
                assert!(example.get("requirements").is_some());
            }
        }
        let has_missing = first["inventory"]["missing_catalog_documentation"]
            .as_array()
            .is_some_and(|missing| !missing.is_empty());
        assert_eq!(build_export(&directory, false).is_err(), has_missing);
    }

    #[test]
    fn legacy_presentation_examples_are_explicit_schema_v2_records() {
        let mut document = json!({
            "title": "sample",
            "examples": ["A prose-only example."]
        });

        annotate_legacy_document("sample", &mut document).expect("annotate legacy document");

        let example = &document["examples"][0];
        assert_eq!(example["description"], "A prose-only example.");
        assert_eq!(example["fixture"], "None");
        assert_eq!(example["requirements"], no_example_requirements());
        assert_eq!(
            example["id"],
            legacy_presentation_example_id("sample", "A prose-only example.")
        );
        assert!(example.get("input").is_none());
    }
}
