use runmat_builtins::{
    builtin_catalog_aliases, builtin_catalog_entries, builtin_constant_catalog_entries,
    public_builtin_catalog_entries, validate_builtin_catalog, validate_builtin_documentation,
    BuiltinCatalogAudience, BuiltinConstantCatalogEntry, BuiltinDocumentation,
    BuiltinDocumentationAuthority, BuiltinDocumentationLinkTarget,
};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::Path;

#[path = "export_builtin_documentation/cli.rs"]
mod cli;

use cli::{parse_options, ExportMode};

const EXPORT_SCHEMA_VERSION: u32 = 2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = parse_options(env::args_os().skip(1))?;
    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let legacy_directory = repository.join("docs/builtins/reference");
    let export = match options.mode {
        ExportMode::Normal { transition } => {
            build_export(&legacy_directory, transition, &options.identities)?
        }
        ExportMode::CatalogPreview => {
            build_export_mode(&legacy_directory, options.mode, &options.identities)?
        }
    };
    if options.check_only {
        return Ok(());
    }

    let encoded = format!("{}\n", serde_json::to_string_pretty(&export)?);
    if let Some(path) = options.output {
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
    identities: &BTreeSet<String>,
) -> Result<Value, Box<dyn std::error::Error>> {
    build_export_mode(
        legacy_directory,
        ExportMode::Normal { transition },
        identities,
    )
}

fn build_export_mode(
    legacy_directory: &Path,
    mode: ExportMode,
    identities: &BTreeSet<String>,
) -> Result<Value, Box<dyn std::error::Error>> {
    build_export_with_legacy(read_legacy_documents(legacy_directory)?, mode, identities)
}

fn build_export_with_legacy(
    mut legacy: BTreeMap<String, Value>,
    mode: ExportMode,
    identities: &BTreeSet<String>,
) -> Result<Value, Box<dyn std::error::Error>> {
    if mode == ExportMode::CatalogPreview && identities.is_empty() {
        return Err("catalog preview requires selected identities".into());
    }
    for entry in builtin_catalog_entries()
        .iter()
        .filter(|entry| entry.audience == BuiltinCatalogAudience::Fixture)
    {
        if legacy.contains_key(&entry.identity.name.to_ascii_lowercase()) {
            return Err(format!(
                "fixture {} must not have a public legacy documentation sidecar",
                entry.identity.name
            )
            .into());
        }
    }
    let mut catalog_entries = public_builtin_catalog_entries().to_vec();
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
        .chain(
            builtin_constant_catalog_entries()
                .iter()
                .map(|entry| entry.name.to_ascii_lowercase()),
        )
        .chain(legacy.keys().cloned())
        .collect::<BTreeSet<_>>();
    if let Some(name) = identities.difference(&complete_names).next() {
        return Err(format!("unknown documentation identity: {name}").into());
    }
    if mode == ExportMode::CatalogPreview {
        let canonical_names = catalog_entries
            .iter()
            .filter(|entry| entry.documentation.authority == BuiltinDocumentationAuthority::Catalog)
            .map(|entry| entry.identity.name.to_ascii_lowercase())
            .chain(
                builtin_constant_catalog_entries()
                    .iter()
                    .filter(|entry| entry.documentation.is_some())
                    .map(|entry| entry.name.to_ascii_lowercase()),
            )
            .collect::<BTreeSet<_>>();
        if let Some(name) = identities.difference(&canonical_names).next() {
            return Err(format!("catalog preview requires canonical documentation: {name}").into());
        }
    }
    // A scoped export remains strict about duplicate ownership and missing
    // documentation within its scope. Links resolve against the full catalog.
    // Unrelated unfinished cutovers cannot block field-conservation checks.
    if !identities.is_empty() {
        legacy.retain(|name, _| identities.contains(name));
        catalog_entries
            .retain(|entry| identities.contains(&entry.identity.name.to_ascii_lowercase()));
    }
    let legacy_sidecar_count = legacy.len();
    let selected_catalog_count = catalog_entries.len();
    let mut documents = BTreeMap::new();
    let mut slugs = BTreeMap::new();
    let mut missing_legacy = Vec::new();

    for entry in catalog_entries {
        let key = entry.identity.name.to_ascii_lowercase();
        match entry.documentation.authority {
            BuiltinDocumentationAuthority::Catalog => {
                if legacy.contains_key(&key) && mode != ExportMode::CatalogPreview {
                    return Err(format!(
                        "{} has canonical catalog documentation and a legacy sidecar",
                        entry.identity.name
                    )
                    .into());
                }
                validate_catalog_documentation(entry, &complete_names)?;
                let document = catalog_document(entry);
                record_document(&mut documents, &mut slugs, key, document)?;
                if mode == ExportMode::CatalogPreview {
                    legacy.remove(&entry.identity.name.to_ascii_lowercase());
                }
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

    if !missing_legacy.is_empty() && mode == (ExportMode::Normal { transition: false }) {
        return Err(format!(
            "catalog identities declare legacy documentation but have no sidecar: {}",
            missing_legacy.join(", ")
        )
        .into());
    }

    for entry in builtin_constant_catalog_entries() {
        if !identities.is_empty() && !identities.contains(&entry.name.to_ascii_lowercase()) {
            continue;
        }
        let Some(documentation) = &entry.documentation else {
            continue;
        };
        let key = entry.name.to_ascii_lowercase();
        if catalog_names.contains(&key)
            || builtin_catalog_aliases()
                .iter()
                .any(|alias| alias.alias.name.eq_ignore_ascii_case(entry.name))
        {
            return Err(format!(
                "{} declares separate constant and callable documentation",
                entry.name
            )
            .into());
        }
        if legacy.contains_key(&key) && mode != ExportMode::CatalogPreview {
            return Err(format!(
                "{} has canonical constant documentation and a legacy sidecar",
                entry.name
            )
            .into());
        }
        let errors = validate_builtin_documentation(entry.name, documentation);
        if !errors.is_empty() {
            return Err(format!(
                "{} has invalid canonical constant documentation: {errors:#?}",
                entry.name
            )
            .into());
        }
        validate_documentation_links(entry.name, documentation, &complete_names)?;
        record_document(
            &mut documents,
            &mut slugs,
            key,
            constant_document(entry, documentation),
        )?;
        if mode == ExportMode::CatalogPreview {
            legacy.remove(&entry.name.to_ascii_lowercase());
        }
    }

    for (key, mut document) in legacy {
        annotate_legacy_document(&key, &mut document)?;
        record_document(&mut documents, &mut slugs, key, document)?;
    }

    let mut export = json!({
        "schema_version": EXPORT_SCHEMA_VERSION,
        "inventory": {
            "documents": documents.len(),
            "catalog_identities": selected_catalog_count,
            "legacy_sidecars": legacy_sidecar_count,
            "missing_catalog_documentation": missing_legacy,
        },
        "builtins": documents.into_values().collect::<Vec<_>>(),
    });
    if mode == ExportMode::CatalogPreview {
        export["preview"] = json!({
            "kind": "catalog",
            "final_acceptance": false,
            "sidecars_still_present": legacy_sidecar_count,
        });
    }
    Ok(export)
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
    format!("legacy-presentation-{:x}", digest.finalize())
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
    validate_documentation_links(name, documentation, complete_names)
}

fn validate_documentation_links(
    name: &str,
    documentation: &BuiltinDocumentation,
    complete_names: &BTreeSet<String>,
) -> Result<(), String> {
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
    let mut document =
        documentation_document(entry.identity.name, entry.category, &entry.documentation);
    document["catalog"] = json!({
        "descriptor": entry.descriptor,
        "contract": entry.contract,
        "placement": entry.placement,
        "link": entry.link,
        "bindings": entry.bindings,
        "extensions": entry.extensions,
        "integer_capabilities": entry.integer_capabilities,
        "integer_audit": entry.integer_audit,
    });
    document
}

fn constant_document(
    entry: &BuiltinConstantCatalogEntry,
    documentation: &BuiltinDocumentation,
) -> Value {
    let mut document = documentation_document(entry.name, "constants", documentation);
    document["catalog"] = json!({ "constant": { "name": entry.name, "kind": entry.kind } });
    document
}

fn documentation_document(
    name: &str,
    category: &str,
    documentation: &BuiltinDocumentation,
) -> Value {
    let title = documentation.title.unwrap_or(name);
    let slug = documentation.slug.unwrap_or(name).to_ascii_lowercase();
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
    object.insert("key".into(), Value::String(name.to_ascii_lowercase()));
    object.insert("module_stem".into(), Value::String(name.into()));
    object.insert("authority".into(), Value::String("catalog".into()));
    object.insert("title".into(), Value::String(title.into()));
    object.insert("slug".into(), Value::String(slug));
    object.insert("category".into(), Value::String(category.into()));
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
    Value::Object(object)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_legacy(names: &[&str]) -> BTreeMap<String, Value> {
        names
            .iter()
            .map(|name| (name.to_string(), json!({ "title": name })))
            .collect()
    }

    #[test]
    fn constant_documentation_exports_bare_value_examples_without_callable_contracts() {
        use runmat_builtins::{
            BuiltinCatalogProvenance, BuiltinConstantKind, BuiltinExample,
            BuiltinExampleCompatibility, BuiltinExampleFixture, BuiltinExampleHarness,
            BuiltinExampleRequirements, BuiltinExampleVerification,
        };
        const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
            authority: BuiltinDocumentationAuthority::Catalog,
            summary: "Circle constant.",
            description: "The real double scalar circle constant.",
            keywords: &["constant"],
            examples: &[BuiltinExample {
                id: "bare-value", title: "Read a constant", program: "value = pi;",
                display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
                harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
                requirements: BuiltinExampleRequirements::NONE,
                verification: BuiltinExampleVerification::Assertions {
                    source: "assert(isa(value, 'double')); assert(isscalar(value)); assert(value > 3 && value < 4);",
                },
            }],
            ..BuiltinDocumentation::EMPTY
        };
        let entry = BuiltinConstantCatalogEntry {
            name: "pi",
            kind: BuiltinConstantKind::RealDouble,
            literal: None,
            provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
            documentation: Some(DOCUMENTATION),
        };
        assert!(validate_builtin_documentation(entry.name, &DOCUMENTATION).is_empty());
        let document = constant_document(&entry, &DOCUMENTATION);
        assert_eq!(document["key"], "pi");
        assert_eq!(document["authority"], "catalog");
        assert_eq!(document["examples"][0]["input"], "value = pi;");
        assert_eq!(document["catalog"]["constant"]["kind"], "real_double");
        for field in ["descriptor", "contract", "bindings", "link"] {
            assert!(document["catalog"].get(field).is_none());
        }
        let mut invalid = DOCUMENTATION;
        invalid.examples = &[];
        assert!(!validate_builtin_documentation(entry.name, &invalid).is_empty());
        assert!(
            !validate_builtin_documentation(entry.name, &BuiltinDocumentation::EMPTY).is_empty()
        );
    }

    #[test]
    fn scoped_export_validates_ownership_and_identity_without_unrelated_cutovers() {
        let selected = BTreeSet::from(["minreal".to_string()]);
        let export = build_export_with_legacy(
            BTreeMap::new(),
            ExportMode::Normal { transition: false },
            &selected,
        )
        .expect("canonical minreal");
        assert_eq!(export["inventory"]["documents"], 1);
        assert_eq!(export["inventory"]["legacy_sidecars"], 0);
        assert_eq!(export["builtins"][0]["key"], "minreal");
        assert_eq!(export["builtins"][0]["authority"], "catalog");
        assert!(export.get("preview").is_none());
        let unknown = BTreeSet::from(["missing_documentation_identity".to_string()]);
        assert!(build_export_with_legacy(
            BTreeMap::new(),
            ExportMode::Normal { transition: true },
            &unknown,
        )
        .expect_err("unknown identity")
        .to_string()
        .contains("unknown documentation identity"));
    }

    #[test]
    fn catalog_preview_is_selected_and_does_not_weaken_normal_ownership() {
        let selected = BTreeSet::from(["colon".to_string()]);
        assert!(build_export_with_legacy(
            synthetic_legacy(&["colon"]),
            ExportMode::Normal { transition: false },
            &selected,
        )
        .expect_err("normal export rejects duplicate authority")
        .to_string()
        .contains("canonical catalog documentation and a legacy sidecar"));

        let preview = build_export_with_legacy(
            synthetic_legacy(&["colon"]),
            ExportMode::CatalogPreview,
            &selected,
        )
        .expect("selected canonical preview");
        assert_eq!(preview["inventory"]["documents"], 1);
        assert_eq!(preview["inventory"]["legacy_sidecars"], 1);
        assert_eq!(preview["preview"]["final_acceptance"], false);
        assert_eq!(preview["builtins"][0]["key"], "colon");
        assert_eq!(preview["builtins"][0]["authority"], "catalog");
        assert!(build_export_with_legacy(
            synthetic_legacy(&["colon"]),
            ExportMode::CatalogPreview,
            &BTreeSet::new(),
        )
        .is_err());
        assert!(build_export_with_legacy(
            synthetic_legacy(&["synthetic_legacy_only_documentation_fixture"]),
            ExportMode::CatalogPreview,
            &BTreeSet::from(["synthetic_legacy_only_documentation_fixture".to_string()]),
        )
        .is_err());
    }

    #[test]
    fn catalog_preview_cli_requires_selection_and_excludes_acceptance_modes() {
        let parse = |arguments: &[&str]| {
            parse_options(arguments.iter().map(std::ffi::OsString::from))
                .map(|options| (options.mode, options.identities))
        };
        assert!(parse(&["--catalog-preview"]).is_err());
        assert!(parse(&["--catalog-preview", "--identity", "colon", "--check"]).is_err());
        assert!(parse(&["--catalog-preview", "--identity", "colon", "--transition"]).is_err());
        let (mode, identities) =
            parse(&["--catalog-preview", "--identity", "colon"]).expect("valid preview");
        assert!(mode == ExportMode::CatalogPreview);
        assert_eq!(identities, BTreeSet::from(["colon".to_string()]));
    }

    #[test]
    fn current_catalog_preview_is_deterministic_without_accepting_duplicate_authority() {
        let selected = BTreeSet::from(["colon".to_string(), "range".to_string()]);
        let legacy = synthetic_legacy(&["colon", "range"]);
        let first = build_export_with_legacy(legacy.clone(), ExportMode::CatalogPreview, &selected)
            .expect("first preview");
        let second =
            build_export_with_legacy(legacy.clone(), ExportMode::CatalogPreview, &selected)
                .expect("second preview");
        assert_eq!(first, second);
        assert_eq!(first["inventory"]["documents"], 2);
        assert_eq!(first["inventory"]["legacy_sidecars"], 2);
        assert_eq!(first["schema_version"], EXPORT_SCHEMA_VERSION);
        assert_eq!(first["preview"]["final_acceptance"], false);
        assert!(first["builtins"]
            .as_array()
            .expect("documentation rows")
            .iter()
            .all(|document| document["key"] != "overidx.subsref"
                && document["key"] != "overidx.subsasgn"));
        for document in first["builtins"]
            .as_array()
            .expect("documentation export rows")
        {
            let Some(examples) = document["examples"].as_array() else {
                continue;
            };
            for example in examples {
                assert!(example.is_object());
                assert!(example.get("fixture").is_some());
                assert!(example.get("requirements").is_some());
            }
        }
        assert!(build_export_with_legacy(
            legacy,
            ExportMode::Normal { transition: true },
            &selected,
        )
        .is_err());
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
