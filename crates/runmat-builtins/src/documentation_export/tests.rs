use super::*;

#[test]
fn presentation_retains_full_rows_but_cannot_be_mistaken_for_qualification() {
    let selected = BTreeSet::from([
        "minreal".to_string(),
        "legacy_documentation_fixture".to_string(),
    ]);
    let legacy = BTreeMap::from([(
        "legacy_documentation_fixture".to_string(),
        json!({"title":"Legacy fixture", "examples":["An explanation without an executable fixture."]}),
    )]);
    let strict = build_export_with_legacy(
        legacy.clone(),
        ExportMode::Normal { transition: true },
        &selected,
    )
    .unwrap();
    let mut presentation = project_export(
        legacy,
        ExportMode::Normal { transition: true },
        &selected,
        ValidationPurpose::PresentationOnly,
    )
    .unwrap();
    assert_eq!(
        presentation["presentation"],
        json!({"kind":"editor_prose", "executable_qualification":false})
    );
    presentation.as_object_mut().unwrap().remove("presentation");
    assert_eq!(
        presentation, strict,
        "presentation must not trim or repair rows"
    );
    for purpose in [
        ValidationPurpose::Qualified,
        ValidationPurpose::PresentationOnly,
    ] {
        assert!(project_export(
            synthetic_legacy(&["colon"]),
            ExportMode::Normal { transition: true },
            &BTreeSet::from(["colon".to_string()]),
            purpose,
        )
        .unwrap_err()
        .to_string()
        .contains("canonical catalog documentation and a legacy sidecar"));
    }
}

#[test]
fn unavailable_package_assets_are_not_a_silent_canonical_only_fallback() {
    let not_a_directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    for error in [
        export_from_directory(
            &not_a_directory,
            ExportMode::Normal { transition: true },
            &BTreeSet::new(),
        )
        .unwrap_err(),
        export_presentation_from_directory(&not_a_directory, true).unwrap_err(),
    ] {
        assert!(error
            .to_string()
            .contains("authoritative builtin documentation assets unavailable"));
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn canonical_presentation_matches_empty_directory_projection_in_every_field() {
    let empty_directory = std::env::temp_dir().join(format!(
        "runmat-canonical-presentation-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time")
            .as_nanos(),
    ));
    std::fs::create_dir(&empty_directory).expect("create isolated empty directory");
    let directory_projection = export_presentation_from_directory(&empty_directory, false);
    let canonical_projection = export_canonical_presentation();
    std::fs::remove_dir(&empty_directory).expect("remove isolated empty directory");

    let canonical_projection = canonical_projection.expect("complete canonical presentation");
    assert_eq!(
        canonical_projection,
        directory_projection.expect("empty directory presentation"),
        "canonical source choice must preserve every projected field and row",
    );
    assert_eq!(
        canonical_projection["presentation"],
        json!({"kind":"editor_prose", "executable_qualification":false}),
    );
    assert!(canonical_projection.get("preview").is_none());
    // This compares source projections, not a package/publication rehearsal.
}

fn synthetic_legacy(names: &[&str]) -> BTreeMap<String, Value> {
    names
        .iter()
        .map(|name| (name.to_string(), json!({ "title": name })))
        .collect()
}

#[test]
fn constant_documentation_exports_bare_value_examples_without_callable_contracts() {
    use crate::{
        BuiltinCatalogProvenance, BuiltinConstantKind, BuiltinExample, BuiltinExampleCompatibility,
        BuiltinExampleFixture, BuiltinExampleHarness, BuiltinExampleRequirements,
        BuiltinExampleVerification,
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
    assert!(!validate_builtin_documentation(entry.name, &BuiltinDocumentation::EMPTY).is_empty());
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
fn current_catalog_preview_is_deterministic_without_accepting_duplicate_authority() {
    let selected = BTreeSet::from(["colon".to_string(), "range".to_string()]);
    let legacy = synthetic_legacy(&["colon", "range"]);
    let first = build_export_with_legacy(legacy.clone(), ExportMode::CatalogPreview, &selected)
        .expect("first preview");
    let second = build_export_with_legacy(legacy.clone(), ExportMode::CatalogPreview, &selected)
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
    assert!(
        build_export_with_legacy(legacy, ExportMode::Normal { transition: true }, &selected,)
            .is_err()
    );
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
