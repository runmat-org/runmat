use runmat_execution::ExecutableUnitEnvelope;

fn fixture(name: &str) -> ExecutableUnitEnvelope {
    let bytes: &[u8] = match name {
        "pre" => include_bytes!("fixtures/executable-unit-pre-struct-array.json"),
        "components" => include_bytes!("fixtures/executable-unit-stale-components.json"),
        "compiler" => include_bytes!("fixtures/executable-unit-compiler-1.json"),
        "current" => include_bytes!("fixtures/executable-unit-current.json"),
        "mir" => include_bytes!("fixtures/executable-unit-mir-2.json"),
        "analysis" => include_bytes!("fixtures/executable-unit-analysis-2.json"),
        "bytecode" => include_bytes!("fixtures/executable-unit-bytecode-6.json"),
        "registry" => include_bytes!("fixtures/executable-unit-registry-4.json"),
        _ => unreachable!(),
    };
    ExecutableUnitEnvelope::from_canonical_bytes(bytes)
        .unwrap_or_else(|error| panic!("{name} fixture is invalid: {error}"))
}

#[test]
fn frozen_executable_units_keep_independent_stale_axes() {
    let current = fixture("current");
    current
        .manifest
        .identity
        .program
        .validate_current_compiler()
        .unwrap();
    assert_eq!(current.manifest.revisions.mir_schema, 3);
    assert_eq!(current.manifest.revisions.analysis_schema, 3);
    assert_eq!(current.manifest.revisions.bytecode_schema, 7);
    assert_eq!(current.manifest.revisions.function_registry_schema, 5);

    let pre = fixture("pre");
    assert_eq!(pre.manifest.identity.program.semantic_schema(), 1);
    assert_eq!(pre.manifest.identity.program.compiler_schema(), 1);
    assert_eq!(pre.manifest.revisions.mir_schema, 2);
    assert_eq!(pre.manifest.revisions.analysis_schema, 2);
    assert_eq!(pre.manifest.revisions.bytecode_schema, 6);

    let components = fixture("components");
    components
        .manifest
        .identity
        .program
        .validate_current_compiler()
        .unwrap();
    assert_eq!(components.manifest.revisions.mir_schema, 2);
    assert_eq!(components.manifest.revisions.analysis_schema, 2);
    assert_eq!(components.manifest.revisions.bytecode_schema, 6);

    let compiler = fixture("compiler");
    assert_eq!(compiler.manifest.revisions.mir_schema, 3);
    assert_eq!(compiler.manifest.revisions.analysis_schema, 3);
    assert_eq!(compiler.manifest.revisions.bytecode_schema, 7);
    assert_eq!(compiler.manifest.revisions.function_registry_schema, 5);
    assert_eq!(compiler.manifest.identity.program.compiler_schema(), 1);
    assert!(compiler
        .manifest
        .identity
        .program
        .validate_current_compiler()
        .is_err());

    for (name, expected) in [
        ("mir", (2, 3, 7, 5)),
        ("analysis", (3, 2, 7, 5)),
        ("bytecode", (3, 3, 6, 5)),
        ("registry", (3, 3, 7, 4)),
    ] {
        let unit = fixture(name);
        unit.manifest
            .identity
            .program
            .validate_current_compiler()
            .unwrap();
        assert_eq!(
            (
                unit.manifest.revisions.mir_schema,
                unit.manifest.revisions.analysis_schema,
                unit.manifest.revisions.bytecode_schema,
                unit.manifest.revisions.function_registry_schema,
            ),
            expected,
            "{name} fixture must isolate one stale component axis"
        );
    }
}

#[test]
fn revision_admission_does_not_decode_component_payloads() {
    let fixture = include_str!("fixtures/executable-unit-stale-components.json");
    let payload_start = fixture
        .find("\"payloads\":")
        .expect("frozen fixture contains its opaque payload section");
    let hostile = format!(
        "{}\"payloads\":\"changed-schema-bytes\"}}",
        &fixture[..payload_start]
    );

    let admission = ExecutableUnitEnvelope::admission(hostile.as_bytes())
        .expect("the bounded revision header does not decode opaque components");
    assert_eq!(admission.schema_version, 3);
    assert_eq!(admission.revisions.mir_schema, 2);
    assert_eq!(admission.revisions.analysis_schema, 2);
    assert_eq!(admission.revisions.bytecode_schema, 6);
    assert!(ExecutableUnitEnvelope::from_canonical_bytes(hostile.as_bytes()).is_err());
}
