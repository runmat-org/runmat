use runmat_execution::ExecutableUnitEnvelope;

fn fixture(name: &str) -> runmat_execution::ExecutableUnitAdmission {
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
    // Historical payloads stay frozen. Revision admission intentionally reads
    // only their original header, not today's dependency/component structures.
    ExecutableUnitEnvelope::admission(bytes)
        .unwrap_or_else(|error| panic!("{name} fixture is invalid: {error}"))
}

#[test]
fn frozen_executable_units_keep_independent_stale_axes() {
    let current = fixture("current");
    current
        .identity
        .program
        .validate_current_compiler()
        .unwrap();
    assert_eq!(current.revisions.mir_schema, 3);
    assert_eq!(current.revisions.analysis_schema, 3);
    assert_eq!(current.revisions.bytecode_schema, 7);
    assert_eq!(current.revisions.function_registry_schema, 5);

    let pre = fixture("pre");
    assert_eq!(pre.identity.program.semantic_schema(), 1);
    assert_eq!(pre.identity.program.compiler_schema(), 1);
    assert_eq!(pre.revisions.mir_schema, 2);
    assert_eq!(pre.revisions.analysis_schema, 2);
    assert_eq!(pre.revisions.bytecode_schema, 6);

    let components = fixture("components");
    components
        .identity
        .program
        .validate_current_compiler()
        .unwrap();
    assert_eq!(components.revisions.mir_schema, 2);
    assert_eq!(components.revisions.analysis_schema, 2);
    assert_eq!(components.revisions.bytecode_schema, 6);

    let compiler = fixture("compiler");
    assert_eq!(compiler.revisions.mir_schema, 3);
    assert_eq!(compiler.revisions.analysis_schema, 3);
    assert_eq!(compiler.revisions.bytecode_schema, 7);
    assert_eq!(compiler.revisions.function_registry_schema, 5);
    assert_eq!(compiler.identity.program.compiler_schema(), 1);
    assert!(compiler
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
        unit.identity.program.validate_current_compiler().unwrap();
        assert_eq!(
            (
                unit.revisions.mir_schema,
                unit.revisions.analysis_schema,
                unit.revisions.bytecode_schema,
                unit.revisions.function_registry_schema,
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
    assert_eq!(admission.schema_version, 4);
    assert_eq!(admission.revisions.mir_schema, 2);
    assert_eq!(admission.revisions.analysis_schema, 2);
    assert_eq!(admission.revisions.bytecode_schema, 6);
    assert!(ExecutableUnitEnvelope::from_canonical_bytes(hostile.as_bytes()).is_err());
}
