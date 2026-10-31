use lsp_types::{Position, SignatureHelp};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinIntegerAuditKind, BuiltinIntegerBackendRule,
    BuiltinIntegerClass,
};
use runmat_lsp::core::analysis::{
    analyze_document_with_compat, completion_at, signature_help_at, CompatMode,
};

const MESHING_BUILTINS: [&str; 9] = [
    "meshing.generate",
    "meshing.load",
    "meshing.save",
    "meshing.nodes",
    "meshing.elements",
    "meshing.boundaryFaces",
    "meshing.regions",
    "meshing.quality",
    "meshing.evidence",
];

const INTEGER_CLASSES: [BuiltinIntegerClass; 8] = [
    BuiltinIntegerClass::Int8,
    BuiltinIntegerClass::Int16,
    BuiltinIntegerClass::Int32,
    BuiltinIntegerClass::Int64,
    BuiltinIntegerClass::Uint8,
    BuiltinIntegerClass::Uint16,
    BuiltinIntegerClass::Uint32,
    BuiltinIntegerClass::Uint64,
];

fn signature_help(source: &str, lookup: &str) -> SignatureHelp {
    let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
    assert!(
        analysis.syntax_error.is_none(),
        "{source}: {:?}",
        analysis
            .syntax_error
            .as_ref()
            .map(|error| error.message.as_str())
    );
    assert!(
        analysis.lowering_error.is_none(),
        "{source}: {:?}",
        analysis
            .lowering_error
            .as_ref()
            .map(|error| error.message.as_str())
    );
    assert!(
        analysis.compile_error.is_none(),
        "{source}: {:?}",
        analysis
            .compile_error
            .as_ref()
            .map(|error| error.message.as_str())
    );
    let column = source.find(lookup).expect("builtin call") as u32;
    signature_help_at(source, &analysis, &Position::new(0, column))
        .unwrap_or_else(|| panic!("descriptor-backed signature help for {lookup}"))
}

#[test]
fn meshing_descriptors_are_visible_to_signature_help() {
    for (lookup, source) in [
        ("generate", "asset = 1; out = meshing.generate(asset);"),
        (
            "load",
            "out = meshing.load(\"mesh.cbor\", \"evidence.cbor\");",
        ),
        (
            "save",
            "mesh = 1; out = meshing.save(mesh, \"mesh.cbor\", \"evidence.cbor\");",
        ),
        ("nodes", "mesh = 1; out = meshing.nodes(mesh);"),
        ("elements", "mesh = 1; out = meshing.elements(mesh);"),
        (
            "boundaryFaces",
            "mesh = 1; out = meshing.boundaryFaces(mesh);",
        ),
        ("regions", "mesh = 1; out = meshing.regions(mesh);"),
        ("quality", "mesh = 1; out = meshing.quality(mesh);"),
        ("evidence", "mesh = 1; out = meshing.evidence(mesh);"),
    ] {
        let help = signature_help(source, lookup);
        assert!(
            help.signatures
                .iter()
                .any(|signature| signature.label.contains(&format!("meshing.{lookup}"))),
            "{lookup} must expose its namespaced signature"
        );
    }
}

#[test]
fn meshing_packet_is_public_and_has_settled_integer_metadata() {
    let source = "x = 1;";
    let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
    let completions = completion_at(source, &analysis, &Position::new(0, 0));

    for name in MESHING_BUILTINS {
        let builtin = runmat_builtins::public_builtin_catalog_entry_by_name(name)
            .unwrap_or_else(|| panic!("public canonical builtin {name}"));
        let descriptor = builtin.descriptor;
        assert_eq!(
            descriptor.completion_policy,
            BuiltinCompletionPolicy::Public,
            "{name} completion policy"
        );
        assert!(!descriptor.signatures.is_empty(), "{name} signatures");
        assert!(builtin.extensions.is_empty(), "{name} is RunMat-native");
        assert!(
            completions
                .iter()
                .any(|item| item.label.eq_ignore_ascii_case(name)),
            "public completion for {name}"
        );

        if name == "meshing.generate" {
            assert!(builtin.integer_audit.is_none());
            assert_eq!(builtin.integer_capabilities.len(), 1);
            let capability = &builtin.integer_capabilities[0];
            assert_eq!(capability.backend, BuiltinIntegerBackendRule::HostOnly);
            assert_eq!(capability.inputs.len(), 2);
            for input in capability.inputs {
                assert_eq!(
                    input.classes, &INTEGER_CLASSES,
                    "{} integer classes",
                    input.name
                );
            }
        } else {
            assert!(builtin.integer_capabilities.is_empty(), "{name}");
            let audit = builtin
                .integer_audit
                .unwrap_or_else(|| panic!("{name} integer audit"));
            assert_eq!(audit.kind, BuiltinIntegerAuditKind::NotApplicable, "{name}");
            assert!(!audit.notes.is_empty(), "{name} audit rationale");
        }
    }
}
