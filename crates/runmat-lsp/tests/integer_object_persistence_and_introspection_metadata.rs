use lsp_types::Position;
use runmat_builtins::{
    BuiltinIntegerAuditKind, BuiltinIntegerInputAvailability, BuiltinIntegerOutputClassRule,
};
use runmat_lsp::core::analysis::{analyze_document_with_compat, signature_help_at, CompatMode};

const CAPABILITY_NAMES: [&str; 7] = [
    "load",
    "loadobj",
    "memmapfile",
    "metaclass",
    "numArgumentsFromSubscript",
    "numel",
    "orderfields",
];

const INAPPLICABLE_NAMES: [&str; 3] = [
    "matlab.metadata.DynamicProperty.delete",
    "memoize",
    "notify",
];

#[test]
fn persistence_and_introspection_integer_metadata_is_exhaustive() {
    for name in CAPABILITY_NAMES {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(!builtin.integer_capabilities.is_empty(), "{name}");
        assert!(builtin.integer_audit.is_none(), "{name}");
        for capability in builtin.integer_capabilities {
            for input in capability.inputs {
                assert_eq!(input.classes.len(), 8, "{name}: {}", input.name);
            }
        }
    }
    for name in INAPPLICABLE_NAMES {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(builtin.integer_capabilities.is_empty(), "{name}");
        assert_eq!(
            builtin.integer_audit.expect("integer audit").kind,
            BuiltinIntegerAuditKind::NotApplicable,
            "{name}"
        );
    }
}

#[test]
fn loadobj_numel_and_memmapfile_extensions_are_independently_registered() {
    for (name, extension_ids) in [
        ("loadobj", &["loadobj-plain-payload-passthrough"][..]),
        ("numel", &["numel-dimension-selectors"][..]),
        (
            "memmapfile",
            &[
                "memmapfile-integer-property-controls",
                "memmapfile-explicit-gpu-argument",
            ][..],
        ),
    ] {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        for extension_id in extension_ids {
            assert!(
                builtin
                    .extensions
                    .iter()
                    .any(|extension| extension.id == *extension_id),
                "{name}: {extension_id}"
            );
        }
    }

    let loadobj = runmat_builtins::builtin_function_by_name("loadobj").expect("loadobj");
    assert!(loadobj.integer_capabilities.iter().any(|capability| {
        capability.inputs.iter().any(|input| {
            input.name == "integer_a"
                && input.availability == BuiltinIntegerInputAvailability::RunMatOnly
        }) && capability.output_class == BuiltinIntegerOutputClassRule::PreserveInput
    }));
}

#[test]
fn persistence_and_introspection_signatures_are_visible_to_lsp() {
    for (name, source) in [
        ("load", "s=load('state.mat');"),
        ("loadobj", "b=loadobj(struct());"),
        ("memmapfile", "m=memmapfile('state.bin');"),
        ("metaclass", "m=metaclass(uint64(1));"),
        ("numel", "n=numel(uint32([1 2 3]));"),
        ("orderfields", "s=struct();o=orderfields(s,uint8(1));"),
    ] {
        let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
        assert!(analysis.syntax_error.is_none(), "{source}");
        assert!(analysis.lowering_error.is_none(), "{source}");
        let position = Position::new(0, source.find(name).expect("builtin") as u32);
        let help = signature_help_at(source, &analysis, &position).expect("signature help");
        assert!(!help.signatures.is_empty(), "{name}");
    }
}
