use lsp_types::Position;
use runmat_builtins::{
    BuiltinIntegerBackendRule, BuiltinIntegerInputAvailability, BuiltinIntegerOutputClassRule,
};
use runmat_lsp::core::analysis::{analyze_document_with_compat, signature_help_at, CompatMode};

const PACKET: [&str; 8] = [
    "mesh",
    "meshc",
    "openfig",
    "opentoline",
    "pan",
    "parula",
    "patch",
    "pause",
];

#[test]
fn graphics_navigation_and_timing_integer_dispositions_are_explicit() {
    for name in PACKET {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(
            !builtin.integer_capabilities.is_empty() || builtin.integer_audit.is_some(),
            "{name}"
        );
    }
    let openfig = runmat_builtins::builtin_function_by_name("openfig").expect("openfig");
    assert!(openfig.integer_capabilities.is_empty());
    assert_eq!(
        openfig.integer_audit.expect("openfig audit").kind,
        runmat_builtins::BuiltinIntegerAuditKind::NotApplicable
    );
}

#[test]
fn documented_graphics_and_timing_roles_cover_all_integer_classes() {
    for name in ["mesh", "meshc", "parula", "pause"] {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(builtin.integer_capabilities.iter().all(|capability| {
            capability.inputs.iter().all(|input| {
                input.classes.len() == 8
                    && input.availability == BuiltinIntegerInputAvailability::Documented
            })
        }));
    }
    for name in ["mesh", "meshc", "patch"] {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(builtin
            .integer_capabilities
            .iter()
            .any(|capability| capability.backend == BuiltinIntegerBackendRule::GatherFallback));
    }
    let parula = runmat_builtins::builtin_function_by_name("parula").expect("parula");
    assert_eq!(
        parula.integer_capabilities[0].output_class,
        BuiltinIntegerOutputClassRule::Double
    );
}

#[test]
fn runmat_only_structural_forms_have_independent_extension_records() {
    for (name, ids) in [
        (
            "opentoline",
            &[
                "opentoline-integer-line",
                "opentoline-integer-column",
                "opentoline-resident-position",
            ][..],
        ),
        ("pan", &["pan-integer-graphics-target"][..]),
        ("patch", &["patch-integer-axes-handle"][..]),
        ("pause", &["pause-gpu-input"][..]),
    ] {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        for id in ids {
            assert!(
                builtin
                    .extensions
                    .iter()
                    .any(|extension| extension.id == *id),
                "{name}: {id}"
            );
        }
    }
}

#[test]
fn packet_signatures_are_visible_to_lsp() {
    for (name, source) in [
        ("mesh", "h=mesh(uint16([1 2;3 4]));"),
        ("meshc", "h=meshc(uint16([1 2;3 4]));"),
        ("openfig", "h=openfig('plot.fig');"),
        ("opentoline", "opentoline('file.m',1);"),
        ("pan", "p=pan();"),
        ("parula", "c=parula(uint16(6));"),
        ("patch", "h=patch([0 1 0],[0 0 1]);"),
        ("pause", "pause(uint16(0));"),
    ] {
        let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
        assert!(analysis.syntax_error.is_none(), "{source}");
        assert!(analysis.lowering_error.is_none(), "{source}");
        let position = Position::new(0, source.find(name).expect("builtin") as u32);
        let help = signature_help_at(source, &analysis, &position).expect("signature help");
        assert!(!help.signatures.is_empty(), "{name}");
    }
}
