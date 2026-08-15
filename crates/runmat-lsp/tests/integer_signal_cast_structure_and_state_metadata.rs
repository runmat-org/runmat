use lsp_types::Position;
use runmat_builtins::{BuiltinIntegerAuditKind, BuiltinIntegerInputAvailability};
use runmat_lsp::core::analysis::{analyze_document_with_compat, signature_help_at, CompatMode};

const CAPABILITY_PACKET: [(&str, usize, &str); 14] = [
    ("sawtooth", 2, "y=sawtooth(uint8([0 1]));"),
    ("set", 1, "set(gcf,'Visible',uint8(1));"),
    ("setfield", 2, "s=setfield(struct(),'x',uint64(1));"),
    ("setpref", 1, "setpref('g','p',uint32(1));"),
    ("sgtitle", 2, "h=sgtitle(uint64(1));"),
    ("sign", 1, "y=sign(int16([-1 0 1]));"),
    ("sin", 1, "y=sin(uint8(1));"),
    ("sinc", 1, "y=sinc(uint8(1));"),
    ("sind", 1, "y=sind(uint8(30));"),
    ("single", 1, "y=single(uint64(1));"),
    ("sinh", 1, "y=sinh(uint8(1));"),
    ("sinpi", 1, "y=sinpi(uint64(1));"),
    ("size", 2, "y=size(uint8([1 2]),uint8(2));"),
    ("sparse", 4, "y=sparse(uint8(1),uint8(1),uint8(2));"),
];

#[test]
fn signal_cast_structure_and_state_metadata_is_explicit() {
    for (name, expected_forms, _) in CAPABILITY_PACKET {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert_eq!(builtin.integer_capabilities.len(), expected_forms, "{name}");
        assert!(builtin.integer_audit.is_none(), "{name}");
        assert!(builtin
            .integer_capabilities
            .iter()
            .flat_map(|capability| capability.inputs)
            .all(|input| input.classes.len() == 8));
    }
    for name in ["second", "sendmail"] {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(builtin.integer_capabilities.is_empty(), "{name}");
        assert_eq!(
            builtin.integer_audit.expect("integer audit").kind,
            BuiltinIntegerAuditKind::NotApplicable,
            "{name}"
        );
    }
    let sin = runmat_builtins::builtin_function_by_name("sin").expect("sin");
    assert_eq!(
        sin.integer_capabilities[0].inputs[0].availability,
        BuiltinIntegerInputAvailability::RunMatOnly
    );
    let sign = runmat_builtins::builtin_function_by_name("sign").expect("sign");
    assert_eq!(
        sign.integer_capabilities[0].inputs[0].availability,
        BuiltinIntegerInputAvailability::Documented
    );
}

#[test]
fn signal_cast_structure_and_state_signatures_remain_visible_to_lsp() {
    for (name, _, source) in CAPABILITY_PACKET {
        let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
        assert!(analysis.syntax_error.is_none(), "{name}: {source}");
        assert!(analysis.lowering_error.is_none(), "{name}: {source}");
        let position = Position::new(0, source.find(name).expect("builtin") as u32);
        let help = signature_help_at(source, &analysis, &position).expect("signature help");
        assert!(!help.signatures.is_empty(), "{name}");
    }
}
