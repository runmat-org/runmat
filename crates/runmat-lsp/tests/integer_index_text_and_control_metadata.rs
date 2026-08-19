use lsp_types::Position;
use runmat_lsp::core::analysis::{analyze_document_with_compat, signature_help_at, CompatMode};

const CAPABILITY_NAMES: [&str; 11] = [
    "hold",
    "horzcat",
    "hour",
    "icdf",
    "importdata",
    "impulse",
    "ind2sub",
    "ind2word",
    "inputname",
    "insertAfter",
    "insertBefore",
];

const INAPPLICABLE_NAMES: [&str; 2] = ["htmlTree", "input"];

#[test]
fn index_text_and_control_integer_metadata_is_explicit() {
    for name in CAPABILITY_NAMES {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(!builtin.integer_capabilities.is_empty(), "{name}");
    }
    for name in INAPPLICABLE_NAMES {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(builtin.integer_audit.is_some(), "{name}");
    }
}

#[test]
fn index_text_and_control_signatures_are_visible_to_lsp() {
    for (name, source) in [
        ("hold", "hold(1);"),
        ("horzcat", "a = horzcat(uint8(1),uint8(2));"),
        ("hour", "h = hour(739000);"),
        ("htmlTree", "t = htmlTree('<p>x</p>');"),
        ("icdf", "x = icdf('Normal',0.5);"),
        ("importdata", "a = importdata('a.csv',',',uint16(1));"),
        (
            "impulse",
            "sys = tf([1],[1 1]); y = impulse(sys,uint16(2));",
        ),
        ("ind2sub", "[r,c] = ind2sub(uint16([2 3]),uint16(4));"),
        (
            "ind2word",
            "enc = wordEncoding([\"one\" \"two\"]); w = ind2word(enc,uint16(1));",
        ),
        ("input", "x = input('value: ');"),
        ("inputname", "n = inputname(uint16(1));"),
        ("insertAfter", "s = insertAfter('ab',uint16(1),'x');"),
        ("insertBefore", "s = insertBefore('ab',uint16(1),'x');"),
    ] {
        let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
        assert!(analysis.syntax_error.is_none(), "{source}");
        assert!(analysis.lowering_error.is_none(), "{source}");
        let position = Position::new(0, source.find(name).expect("builtin") as u32);
        let help = signature_help_at(source, &analysis, &position).expect("signature help");
        assert!(!help.signatures.is_empty(), "{name}");
    }
}
