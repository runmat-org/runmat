use lsp_types::Position;
use runmat_lsp::core::analysis::{analyze_document_with_compat, signature_help_at, CompatMode};

const PACKET: [(&str, &str); 8] = [
    ("normalizeWords", "x=normalizeWords('running');"),
    ("readWordEmbedding", "e=readWordEmbedding('vectors.vec');"),
    ("regexp", "i=regexp('abc','b');"),
    ("regexpi", "i=regexpi('AbC','a');"),
    ("replace", "s=replace('abc','b','x');"),
    ("rethrow", "try; error('x'); catch err; rethrow(err); end;"),
    ("splitlines", "s=splitlines('a');"),
    ("strip", "s=strip(' a ');"),
];

#[test]
fn text_only_packet_exposes_integer_inapplicable_metadata() {
    for (name, _) in PACKET {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(builtin.integer_capabilities.is_empty(), "{name}");
        assert_eq!(
            builtin.integer_audit.expect("integer audit").kind,
            runmat_builtins::BuiltinIntegerAuditKind::NotApplicable,
            "{name}"
        );
    }
}

#[test]
fn text_only_packet_signatures_remain_visible_to_lsp() {
    for (name, source) in PACKET {
        let analysis = analyze_document_with_compat(source, CompatMode::Matlab);
        assert!(analysis.syntax_error.is_none(), "{source}");
        assert!(analysis.lowering_error.is_none(), "{source}");
        let position = Position::new(0, source.find(name).expect("builtin") as u32);
        let help = signature_help_at(source, &analysis, &position).expect("signature help");
        assert!(!help.signatures.is_empty(), "{name}");
    }
}
