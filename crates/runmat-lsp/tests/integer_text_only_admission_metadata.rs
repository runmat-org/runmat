use lsp_types::Position;
use runmat_lsp::core::analysis::{analyze_document_with_compat, signature_help_at, CompatMode};

const INAPPLICABLE_PACKET: [(&str, &str); 21] = [
    ("normalizeWords", "x=normalizeWords('running');"),
    ("readWordEmbedding", "e=readWordEmbedding('vectors.vec');"),
    ("regexp", "i=regexp('abc','b');"),
    ("regexpi", "i=regexpi('AbC','a');"),
    ("replace", "s=replace('abc','b','x');"),
    ("rethrow", "try; error('x'); catch err; rethrow(err); end;"),
    ("splitlines", "s=splitlines('a');"),
    ("strip", "s=strip(' a ');"),
    ("str2double", "x=str2double('3');"),
    ("str2func", "f=str2func('sin');"),
    ("str2num", "x=str2num('3');"),
    ("strcat", "s=strcat('a','b');"),
    ("strcmp", "tf=strcmp('a','b');"),
    ("strcmpi", "tf=strcmpi('a','A');"),
    ("strjoin", "s=strjoin(['a';'b'],',');"),
    ("strjust", "s=strjust('a');"),
    ("strlength", "n=strlength('a');"),
    ("strrep", "s=strrep('abc','b','x');"),
    ("strsplit", "s=strsplit('a b');"),
    ("strtok", "s=strtok('a b');"),
    ("strtrim", "s=strtrim(' a ');"),
];

const INTEGER_PACKET: [(&str, &str); 4] = [
    ("string", "s=string(uint64(3));"),
    ("strings", "s=strings(uint8(2),uint16(3));"),
    ("strncmp", "tf=strncmp('abc','abd',uint32(2));"),
    ("strncmpi", "tf=strncmpi('ABC','abd',int64(2));"),
];

#[test]
fn text_only_packet_exposes_integer_inapplicable_metadata() {
    for (name, _) in INAPPLICABLE_PACKET {
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
fn string_conversion_and_prefix_comparison_expose_all_integer_classes() {
    for (name, _) in INTEGER_PACKET {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(builtin.integer_audit.is_none(), "{name}");
        assert!(!builtin.integer_capabilities.is_empty(), "{name}");
        for capability in builtin.integer_capabilities {
            for input in capability.inputs {
                assert_eq!(input.classes.len(), 8, "{name}: {}", capability.form);
            }
        }
    }
}

#[test]
fn text_only_packet_signatures_remain_visible_to_lsp() {
    for (name, source) in INAPPLICABLE_PACKET.into_iter().chain(INTEGER_PACKET) {
        let analysis = analyze_document_with_compat(source, CompatMode::Matlab);
        assert!(analysis.syntax_error.is_none(), "{source}");
        assert!(analysis.lowering_error.is_none(), "{source}");
        let position = Position::new(0, source.find(name).expect("builtin") as u32);
        let help = signature_help_at(source, &analysis, &position).expect("signature help");
        assert!(!help.signatures.is_empty(), "{name}");
    }
}
