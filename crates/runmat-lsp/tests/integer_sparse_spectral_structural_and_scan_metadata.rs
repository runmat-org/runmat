use lsp_types::Position;
use runmat_builtins::BuiltinIntegerInputAvailability;
use runmat_lsp::core::analysis::{analyze_document_with_compat, signature_help_at, CompatMode};

const CAPABILITY_PACKET: [(&str, usize, &str); 16] = [
    ("spdiags", 2, "y=spdiags(uint8([1 2]));"),
    ("spectrogram", 2, "y=spectrogram(uint8([1 2 3]));"),
    ("speye", 1, "y=speye(uint8(3));"),
    ("sphere", 1, "y=sphere(uint8(4));"),
    ("spline", 3, "y=spline(uint8([1 2]),uint8([3 4]));"),
    ("split", 1, "y=split(\"a,b\",\",\",uint8(2));"),
    (
        "splitapply",
        2,
        "y=splitapply(@sum,uint8([1;2]),uint8([1;1]));",
    ),
    ("spones", 1, "y=spones(uint8([1 0]));"),
    ("sprand", 2, "y=sprand(uint8(2),uint8(2),1);"),
    ("sprintf", 1, "y=sprintf('%u',uint64(1));"),
    ("sqrt", 1, "y=sqrt(uint8(4));"),
    ("square", 2, "y=square(uint8([0 1]));"),
    ("squareform", 1, "y=squareform(uint8([1 2 3]));"),
    ("squeeze", 1, "y=squeeze(uint8(ones(1,1,2)));"),
    ("ss", 2, "y=ss(uint8(1),1,1,1);"),
    ("sscanf", 1, "y=sscanf(\"1\",\"%ld\",uint8(1));"),
];

#[test]
fn sparse_spectral_structural_and_scan_metadata_is_explicit() {
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

    for name in ["sqrt", "squareform", "ss"] {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(builtin
            .integer_capabilities
            .iter()
            .flat_map(|capability| capability.inputs)
            .any(|input| input.availability == BuiltinIntegerInputAvailability::RunMatOnly));
    }
    for name in ["speye", "sprintf", "squeeze", "sscanf"] {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(builtin
            .integer_capabilities
            .iter()
            .flat_map(|capability| capability.inputs)
            .any(|input| input.availability == BuiltinIntegerInputAvailability::Documented));
    }
}

#[test]
fn sparse_spectral_structural_and_scan_signatures_remain_visible_to_lsp() {
    for (name, _, source) in CAPABILITY_PACKET {
        let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
        assert!(analysis.syntax_error.is_none(), "{name}: {source}");
        assert!(analysis.lowering_error.is_none(), "{name}: {source}");
        let position = Position::new(0, source.find(name).expect("builtin") as u32);
        let help = signature_help_at(source, &analysis, &position).expect("signature help");
        assert!(!help.signatures.is_empty(), "{name}");
    }
}
