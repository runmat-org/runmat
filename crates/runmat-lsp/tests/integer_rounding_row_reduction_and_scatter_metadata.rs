use lsp_types::Position;
use runmat_builtins::{BuiltinIntegerAuditKind, BuiltinIntegerInputAvailability};
use runmat_lsp::core::analysis::{analyze_document_with_compat, signature_help_at, CompatMode};

#[test]
fn rounding_row_reduction_and_scatter_metadata_is_complete() {
    for (name, form_count) in [
        ("roots", 1),
        ("rot90", 1),
        ("round", 2),
        ("rref", 2),
        ("runtests", 1),
        ("scatter3", 2),
        ("scatterhist", 3),
        ("scatterplot", 2),
    ] {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert_eq!(builtin.integer_capabilities.len(), form_count, "{name}");
        assert!(builtin.integer_audit.is_none(), "{name}");
    }
    for name in ["rowfilter", "saveas", "savefig", "saveobj"] {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert_eq!(
            builtin.integer_audit.expect("integer audit").kind,
            BuiltinIntegerAuditKind::NotApplicable,
            "{name}"
        );
    }
}

#[test]
fn metadata_distinguishes_documented_and_runmat_only_integer_roles() {
    for name in ["rot90", "round", "runtests", "scatter3"] {
        let builtin = runmat_builtins::builtin_function_by_name(name).unwrap();
        assert!(
            builtin.integer_capabilities.iter().any(|capability| {
                capability
                    .inputs
                    .iter()
                    .any(|input| input.availability == BuiltinIntegerInputAvailability::Documented)
            }),
            "{name}"
        );
    }
    for name in ["roots", "rref", "scatterhist", "scatterplot"] {
        let builtin = runmat_builtins::builtin_function_by_name(name).unwrap();
        assert!(
            builtin.integer_capabilities.iter().any(|capability| {
                capability
                    .inputs
                    .iter()
                    .any(|input| input.availability == BuiltinIntegerInputAvailability::RunMatOnly)
            }),
            "{name}"
        );
    }
}

#[test]
fn packet_signatures_are_available_to_lsp() {
    for (name, source) in [
        ("roots", "r=roots(uint8([1 0 1]));"),
        ("rot90", "r=rot90(uint64([1 2;3 4]),int8(-1));"),
        ("round", "r=round(uint64([1 2]));"),
        ("rowfilter", "f=rowfilter(['A']);"),
        ("rref", "r=rref(uint8([1 0;0 1]));"),
        (
            "runtests",
            "r=runtests('tests','IncludeSubfolders',uint8(1));",
        ),
        ("saveas", "ok=saveas(1,'x.fig');"),
        ("savefig", "ok=savefig('x.fig');"),
        ("saveobj", "s=saveobj(1);"),
        ("scatter3", "h=scatter3(uint8(1),uint8(2),uint8(3));"),
        ("scatterhist", "h=scatterhist(uint8([1 2]),uint8([2 3]));"),
        ("scatterplot", "h=scatterplot(uint8([1 2]),uint8(1));"),
    ] {
        let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
        assert!(analysis.syntax_error.is_none(), "{source}");
        assert!(analysis.lowering_error.is_none(), "{source}");
        let position = Position::new(0, source.find(name).unwrap() as u32);
        let help = signature_help_at(source, &analysis, &position)
            .unwrap_or_else(|| panic!("signature help missing for {name}: {:?}", analysis.tokens));
        assert!(!help.signatures.is_empty(), "{name}");
    }
}
