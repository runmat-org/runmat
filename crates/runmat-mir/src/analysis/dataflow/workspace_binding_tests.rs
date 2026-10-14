use super::*;

fn diagnostics(source: &str) -> Vec<MirDiagnostic> {
    let parsed = runmat_parser::parse(source).expect("parse workspace declaration");
    let hir = runmat_hir::lower(&parsed, &runmat_hir::LoweringContext::empty())
        .expect("lower workspace declaration HIR");
    let mir =
        crate::lowering::lower_assembly(&hir.assembly).expect("lower workspace declaration MIR");
    crate::analysis::analyze_assembly(&mir).diagnostics
}

#[test]
fn global_and_persistent_declarations_assign_only_their_exact_bindings() {
    for source in [
        "function out = declared_global()\n global item;\n out = item;\n end",
        "function out = declared_persistent()\n persistent item;\n out = item;\n end",
    ] {
        let found = diagnostics(source);
        assert!(
            found.iter().all(|item| item.code != "RM-MIR0001"),
            "declared storage is initialized by its declaration: {found:#?}"
        );
    }

    let mut state = [InitFact::Unassigned, InitFact::Unassigned];
    mark_declared_storage(
        &runmat_hir::WorkspaceEffect::MutatesGlobal,
        &[MirLocalId(0)],
        &mut state,
    );
    assert_eq!(state, [InitFact::DefinitelyAssigned, InitFact::Unassigned]);
    mark_declared_storage(
        &runmat_hir::WorkspaceEffect::ReadsWorkspace,
        &[MirLocalId(1)],
        &mut state,
    );
    assert_eq!(state, [InitFact::DefinitelyAssigned, InitFact::Unassigned]);
}

#[test]
fn ordinary_local_still_requires_assignment() {
    let found = diagnostics("function out = ordinary()\n seen = out;\n out = 1;\n end");
    assert!(found.iter().any(|item| item.code == "RM-MIR0001"));
}
