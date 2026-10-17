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

#[test]
fn admitted_incoming_workspace_is_initialized_without_seeding_new_locals() {
    let variables = std::collections::HashMap::from([("committed".to_string(), 0usize)]);
    let parsed = runmat_parser::parse("seen = committed; committed = seen + 1;")
        .expect("parse workspace continuation");
    let hir = runmat_hir::lower(&parsed, &runmat_hir::LoweringContext::new(&variables))
        .expect("lower workspace continuation HIR");
    let binding = hir
        .assembly
        .bindings
        .iter()
        .find(|binding| binding.name.0 == "committed")
        .expect("incoming binding");
    assert_eq!(binding.role, runmat_hir::BindingRole::IncomingWorkspace);
    let mir =
        crate::lowering::lower_assembly(&hir.assembly).expect("lower workspace continuation MIR");
    assert!(mir
        .bodies
        .values()
        .any(|body| body.locals.iter().any(|local| {
            local.binding == Some(binding.id)
                && local.kind == crate::MirLocalKind::IncomingWorkspace
        })));
    let found = crate::analysis::analyze_assembly(&mir).diagnostics;
    assert!(
        found.iter().all(|item| item.code != "RM-MIR0001"),
        "incoming workspace value must be assigned on entry: {found:#?}"
    );
    assert!(hir
        .assembly
        .bindings
        .iter()
        .any(|binding| binding.name.0 == "seen" && binding.role == runmat_hir::BindingRole::Local));
}

#[test]
fn declared_storage_does_not_infer_a_new_cell_from_brace_output() {
    use runmat_types::{ProgramFunctionId, ProgramPointId, RegionValueId, ValueKindFact};

    for declaration in ["persistent", "global"] {
        let source = format!(
            "function out = declared_cell_target()\n {declaration} item;\n [item{{2}}, other] = deal(9, 4);\n out = item;\n end"
        );
        let parsed = runmat_parser::parse(&source).expect("parse declared brace target");
        let hir = runmat_hir::lower(&parsed, &runmat_hir::LoweringContext::empty())
            .expect("lower declared brace target HIR");
        let mir = crate::lowering::lower_assembly(&hir.assembly)
            .expect("lower declared brace target MIR");
        let analysis = crate::analysis::analyze_assembly(&mir);
        let (function, block, position, local) = mir
            .bodies
            .iter()
            .find_map(|(function, body)| {
                body.blocks.iter().find_map(|block| {
                    block
                        .statements
                        .iter()
                        .enumerate()
                        .find_map(|(position, statement)| {
                            let crate::MirStmtKind::MultiAssign { targets, .. } = &statement.kind
                            else {
                                return None;
                            };
                            targets.targets.iter().find_map(|target| {
                                let crate::MirOutputTarget::Sequence(
                                    crate::MirSequenceTarget::CellContents {
                                        base: crate::MirPlace::Local(local),
                                        ..
                                    },
                                ) = target
                                else {
                                    return None;
                                };
                                Some((function, block, position, local))
                            })
                        })
                })
            })
            .expect("source lowered to a prepared cell destination");
        let function = ProgramFunctionId(u32::try_from(function.0).unwrap());
        let point = ProgramPointId {
            function,
            block: u32::try_from(block.id.0).unwrap(),
            position: u32::try_from(position + 1).unwrap(),
        };
        let local = analysis
            .facts_at(point)
            .expect("post-assignment source fact")
            .local(RegionValueId {
                function,
                local: u32::try_from(local.0).unwrap(),
            })
            .expect("declared root fact");
        assert_eq!(
            local.assignment,
            crate::analysis::AssignmentFact::DefinitelyAssigned
        );
        assert!(
            !matches!(
                local.fact.as_ref().map(|fact| &fact.kind),
                Some(ValueKindFact::Cell(_))
            ),
            "{declaration} declaration must not become a proven Cell from a missing payload"
        );
    }
}
