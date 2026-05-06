use runmat_hir::{lower, HirCallableRef, LoweringContext};
use runmat_mir::{
    analysis::{
        analyze_assembly, analyze_body, analyze_liveness, analyze_spawn_boundaries,
        diagnose_uninitialized_reads, summarize_body, summarize_spawn_safety, AnalysisStore,
        InitFact, MirLocalKey,
    },
    lowering::lower_assembly,
    AsyncBehaviorFact, MirAggregateKind, MirBody, MirLocalKind, MirOperand, MirOutputTarget,
    MirPlace, MirRvalue, MirStmtKind, MirTerminatorKind,
};

fn lower_mir(src: &str) -> runmat_mir::MirAssembly {
    let ast = runmat_parser::parse(src).unwrap();
    let hir = lower(&ast, &LoweringContext::empty()).unwrap();
    lower_assembly(&hir.assembly).unwrap()
}

fn analyze_single_body(src: &str) -> (MirBody, AnalysisStore) {
    let mir = lower_mir(src);
    let body = mir.bodies.values().next().unwrap().clone();
    let mut store = AnalysisStore::default();
    analyze_body(&body, &mut store);
    (body, store)
}

fn first_local_of_kind(body: &MirBody, kind: MirLocalKind) -> runmat_mir::MirLocalId {
    body.locals
        .iter()
        .find(|local| local.kind == kind)
        .unwrap()
        .id
}

#[test]
fn simple_function_lowers_to_single_block_with_binding_locals() {
    let mir = lower_mir("function y = f(x); y = x + 1; end");
    let body = mir.bodies.values().next().unwrap();

    assert_eq!(body.blocks.len(), 1);
    assert_eq!(body.locals.len(), 4);
    assert_eq!(body.source_map.locals.len(), body.locals.len());
    assert_eq!(body.blocks[0].statements.len(), 1);
    assert!(matches!(
        body.blocks[0].statements[0].kind,
        MirStmtKind::Assign {
            value: MirRvalue::Binary(_, _, _),
            ..
        }
    ));
    assert!(matches!(
        body.blocks[0].terminator.kind,
        MirTerminatorKind::Return(ref outputs) if outputs.len() == 1
    ));
}

#[test]
fn dataflow_marks_parameters_and_assigned_outputs_definitely_assigned() {
    let (body, store) = analyze_single_body("function y = f(x); y = x; end");
    let param = first_local_of_kind(&body, MirLocalKind::Parameter);
    let output = first_local_of_kind(&body, MirLocalKind::Output);

    assert_eq!(
        store
            .mir_locals
            .get(&MirLocalKey {
                function: body.function,
                local: param,
            })
            .unwrap()
            .initialized,
        InitFact::DefinitelyAssigned
    );
    assert_eq!(
        store
            .mir_locals
            .get(&MirLocalKey {
                function: body.function,
                local: output,
            })
            .unwrap()
            .initialized,
        InitFact::DefinitelyAssigned
    );
}

#[test]
fn dataflow_joins_branch_assignment_as_maybe_assigned() {
    let (body, store) = analyze_single_body("function y = f(c); if c; y = 1; end; end");
    let output = first_local_of_kind(&body, MirLocalKind::Output);

    assert_eq!(
        store
            .mir_locals
            .get(&MirLocalKey {
                function: body.function,
                local: output,
            })
            .unwrap()
            .initialized,
        InitFact::MaybeAssigned
    );
}

#[test]
fn dataflow_widens_loop_assignment_as_maybe_assigned() {
    let (body, store) = analyze_single_body("function y = f(c); while c; y = 1; end; end");
    let output = first_local_of_kind(&body, MirLocalKind::Output);

    assert_eq!(
        store
            .mir_locals
            .get(&MirLocalKey {
                function: body.function,
                local: output,
            })
            .unwrap()
            .initialized,
        InitFact::MaybeAssigned
    );
}

#[test]
fn diagnostics_report_unassigned_local_read() {
    let mir = lower_mir("function y = f(); z = y; y = 1; end");
    let body = mir.bodies.values().next().unwrap();

    let diagnostics = diagnose_uninitialized_reads(body);

    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "RM-MIR0001"
            && diagnostic.category.as_deref() == Some("definite-assignment")
            && diagnostic.primary.span.start < diagnostic.primary.span.end
    }));
}

#[test]
fn diagnostics_report_maybe_assigned_local_read_after_branch() {
    let mir = lower_mir("function y = f(c); if c; y = 1; end; x = y; end");
    let body = mir.bodies.values().next().unwrap();

    let diagnostics = diagnose_uninitialized_reads(body);

    assert!(diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-MIR0002"));
}

#[test]
fn summary_records_function_outputs_and_store_entry() {
    let mir = lower_mir("function y = f(x); y = x; end");
    let body = mir.bodies.values().next().unwrap();
    let mut store = AnalysisStore::default();

    let summary = summarize_body(body, &mut store);

    assert_eq!(summary.function, body.function);
    assert_eq!(summary.outputs.len(), 1);
    assert!(store.functions.contains_key(&body.function));
    assert_eq!(
        summary.effects.async_behavior,
        Some(AsyncBehaviorFact::NeverSuspends)
    );
}

#[test]
fn summary_marks_spawn_as_requiring_async_runtime() {
    let mir = lower_mir("function y = f(g); y = spawn(g); end");
    let body = mir.bodies.values().next().unwrap();
    let mut store = AnalysisStore::default();

    let summary = summarize_body(body, &mut store);

    assert_eq!(
        summary.effects.async_behavior,
        Some(AsyncBehaviorFact::RequiresAsyncRuntime)
    );
}

#[test]
fn spawn_boundary_analysis_records_spawn_site() {
    let mir = lower_mir("function y = f(g); y = spawn(g); end");
    let body = mir.bodies.values().next().unwrap();

    let boundaries = analyze_spawn_boundaries(body);

    assert_eq!(boundaries.len(), 1);
    assert!(matches!(
        boundaries[0].safety,
        runmat_hir::SpawnSafetyFact::RequiresIsolation
    ));
    assert!(matches!(boundaries[0].future, MirOperand::Local(_)));
}

#[test]
fn spawn_safety_summary_is_conservative_for_spawn_sites() {
    let mir = lower_mir("function y = f(g); y = spawn(g); end");
    let body = mir.bodies.values().next().unwrap();

    let summary = summarize_spawn_safety(body);

    assert_eq!(summary.function, body.function);
    assert!(matches!(
        summary.safety,
        runmat_hir::SpawnSafetyFact::RequiresIsolation
    ));
}

#[test]
fn analyze_assembly_populates_store_products_by_function() {
    let mir = lower_mir("async function y = f(g, x); t = spawn(g); await(t); y = x; end");

    let store = analyze_assembly(&mir);
    let body = mir.bodies.values().next().unwrap();

    assert!(store.functions.contains_key(&body.function));
    assert!(store.liveness.contains_key(&body.function));
    assert!(store.spawn_boundaries.contains_key(&body.function));
    assert_eq!(store.spawn_boundaries.get(&body.function).unwrap().len(), 1);
    assert_eq!(
        store
            .liveness
            .get(&body.function)
            .unwrap()
            .live_across_await
            .len(),
        1
    );
}

#[test]
fn analyze_assembly_scopes_mir_local_facts_by_function() {
    let mir = lower_mir("function y = f(x); y = x; end\nfunction z = g(a); z = a; end");

    let store = analyze_assembly(&mir);
    let expected_locals: usize = mir.bodies.values().map(|body| body.locals.len()).sum();

    assert_eq!(store.mir_locals.len(), expected_locals);
    for body in mir.bodies.values() {
        for local in &body.locals {
            assert!(store.mir_locals.contains_key(&MirLocalKey {
                function: body.function,
                local: local.id,
            }));
        }
    }
}

#[test]
fn analyze_assembly_collects_structured_diagnostics() {
    let mir = lower_mir("function y = f(c); if c; y = 1; end; z = y; end");

    let store = analyze_assembly(&mir);

    assert!(store
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-MIR0002"
            && diagnostic.category.as_deref() == Some("definite-assignment")));
}

#[test]
fn analysis_store_diagnostics_serialize_with_store() {
    let mir = lower_mir("function y = f(c); if c; y = 1; end; z = y; end");
    let store = analyze_assembly(&mir);

    let json = serde_json::to_string(&store).unwrap();

    assert!(json.contains("RM-MIR0002"));
    assert!(json.contains("definite-assignment"));
}

#[test]
fn await_statement_lowers_to_await_terminator_with_resume_block() {
    let mir = lower_mir("async function y = f(g); await(g); y = 1; end");
    let body = mir.bodies.values().next().unwrap();

    let MirTerminatorKind::Await { resume, .. } = body.blocks[0].terminator.kind else {
        panic!("expected await terminator");
    };
    assert_eq!(resume, body.blocks[1].id);
    assert_eq!(body.blocks[1].statements.len(), 1);
    assert!(matches!(
        body.blocks[1].terminator.kind,
        MirTerminatorKind::Return(_)
    ));
}

#[test]
fn await_terminator_marks_summary_as_requiring_async_runtime() {
    let mir = lower_mir("async function y = f(g); await(g); y = 1; end");
    let body = mir.bodies.values().next().unwrap();
    let mut store = AnalysisStore::default();

    let summary = summarize_body(body, &mut store);

    assert_eq!(
        summary.effects.async_behavior,
        Some(AsyncBehaviorFact::RequiresAsyncRuntime)
    );
}

#[test]
fn await_assignment_lowers_to_result_place() {
    let mir = lower_mir("async function y = f(g); y = await(g); end");
    let body = mir.bodies.values().next().unwrap();

    let MirTerminatorKind::Await { result, resume, .. } = &body.blocks[0].terminator.kind else {
        panic!("expected await terminator");
    };
    assert!(matches!(result, Some(MirPlace::Local(_))));
    assert_eq!(*resume, body.blocks[1].id);
}

#[test]
fn dataflow_marks_await_assignment_output_definitely_assigned() {
    let (body, store) = analyze_single_body("async function y = f(g); y = await(g); end");
    let output = first_local_of_kind(&body, MirLocalKind::Output);

    assert_eq!(
        store
            .mir_locals
            .get(&MirLocalKey {
                function: body.function,
                local: output,
            })
            .unwrap()
            .initialized,
        InitFact::DefinitelyAssigned
    );
}

#[test]
fn liveness_records_locals_live_across_await() {
    let mir = lower_mir("async function y = f(g, x); await(g); y = x; end");
    let body = mir.bodies.values().next().unwrap();

    let facts = analyze_liveness(body);

    assert_eq!(facts.live_across_await.len(), 1);
    let live = &facts.live_across_await[0].1;
    let param_locals: Vec<_> = body
        .locals
        .iter()
        .filter(|local| matches!(local.kind, MirLocalKind::Parameter))
        .map(|local| local.id)
        .collect();
    assert!(param_locals.iter().any(|local| live.contains(local)));
    assert!(live.len() >= 2);
}

#[test]
fn global_statement_lowers_to_workspace_effect() {
    let mir = lower_mir("function y = f(); global g; y = 1; end");
    let body = mir.bodies.values().next().unwrap();

    assert!(matches!(
        body.blocks[0].statements[0].kind,
        MirStmtKind::WorkspaceEffect {
            effect: runmat_hir::WorkspaceEffect::MutatesGlobal,
            ..
        }
    ));
}

#[test]
fn persistent_statement_lowers_to_summary_workspace_effect() {
    let mir = lower_mir("function y = f(); persistent p; y = 1; end");
    let body = mir.bodies.values().next().unwrap();
    let mut store = AnalysisStore::default();

    let summary = summarize_body(body, &mut store);

    assert!(summary
        .effects
        .workspace
        .contains(&runmat_hir::WorkspaceEffect::MutatesPersistent));
}

#[test]
fn direct_function_call_preserves_callee_and_requested_outputs() {
    let mir = lower_mir("function y = g(x); y = f(x); end\nfunction z = f(a); z = a; end");
    let stmt = mir
        .bodies
        .values()
        .flat_map(|body| &body.blocks[0].statements)
        .find(|stmt| {
            matches!(
                stmt.kind,
                MirStmtKind::Assign {
                    value: MirRvalue::Call(_),
                    ..
                }
            )
        })
        .unwrap();
    let MirStmtKind::Assign {
        value: MirRvalue::Call(call),
        ..
    } = &stmt.kind
    else {
        panic!("expected call assignment");
    };

    assert!(matches!(call.callee, HirCallableRef::Function(_)));
    assert!(matches!(
        call.requested_outputs,
        runmat_hir::RequestedOutputCount::One
    ));
}

#[test]
fn command_call_lowers_to_zero_output_call_with_string_args() {
    let mir = lower_mir("function y = f()\nformat long\ny = 1\nend");
    let body = mir.bodies.values().next().unwrap();

    let MirStmtKind::Expr(MirRvalue::Call(call)) = &body.blocks[0].statements[0].kind else {
        panic!("expected command call expression");
    };

    assert!(matches!(
        call.requested_outputs,
        runmat_hir::RequestedOutputCount::Zero
    ));
    assert_eq!(call.args.len(), 1);
    assert!(matches!(
        call.args[0],
        MirOperand::Constant(runmat_mir::MirConstant::String(ref value)) if value.0 == "long"
    ));
}

#[test]
fn tensor_literal_lowers_to_mir_aggregate() {
    let mir = lower_mir("function y = make_tensor(x); y = [x, x + 1]; end");
    let body = mir.bodies.values().next().unwrap();

    assert!(body
        .locals
        .iter()
        .any(|local| matches!(local.kind, MirLocalKind::Temporary)));
    assert!(matches!(
        body.blocks[0].statements[1].kind,
        MirStmtKind::Assign {
            value: MirRvalue::Aggregate {
                kind: MirAggregateKind::Tensor,
                ref elements,
            },
            ..
        } if elements.len() == 2
    ));
}

#[test]
fn cell_literal_lowers_to_mir_aggregate() {
    let mir = lower_mir("function y = make_cell(x); y = {x, x + 1}; end");
    let body = mir.bodies.values().next().unwrap();

    assert!(matches!(
        body.blocks[0].statements[1].kind,
        MirStmtKind::Assign {
            value: MirRvalue::Aggregate {
                kind: MirAggregateKind::Cell,
                ref elements,
            },
            ..
        } if elements.len() == 2
    ));
}

#[test]
fn complex_call_arguments_lower_through_temporary_locals() {
    let mir = lower_mir("function y = g(x); y = f(x + 1); end\nfunction z = f(a); z = a; end");
    let body = mir
        .bodies
        .values()
        .find(|body| body.blocks[0].statements.len() == 2)
        .unwrap();

    assert!(body
        .locals
        .iter()
        .any(|local| matches!(local.kind, MirLocalKind::Temporary) && local.binding.is_none()));
    assert!(matches!(
        body.blocks[0].statements[0].kind,
        MirStmtKind::Assign {
            place: MirPlace::Local(_),
            value: MirRvalue::Binary(_, _, _),
        }
    ));

    let MirStmtKind::Assign {
        value: MirRvalue::Call(call),
        ..
    } = &body.blocks[0].statements[1].kind
    else {
        panic!("expected call assignment after temp");
    };
    assert!(matches!(call.args[0], MirOperand::Local(_)));
}

#[test]
fn nested_binary_operands_lower_through_temporary_locals() {
    let mir = lower_mir("function y = calc(a, b, c); y = (a + b) * c; end");
    let body = mir.bodies.values().next().unwrap();

    assert_eq!(body.blocks[0].statements.len(), 2);
    assert!(matches!(
        body.blocks[0].statements[0].kind,
        MirStmtKind::Assign {
            place: MirPlace::Local(_),
            value: MirRvalue::Binary(_, _, _),
        }
    ));
    assert!(matches!(
        body.blocks[0].statements[1].kind,
        MirStmtKind::Assign {
            value: MirRvalue::Binary(MirOperand::Local(_), _, _),
            ..
        }
    ));
}

#[test]
fn multi_assign_preserves_discard_targets_and_requested_outputs() {
    let mir = lower_mir("function idx = pick(x); [~, idx] = max(x); end");
    let body = mir.bodies.values().next().unwrap();

    let MirStmtKind::MultiAssign {
        targets,
        value: MirRvalue::Call(call),
    } = &body.blocks[0].statements[0].kind
    else {
        panic!("expected multi-output call assignment");
    };

    assert_eq!(targets.targets.len(), 2);
    assert!(matches!(targets.targets[0], MirOutputTarget::Discard));
    assert!(matches!(targets.targets[1], MirOutputTarget::Place(_)));
    assert!(matches!(
        call.requested_outputs,
        runmat_hir::RequestedOutputCount::Exactly(2)
    ));
}

#[test]
fn indexed_assignment_lowers_to_index_place() {
    let mir = lower_mir("function y = write_index(x); x(1) = 2; y = x; end");
    let body = mir.bodies.values().next().unwrap();

    assert!(matches!(
        body.blocks[0].statements[0].kind,
        MirStmtKind::PlaceMutation(ref mutation)
            if mutation.kind == runmat_hir::PlaceMutationKind::IndexedAssign
                && mutation.creation_policy == runmat_hir::AssignmentCreationPolicy::CreateArrayByIndex
    ));
    assert!(matches!(
        body.blocks[0].statements[1].kind,
        MirStmtKind::Assign {
            place: MirPlace::Index(_, _),
            ..
        }
    ));
}

#[test]
fn indexed_empty_assignment_lowers_to_delete_mutation() {
    let mir = lower_mir("function y = delete_index(x); x(1) = []; y = x; end");
    let body = mir.bodies.values().next().unwrap();

    assert!(matches!(
        body.blocks[0].statements[0].kind,
        MirStmtKind::PlaceMutation(ref mutation)
            if mutation.kind == runmat_hir::PlaceMutationKind::Delete
                && mutation.creation_policy == runmat_hir::AssignmentCreationPolicy::ExistingOnly
    ));
}

#[test]
fn index_components_lower_to_mir_operands() {
    let mir = lower_mir("function y = read_index(a, i); y = a(i + 1); end");
    let body = mir.bodies.values().next().unwrap();

    assert!(body
        .locals
        .iter()
        .any(|local| matches!(local.kind, MirLocalKind::Temporary)));
    assert!(matches!(
        body.blocks[0].statements[1].kind,
        MirStmtKind::Assign {
            value: MirRvalue::Index { .. },
            ..
        }
    ));
}

#[test]
fn diagnostics_report_unassigned_index_operand_read() {
    let mir = lower_mir("function y = read_index(a); y = a(y); end");
    let body = mir.bodies.values().next().unwrap();

    let diagnostics = diagnose_uninitialized_reads(body);

    assert!(diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-MIR0001"));
}

#[test]
fn member_assignment_lowers_to_member_place() {
    let mir = lower_mir("function s = write_member(s); s.value = 2; end");
    let body = mir.bodies.values().next().unwrap();

    assert!(matches!(
        body.blocks[0].statements[0].kind,
        MirStmtKind::PlaceMutation(ref mutation)
            if mutation.kind == runmat_hir::PlaceMutationKind::MemberAssign
                && mutation.creation_policy == runmat_hir::AssignmentCreationPolicy::CreateStructFieldPath
    ));
    assert!(matches!(
        body.blocks[0].statements[1].kind,
        MirStmtKind::Assign {
            place: MirPlace::Member(_, _),
            ..
        }
    ));
}

#[test]
fn if_statement_lowers_to_branch_blocks_and_merge() {
    let mir = lower_mir("function y = choose(c, x); if c; y = x; else; y = 0; end; end");
    let body = mir.bodies.values().next().unwrap();

    assert_eq!(body.blocks.len(), 4);
    assert!(matches!(
        body.blocks[0].terminator.kind,
        MirTerminatorKind::Branch { .. }
    ));
    assert!(matches!(
        body.blocks[1].terminator.kind,
        MirTerminatorKind::Goto(_)
    ));
    assert!(matches!(
        body.blocks[2].terminator.kind,
        MirTerminatorKind::Goto(_)
    ));
    assert!(matches!(
        body.blocks[3].terminator.kind,
        MirTerminatorKind::Return(_)
    ));
    assert_eq!(body.source_map.statements.len(), 3);
}

#[test]
fn if_statement_flows_to_following_statements() {
    let mir = lower_mir("function y = choose(c); if c; y = 1; else; y = 2; end; y = y + 1; end");
    let body = mir.bodies.values().next().unwrap();

    assert_eq!(body.blocks.len(), 4);
    assert_eq!(body.blocks[3].statements.len(), 1);
    assert!(matches!(
        body.blocks[3].statements[0].kind,
        MirStmtKind::Assign { .. }
    ));
    assert!(matches!(
        body.blocks[3].terminator.kind,
        MirTerminatorKind::Return(_)
    ));
}

#[test]
fn elseif_statement_lowers_to_nested_branch_cfg() {
    let mir = lower_mir(
        "function y = choose(c, d); if c; y = 1; elseif d; y = 2; else; y = 3; end; y = y + 1; end",
    );
    let body = mir.bodies.values().next().unwrap();

    assert!(matches!(
        body.blocks[0].terminator.kind,
        MirTerminatorKind::Branch { .. }
    ));
    assert!(
        body.blocks
            .iter()
            .filter(|block| matches!(block.terminator.kind, MirTerminatorKind::Branch { .. }))
            .count()
            >= 2
    );
    assert!(body.blocks.iter().any(|block| block.statements.len() == 1
        && matches!(block.terminator.kind, MirTerminatorKind::Return(_))));
}

#[test]
fn diagnostics_report_maybe_assigned_local_read_after_elseif_without_else() {
    let mir = lower_mir("function y = choose(c, d); if c; y = 1; elseif d; y = 2; end; z = y; end");
    let body = mir.bodies.values().next().unwrap();

    let diagnostics = diagnose_uninitialized_reads(body);

    assert!(diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-MIR0002"));
}

#[test]
fn switch_statement_lowers_to_switch_blocks_and_continuation() {
    let mir = lower_mir(
        "function y = choose(x); switch x; case 1; y = 1; otherwise; y = 2; end; y = y + 1; end",
    );
    let body = mir.bodies.values().next().unwrap();

    let MirTerminatorKind::Switch {
        cases, otherwise, ..
    } = &body.blocks[0].terminator.kind
    else {
        panic!("expected switch terminator");
    };
    assert_eq!(cases.len(), 1);
    assert!(body.blocks.iter().any(|block| block.id == *otherwise));
    assert_eq!(body.blocks[1].statements.len(), 1);
    assert!(matches!(
        body.blocks[1].terminator.kind,
        MirTerminatorKind::Return(_)
    ));
}

#[test]
fn diagnostics_report_maybe_assigned_local_read_after_switch() {
    let mir =
        lower_mir("function y = choose(x); switch x; case 1; y = 1; otherwise; end; z = y; end");
    let body = mir.bodies.values().next().unwrap();

    let diagnostics = diagnose_uninitialized_reads(body);

    assert!(diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-MIR0002"));
}

#[test]
fn while_loop_lowers_to_branch_body_backedge_and_exit() {
    let mir = lower_mir("function y = spin(c, x); while c; y = x; end; end");
    let body = mir.bodies.values().next().unwrap();

    assert_eq!(body.blocks.len(), 3);
    assert!(matches!(
        body.blocks[0].terminator.kind,
        MirTerminatorKind::Branch { .. }
    ));
    assert!(matches!(
        body.blocks[1].terminator.kind,
        MirTerminatorKind::Goto(target) if target == body.blocks[0].id
    ));
    assert!(matches!(
        body.blocks[2].terminator.kind,
        MirTerminatorKind::Return(_)
    ));
    assert_eq!(body.source_map.statements.len(), 2);
}

#[test]
fn while_loop_exit_flows_to_following_statements() {
    let mir = lower_mir("function y = after_loop(c); while c; y = 1; end; y = y + 1; end");
    let body = mir.bodies.values().next().unwrap();

    let MirTerminatorKind::Branch { else_block, .. } = body.blocks[0].terminator.kind else {
        panic!("expected while branch");
    };
    assert_eq!(else_block, body.blocks[2].id);
    assert_eq!(body.blocks[2].statements.len(), 1);
    assert!(matches!(
        body.blocks[2].terminator.kind,
        MirTerminatorKind::Return(_)
    ));
}

#[test]
fn for_loop_lowers_to_iteration_terminator_body_backedge_and_exit() {
    let mir = lower_mir("function y = sum_to(n); y = 0; for i = 1:n; y = y + i; end; end");
    let body = mir.bodies.values().next().unwrap();

    assert_eq!(body.blocks.len(), 3);
    assert!(matches!(
        body.blocks[0].terminator.kind,
        MirTerminatorKind::For { .. }
    ));
    assert!(matches!(
        body.blocks[1].terminator.kind,
        MirTerminatorKind::Goto(target) if target == body.blocks[0].id
    ));
    assert!(matches!(
        body.blocks[2].terminator.kind,
        MirTerminatorKind::Return(_)
    ));
    assert_eq!(body.blocks[0].statements.len(), 1);
    assert_eq!(body.source_map.statements.len(), 3);
}

#[test]
fn try_catch_lowers_to_try_catch_blocks_and_merge() {
    let mir = lower_mir("function y = guarded(x); try; y = x; catch err; y = 0; end; end");
    let body = mir.bodies.values().next().unwrap();

    assert_eq!(body.blocks.len(), 4);
    assert!(matches!(
        body.blocks[0].terminator.kind,
        MirTerminatorKind::TryCatch { .. }
    ));
    assert!(matches!(
        body.blocks[1].terminator.kind,
        MirTerminatorKind::Goto(_)
    ));
    assert!(matches!(
        body.blocks[2].terminator.kind,
        MirTerminatorKind::Goto(_)
    ));
    assert!(matches!(
        body.blocks[3].terminator.kind,
        MirTerminatorKind::Return(_)
    ));
    assert_eq!(body.source_map.statements.len(), 3);
}

#[test]
fn try_catch_flows_to_following_statements() {
    let mir =
        lower_mir("function y = guarded(x); try; y = x; catch err; y = 0; end; y = y + 1; end");
    let body = mir.bodies.values().next().unwrap();

    assert_eq!(body.blocks.len(), 4);
    assert_eq!(body.blocks[3].statements.len(), 1);
    assert!(matches!(
        body.blocks[3].terminator.kind,
        MirTerminatorKind::Return(_)
    ));
}

#[test]
fn break_in_loop_lowers_to_exit_edge() {
    let mir = lower_mir("function y = first(c); while c; break; end; y = 1; end");
    let body = mir.bodies.values().next().unwrap();

    let MirTerminatorKind::Branch { else_block, .. } = body.blocks[0].terminator.kind else {
        panic!("expected while branch");
    };
    assert!(matches!(
        body.blocks[1].terminator.kind,
        MirTerminatorKind::Goto(target) if target == else_block
    ));
}

#[test]
fn continue_in_loop_lowers_to_loop_condition_edge() {
    let mir = lower_mir("function y = again(c); while c; continue; end; y = 1; end");
    let body = mir.bodies.values().next().unwrap();

    assert!(matches!(
        body.blocks[1].terminator.kind,
        MirTerminatorKind::Goto(target) if target == body.blocks[0].id
    ));
}

#[test]
fn return_in_nested_block_lowers_to_return_terminator() {
    let mir = lower_mir("function y = done(c); if c; return; else; y = 1; end; end");
    let body = mir.bodies.values().next().unwrap();

    assert!(matches!(
        body.blocks[1].terminator.kind,
        MirTerminatorKind::Return(_)
    ));
    assert!(matches!(
        body.blocks[2].terminator.kind,
        MirTerminatorKind::Goto(_)
    ));
}
