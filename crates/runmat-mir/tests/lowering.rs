use runmat_hir::{lower, HirCallableRef, LoweringContext};
use runmat_mir::{
    lowering::lower_assembly, MirOutputTarget, MirPlace, MirRvalue, MirStmtKind, MirTerminatorKind,
};

fn lower_mir(src: &str) -> runmat_mir::MirAssembly {
    let ast = runmat_parser::parse(src).unwrap();
    let hir = lower(&ast, &LoweringContext::empty()).unwrap();
    lower_assembly(&hir.assembly).unwrap()
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
        MirStmtKind::Assign {
            place: MirPlace::Index(_, _),
            ..
        }
    ));
}

#[test]
fn member_assignment_lowers_to_member_place() {
    let mir = lower_mir("function s = write_member(s); s.value = 2; end");
    let body = mir.bodies.values().next().unwrap();

    assert!(matches!(
        body.blocks[0].statements[0].kind,
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
