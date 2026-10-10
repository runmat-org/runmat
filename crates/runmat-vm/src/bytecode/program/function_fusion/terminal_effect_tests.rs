use super::*;

fn reduction_body() -> FunctionBytecode {
    let ast = runmat_parser::parse("function y = f(x)\ny = min(x + x);\nend\n").unwrap();
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).unwrap();
    let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).unwrap();
    crate::bytecode::compile_semantic_function_registry(&hir.assembly, &mir)
        .unwrap()
        .remove(&hir.assembly.functions[0].id)
        .unwrap()
}

#[test]
fn function_scaffold_allows_only_declared_terminal_reduction_throw_suspend() {
    let body = reduction_body();
    let projection = body.fusion_stage_projection.as_ref().unwrap();
    let local = projection
        .private_temporary_slots
        .first()
        .expect("private producer")
        .local;
    let producer = projection
        .statements
        .iter()
        .find(|row| row.contract.defs.iter().any(|value| value.local == local))
        .expect("selected producer")
        .contract
        .point;
    let terminal = projection
        .statements
        .iter()
        .find(|row| row.contract.uses.iter().any(|value| value.local == local))
        .expect("selected terminal reduction")
        .contract
        .point;
    let terminal_row = projection.statement_at(terminal).unwrap();
    assert!(terminal_row
        .contract
        .effects
        .0
        .contains(&EffectKind::MayThrow));
    assert!(terminal_row
        .contract
        .effects
        .0
        .contains(&EffectKind::MaySuspend));
    body.validate_fusion_scaffold()
        .expect("canonical terminal effects are admissible");

    let mut fieldless_legacy = body.clone();
    fieldless_legacy
        .fusion_scaffold
        .as_mut()
        .unwrap()
        .private_layout = None;
    assert!(
        fieldless_legacy.validate_fusion_scaffold().is_err(),
        "old fieldless hints cannot gain a new effect admission"
    );

    for (point, effect) in [
        (producer, EffectKind::MayThrow),
        (terminal, EffectKind::WorkspaceWrite),
        (terminal, EffectKind::Unknown),
    ] {
        let mut changed = body.clone();
        std::sync::Arc::make_mut(changed.fusion_stage_projection.as_mut().unwrap())
            .statements
            .iter_mut()
            .find(|row| row.contract.point == point)
            .unwrap()
            .contract
            .effects
            .0
            .insert(effect);
        changed.reseal_fusion_stage_after_owned_remap();
        assert!(changed.validate_fusion_stage_projection().is_ok());
        assert!(changed.validate_fusion_scaffold().is_err(), "effect role");
    }
}
