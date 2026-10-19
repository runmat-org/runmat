use super::*;

fn folded_body() -> FunctionBytecode {
    let ast = runmat_parser::parse("function y = f(x)\nb = x;\ny = min(b + b);\nend\n").unwrap();
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).unwrap();
    let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).unwrap();
    crate::bytecode::compile_semantic_function_registry(&hir.assembly, &mir)
        .unwrap()
        .remove(&hir.assembly.functions[0].id)
        .unwrap()
}

#[test]
fn function_private_layout_rejoins_selected_roles_and_all_named_bindings() {
    let body = folded_body();
    let pair = *body
        .fusion_stage_projection
        .as_ref()
        .unwrap()
        .private_temporary_slots
        .first()
        .expect("actual compiler private temporary");
    body.validate_fusion_scaffold()
        .expect("compiler-selected layout");

    let mut aliased = body.clone();
    let layout = aliased
        .fusion_scaffold
        .as_mut()
        .unwrap()
        .private_layout
        .as_mut()
        .unwrap();
    layout.bound_slots.push(pair.slot);
    layout.bound_slots.sort_unstable();
    layout.bound_slots.dedup();
    assert!(aliased.validate_fusion_scaffold().is_err(), "binding alias");

    let mut omitted = body.clone();
    let local_binding = *omitted
        .var_names
        .iter()
        .find_map(|(slot, name)| (name == "b").then_some(slot))
        .expect("named non-ABI local");
    omitted
        .fusion_scaffold
        .as_mut()
        .unwrap()
        .private_layout
        .as_mut()
        .unwrap()
        .bound_slots
        .retain(|slot| *slot != local_binding);
    assert!(
        omitted.validate_fusion_scaffold().is_err(),
        "omitted binding"
    );

    let mut changed_slot = body.clone();
    let layout = changed_slot
        .fusion_scaffold
        .as_mut()
        .unwrap()
        .private_layout
        .as_mut()
        .unwrap();
    layout
        .local_slots
        .iter_mut()
        .find(|row| row.local == pair.local)
        .unwrap()
        .slot += 1;
    assert!(
        changed_slot.validate_fusion_scaffold().is_err(),
        "slot join"
    );

    let mut wrong_owner = body.clone();
    wrong_owner
        .fusion_scaffold
        .as_mut()
        .unwrap()
        .private_layout
        .as_mut()
        .unwrap()
        .function = runmat_hir::FunctionId(body.function.0 + 1);
    assert!(wrong_owner.validate_fusion_scaffold().is_err(), "owner");
}
