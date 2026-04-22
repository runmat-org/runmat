use runmat_accelerate::InstrSpan;
use runmat_vm::accel::fusion as accel_fusion;
use runmat_vm::Instr;

#[test]
fn store_reload_span_with_one_live_result_is_legal() {
    let instructions = vec![
        Instr::LoadVar(0),
        Instr::LoadConst(0.0),
        Instr::Add,
        Instr::StoreVar(1),
        Instr::LoadVar(1),
    ];
    let span = InstrSpan { start: 0, end: 4 };

    assert_eq!(
        accel_fusion::fusion_span_live_result_count(&instructions, &span),
        Some(1)
    );
    assert!(!accel_fusion::fusion_span_has_vm_barrier(
        &instructions,
        &span
    ));
}

#[test]
fn span_leaving_multiple_live_results_is_illegal() {
    let instructions = vec![
        Instr::LoadVar(0),
        Instr::LoadConst(0.0),
        Instr::Add,
        Instr::LoadVar(1),
    ];
    let span = InstrSpan { start: 0, end: 3 };

    assert_eq!(
        accel_fusion::fusion_span_live_result_count(&instructions, &span),
        Some(2)
    );
    assert!(accel_fusion::fusion_span_has_vm_barrier(
        &instructions,
        &span
    ));
}

#[test]
fn stored_value_observed_after_span_is_legal_when_materialized() {
    let instructions = vec![
        Instr::LoadVar(0),
        Instr::LoadConst(0.0),
        Instr::Add,
        Instr::StoreVar(1),
        Instr::LoadVar(1),
        Instr::LoadVar(1),
    ];
    let span = InstrSpan { start: 0, end: 4 };

    assert!(!accel_fusion::fusion_span_has_vm_barrier(
        &instructions,
        &span
    ));
}

#[test]
fn overwritten_store_before_later_load_is_legal() {
    let instructions = vec![
        Instr::LoadVar(0),
        Instr::LoadConst(0.0),
        Instr::Add,
        Instr::StoreVar(1),
        Instr::LoadVar(1),
        Instr::LoadConst(1.0),
        Instr::StoreVar(1),
        Instr::LoadVar(1),
    ];
    let span = InstrSpan { start: 0, end: 4 };

    assert!(!accel_fusion::fusion_span_has_vm_barrier(
        &instructions,
        &span
    ));
}
