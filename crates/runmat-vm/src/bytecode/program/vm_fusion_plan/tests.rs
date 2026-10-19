use super::*;
use crate::bytecode::{FusionInstructionKind, FusionInstructionWindow};
use runmat_accelerate::graph::InstrSpan;
use runmat_builtins::Type;

#[path = "tests/provenance.rs"]
mod provenance;

fn planned(instructions: Vec<Instr>, span: InstrSpan) -> FusionPlanRef {
    let mut bytecode = Bytecode::empty();
    bytecode.var_types = vec![Type::Num; 5];
    bytecode.instructions = instructions;
    bytecode.fusion_metadata.mir_fusion_candidate_group_count = 1;
    bytecode.fusion_metadata.instruction_windows = vec![FusionInstructionWindow {
        span,
        kind: FusionInstructionKind::Elementwise,
    }];
    bytecode
        .prepare_vm_fusion_plan()
        .0
        .expect("runtime graph-derived plan")
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn semantic_region_identity_covers_executable_stream_and_catch_authority() {
    let mut bytecode = Bytecode::empty();
    bytecode.var_types = vec![Type::Num; 3];
    bytecode.instructions = vec![
        Instr::LoadVar(0),
        Instr::LoadVar(1),
        Instr::Add,
        Instr::StoreVar(2),
    ];
    bytecode.catch_scopes = vec![None; bytecode.instructions.len()];
    bytecode.fusion_metadata.mir_fusion_candidate_group_count = 1;
    bytecode.fusion_metadata.instruction_windows = vec![FusionInstructionWindow {
        span: InstrSpan { start: 2, end: 2 },
        kind: FusionInstructionKind::Elementwise,
    }];
    let plan = bytecode
        .prepare_vm_fusion_plan()
        .0
        .expect("runtime graph-derived plan");
    let executable = vm_fusion_executable_identity(&bytecode).unwrap();
    let baseline = vm_fusion_semantic_identity(executable, &plan.groups[0]).unwrap();
    assert!(baseline == vm_fusion_semantic_identity(executable, &plan.groups[0]).unwrap());

    let mut changed_instruction = bytecode.clone();
    changed_instruction.instructions[2] = Instr::Sub;
    let changed = vm_fusion_semantic_identity(
        vm_fusion_executable_identity(&changed_instruction).unwrap(),
        &plan.groups[0],
    )
    .unwrap();
    assert!(baseline != changed);

    let mut negative_zero = bytecode.clone();
    negative_zero.instructions[0] = Instr::LoadConst(-0.0);
    let negative_zero = vm_fusion_executable_identity(&negative_zero).unwrap();
    let mut positive_zero = bytecode.clone();
    positive_zero.instructions[0] = Instr::LoadConst(0.0);
    let positive_zero = vm_fusion_executable_identity(&positive_zero).unwrap();
    assert!(negative_zero != positive_zero);

    let mut nan_a = bytecode.clone();
    nan_a.instructions[0] = Instr::LoadConst(f64::from_bits(0x7ff8_0000_0000_0042));
    let nan_a = vm_fusion_executable_identity(&nan_a).unwrap();
    let mut nan_b = bytecode.clone();
    nan_b.instructions[0] = Instr::LoadConst(f64::from_bits(0x7ff8_0000_0000_0043));
    let nan_b = vm_fusion_executable_identity(&nan_b).unwrap();
    assert!(nan_a != nan_b);

    let mut integer = bytecode.clone();
    integer.instructions[0] = Instr::LoadInt(runmat_value::IntValue::U64(1_u64 << 63));
    assert!(vm_fusion_executable_identity(&integer).unwrap() != positive_zero);

    let mut changed_catch = bytecode;
    changed_catch.catch_scopes[2] = Some(runmat_mir::CatchScopeId(7));
    let changed = vm_fusion_semantic_identity(
        vm_fusion_executable_identity(&changed_catch).unwrap(),
        &plan.groups[0],
    )
    .unwrap();
    assert!(baseline != changed);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn executable_identity_accepts_nonempty_typed_resume_points_in_canonical_order() {
    let first = runmat_types::ProgramPointId {
        function: runmat_types::ProgramFunctionId(0),
        block: 1,
        position: 2,
    };
    let second = runmat_types::ProgramPointId {
        function: runmat_types::ProgramFunctionId(0),
        block: 2,
        position: 0,
    };
    let mut bytecode = Bytecode::empty();
    bytecode.instructions = vec![Instr::LoadConst(1.0)];
    bytecode.resume_points.insert(second, 7);
    bytecode.resume_points.insert(first, 3);
    let baseline = vm_fusion_executable_identity(&bytecode)
        .expect("typed nonempty resume points must have a canonical JSON sequence identity");

    let mut reversed_insertion = bytecode.clone();
    reversed_insertion.resume_points.clear();
    reversed_insertion.resume_points.insert(first, 3);
    reversed_insertion.resume_points.insert(second, 7);
    assert_eq!(
        vm_fusion_executable_identity(&reversed_insertion),
        Some(baseline)
    );

    let mut changed_pc = bytecode.clone();
    changed_pc.resume_points.insert(second, 8);
    assert_ne!(vm_fusion_executable_identity(&changed_pc), Some(baseline));
    let mut removed = bytecode;
    removed.resume_points.remove(&first);
    assert_ne!(vm_fusion_executable_identity(&removed), Some(baseline));
}

#[test]
fn future_store_is_not_published_by_operator_only_window() {
    let plan = planned(
        vec![
            Instr::LoadVar(0),
            Instr::LoadVar(1),
            Instr::Add,
            Instr::StoreVar(2),
            Instr::LoadVar(2),
            Instr::StoreLocal(3),
        ],
        InstrSpan { start: 2, end: 2 },
    );
    let group = &plan.groups[0];
    assert!(group.kernel.supported);
    assert!(group.materialized_stores.is_empty());
}

#[test]
fn reversed_span_rejects_before_transcript_selection() {
    let instructions = vec![Instr::LoadVar(0), Instr::LoadVar(1), Instr::Add];
    let mut plan = (*planned(instructions.clone(), InstrSpan { start: 2, end: 2 })).clone();
    plan.groups[0].group.span = InstrSpan { start: 2, end: 1 };
    reconcile_stores(&mut plan, &instructions, &[]);
    assert!(!plan.groups[0].kernel.supported);
    assert!(plan.groups[0].materialized_stores.is_empty());
}

#[test]
fn in_span_repeated_alias_stores_keep_exact_pc_order() {
    let plan = planned(
        vec![
            Instr::LoadVar(0),
            Instr::LoadVar(1),
            Instr::Add,
            Instr::StoreVar(2),
            Instr::LoadVar(2),
            Instr::StoreLocal(3),
            Instr::LoadVar(2),
            Instr::LoadVar(4),
            Instr::Mul,
            Instr::StoreVar(2),
        ],
        InstrSpan { start: 2, end: 8 },
    );
    let group = &plan.groups[0];
    assert!(group.kernel.supported);
    assert_eq!(group.materialized_stores.len(), 2);
    assert_eq!(group.materialized_stores[0].binding.kind, VarKind::Global);
    assert_eq!(group.materialized_stores[0].binding.index, 2);
    assert_eq!(group.materialized_stores[1].binding.kind, VarKind::Local);
    assert_eq!(group.materialized_stores[1].binding.index, 3);
    assert_eq!(
        group.materialized_stores[0].value_id, group.materialized_stores[1].value_id,
        "the transcript must preserve two publications of the same value"
    );
}

#[test]
fn missing_store_transcript_rejects_skipped_store() {
    let instructions = vec![
        Instr::LoadVar(0),
        Instr::LoadVar(1),
        Instr::Add,
        Instr::StoreVar(2),
        Instr::LoadVar(2),
        Instr::LoadVar(4),
        Instr::Mul,
    ];
    let mut plan = (*planned(instructions.clone(), InstrSpan { start: 2, end: 6 })).clone();
    assert!(plan.groups[0].kernel.supported);
    reconcile_stores(&mut plan, &instructions, &[]);
    assert!(!plan.groups[0].kernel.supported);
    assert!(plan.groups[0].materialized_stores.is_empty());
}

#[test]
fn input_value_store_inside_window_is_rejected_before_dispatch() {
    let plan = planned(
        vec![
            Instr::LoadVar(0),
            Instr::StoreVar(2),
            Instr::LoadVar(2),
            Instr::LoadVar(1),
            Instr::Add,
        ],
        InstrSpan { start: 1, end: 4 },
    );
    let group = &plan.groups[0];
    assert!(!group.kernel.supported);
    assert!(group.materialized_stores.is_empty());
}

#[test]
fn many_windows_select_only_their_ordered_store_ordinals() {
    let stores: Vec<_> = (0..128)
        .map(|ordinal| AccelStore {
            pc: ordinal * 3 + 2,
            value_id: (ordinal / 2) as u32,
            binding: VarBinding {
                kind: if ordinal % 2 == 0 {
                    VarKind::Global
                } else {
                    VarKind::Local
                },
                index: ordinal,
            },
        })
        .collect();
    for first_ordinal in 0..128 {
        let end_ordinal = (first_ordinal + 3).min(127);
        let selected = stores_in_span(&stores, first_ordinal * 3 + 1, end_ordinal * 3 + 2);
        let ordinals: Vec<_> = selected.iter().map(|store| store.binding.index).collect();
        assert_eq!(ordinals, (first_ordinal..=end_ordinal).collect::<Vec<_>>());
        assert_eq!(selected[0].pc, first_ordinal * 3 + 2);
        assert_eq!(selected.last().unwrap().pc, end_ordinal * 3 + 2);
    }
    assert!(stores_in_span(&stores, 400, 500).is_empty());
    assert!(stores_in_span(&stores, 0, 1).is_empty());
}
