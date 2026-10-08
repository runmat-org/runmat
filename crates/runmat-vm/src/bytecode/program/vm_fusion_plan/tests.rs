use super::*;
use crate::bytecode::{FusionInstructionKind, FusionInstructionWindow};
use runmat_accelerate::graph::InstrSpan;
use runmat_builtins::Type;

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
