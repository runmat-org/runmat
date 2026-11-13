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

#[test]
fn input_transport_requires_an_exact_evaluated_source_not_a_graph_binding_guess() {
    let instructions = vec![
        Instr::CallBuiltinMulti("opaqueProducer".into(), 0, 1),
        Instr::LoadConst(1.0),
        Instr::Add,
    ];
    let plan = planned(instructions.clone(), InstrSpan { start: 2, end: 2 });
    let (graph, stores, loads) =
        build_accel_graph_with_transcript(&instructions, &[Type::Unknown; 0]);
    let selected = &plan.groups[0];
    assert!(selected.kernel.supported);
    let proof = VmFusionInputTransport::project(selected, &graph, &loads, &stores)
        .expect("ordinary producer's evaluated stack output has an exact identity");
    let vars = crate::interpreter::vars::InterpreterVars::from_owned(Vec::new());
    let context = crate::bytecode::program::ExecutionContext::default();
    let inputs = proof
        .gather(
            &[
                runmat_value::Value::Num(41.0),
                runmat_value::Value::Num(1.0),
            ],
            &vars,
            &context,
        )
        .expect("no producer replay or variable lookup is necessary");
    assert!(inputs.contains(&runmat_value::Value::Num(41.0)));

    let mut missing_source = selected.clone();
    missing_source
        .group
        .stack_layout
        .as_mut()
        .unwrap()
        .bindings
        .clear();
    assert!(VmFusionInputTransport::project(&missing_source, &graph, &loads, &stores).is_none());

    let mut wrong_input = selected.clone();
    wrong_input.inputs.push(u32::MAX);
    assert!(!proof.matches(&wrong_input));
    assert!(VmFusionInputTransport::project(&wrong_input, &graph, &loads, &stores).is_none());
}

#[test]
fn input_transport_folded_array_does_not_freeze_a_live_child_binding() {
    for live_child in [false, true] {
        let plan = planned(
            vec![
                Instr::LoadConst(5.0),
                Instr::StoreVar(0),
                Instr::LoadConst(1.0),
                Instr::LoadConst(2.0),
                Instr::Add,
                if live_child {
                    Instr::LoadVar(0)
                } else {
                    Instr::LoadConst(5.0)
                },
                Instr::LoadConst(3.0),
                Instr::CreateMatrix(1, 2),
                Instr::Add,
            ],
            InstrSpan { start: 4, end: 8 },
        );
        assert_eq!(
            plan.groups[0].kernel.supported, !live_child,
            "only a true source literal array can replace an in-region constructor"
        );
    }
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
