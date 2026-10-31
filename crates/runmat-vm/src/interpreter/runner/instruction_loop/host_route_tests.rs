use super::*;

#[test]
fn terminal_builtin_classification_preserves_output_slot_call_semantics() {
    for (name, expected) in [
        ("sin", VmFusionHostInstruction::ElementwiseCall),
        ("sum", VmFusionHostInstruction::Reduction),
    ] {
        for instruction in [
            Instr::CallBuiltinMulti(name.into(), 1, 1),
            Instr::CallBuiltinMultiUsingOutputSlot(name.into(), 1, 7),
        ] {
            assert_eq!(
                host_instruction_kind(&instruction, 0, &[], &[], true, Some(1), &[]),
                expected,
                "{instruction:?}"
            );
        }
        let dynamic = Instr::CallBuiltinMultiUsingOutputSlot(name.into(), 1, 7);
        for count in [None, Some(0), Some(2)] {
            assert_eq!(
                host_instruction_kind(&dynamic, 0, &[], &[], true, count, &[]),
                VmFusionHostInstruction::Unproved,
            );
        }
    }
}

fn compiled_axis_control() -> crate::bytecode::Bytecode {
    let ast = runmat_parser::parse(
        "x = gpuArray(single([1, 2; 3, 4])); y = min(x + x, [], 2); z = gather(y);",
    )
    .expect("parse authentic empty-axis reduction");
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).expect("HIR");
    let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).expect("MIR");
    crate::bytecode::compile(&hir.assembly, &mir, hir.assembly.entrypoints[0].id)
        .expect("authentic empty-axis bytecode")
}

#[test]
fn host_structural_control_requires_exact_sealed_empty_literal_and_store() {
    let bytecode = compiled_axis_control();
    let bridges = crate::bytecode::fusion_private_temp::bridges(&bytecode);
    let controls = crate::bytecode::fusion_private_temp::empty_controls(&bytecode);
    let [control] = controls.as_slice() else {
        panic!(
            "one compiler-derived empty control: {:?}",
            bytecode.instructions
        );
    };
    assert_eq!(
        host_instruction_kind(
            &bytecode.instructions[control.producer_pc],
            control.producer_pc,
            &bridges,
            &controls,
            false,
            None,
            &[],
        ),
        VmFusionHostInstruction::Structural,
    );
    assert_eq!(
        host_instruction_kind(
            &bytecode.instructions[control.store_pc],
            control.store_pc,
            &bridges,
            &controls,
            false,
            None,
            &[],
        ),
        VmFusionHostInstruction::Structural,
    );
    assert_eq!(
        host_instruction_kind(
            &bytecode.instructions[control.producer_pc],
            control.producer_pc,
            &bridges,
            &[],
            false,
            None,
            &[],
        ),
        VmFusionHostInstruction::Unproved,
    );
    assert_eq!(
        host_instruction_kind(
            &bytecode.instructions[control.store_pc],
            control.store_pc,
            &bridges,
            &[],
            false,
            None,
            &[],
        ),
        VmFusionHostInstruction::Unproved,
    );

    let mut changed = bytecode;
    changed.instructions[control.producer_pc] = Instr::CreateMatrix(1, 1);
    changed.reseal_fusion_stage_after_owned_remap();
    let changed_controls = crate::bytecode::fusion_private_temp::empty_controls(&changed);
    assert!(
        changed_controls.is_empty(),
        "nonempty literal has no [] proof"
    );
    assert_eq!(
        host_instruction_kind(
            &changed.instructions[control.producer_pc],
            control.producer_pc,
            &crate::bytecode::fusion_private_temp::bridges(&changed),
            &changed_controls,
            false,
            None,
            &[],
        ),
        VmFusionHostInstruction::Unproved,
    );
}

#[test]
fn matrix_division_host_classification_uses_the_actual_scalar_divisor() {
    let matrix = Value::Tensor(runmat_value::Tensor::new(vec![1.0; 4], vec![2, 2]).unwrap());
    for scalar in [
        Value::Num(2.0),
        Value::Tensor(runmat_value::Tensor::new(vec![2.0], vec![1, 1]).unwrap()),
        Value::Tensor(runmat_value::Tensor::from_f32(vec![2.0], vec![1, 1]).unwrap()),
    ] {
        for (instruction, operands, expected) in [
            (
                Instr::RightDiv,
                vec![matrix.clone(), scalar.clone()],
                VmFusionHostInstruction::ElementwiseBinary,
            ),
            (
                Instr::LeftDiv,
                vec![scalar.clone(), matrix.clone()],
                VmFusionHostInstruction::ElementwiseBinary,
            ),
            (
                Instr::RightDiv,
                vec![scalar.clone(), matrix.clone()],
                VmFusionHostInstruction::Unproved,
            ),
            (
                Instr::LeftDiv,
                vec![matrix.clone(), scalar.clone()],
                VmFusionHostInstruction::Unproved,
            ),
        ] {
            assert_eq!(
                host_instruction_kind(&instruction, 0, &[], &[], false, None, &operands),
                expected
            );
        }
    }
    for instruction in [Instr::RightDiv, Instr::LeftDiv] {
        for operands in [
            vec![],
            vec![Value::Num(2.0)],
            vec![matrix.clone(), matrix.clone()],
        ] {
            assert_eq!(
                host_instruction_kind(&instruction, 0, &[], &[], false, None, &operands),
                VmFusionHostInstruction::Unproved
            );
        }
    }
}
