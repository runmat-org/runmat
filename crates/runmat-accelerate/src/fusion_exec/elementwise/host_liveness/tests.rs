use super::*;
use crate::fusion::{FusionGroup, FusionKernelSpec, FusionKind};
use crate::graph::{InstrSpan, ShapeInfo};
use std::collections::HashMap;

fn source(operations: Vec<FusionOp>, inputs: Vec<ValueId>, output: ValueId) -> FusionGroupPlan {
    FusionGroupPlan {
        index: 0,
        group: FusionGroup {
            id: 0,
            kind: FusionKind::ElementwiseChain,
            nodes: Vec::new(),
            shape: ShapeInfo::Tensor(vec![Some(4)]),
            span: InstrSpan { start: 0, end: 3 },
            pattern: None,
            stack_layout: None,
        },
        operations,
        inputs,
        stack_pattern: Vec::new(),
        constants: HashMap::new(),
        const_values: HashMap::new(),
        materialized_stores: Vec::new(),
        output: Some(output),
        kernel: FusionKernelSpec {
            kind: FusionKind::ElementwiseChain,
            supported: true,
        },
        reduction_data: None,
        reduction_dim: None,
        reduction_flavor: None,
        reduction_axes: None,
        pattern: None,
    }
}

fn primitive(op: PrimitiveOp, inputs: &[ValueId], output: ValueId) -> FusionOp {
    FusionOp::Primitive {
        op,
        inputs: inputs.to_vec(),
        output: Some(output),
    }
}

fn named(name: &str, inputs: &[ValueId], output: ValueId) -> FusionOp {
    FusionOp::Builtin {
        name: name.into(),
        origin: crate::graph::FusionCallOrigin::named(name),
        inputs: inputs.to_vec(),
        output: Some(output),
    }
}

fn real_slot(_: ValueId) -> Option<HostRealSlot> {
    Some(HostRealSlot {
        dtype: NumericDType::F32,
        shape: vec![4],
    })
}

#[test]
fn exact_named_output_joins_the_same_source_ordered_slot_plan() {
    let crate::graph::FusionCallOrigin::Named { binding } =
        crate::graph::FusionCallOrigin::named("sin")
    else {
        panic!("sin must have one canonical fusion binding");
    };
    let chain = source(
        vec![
            primitive(PrimitiveOp::Neg, &[0], 1),
            named("sin", &[1], 2),
            primitive(PrimitiveOp::Neg, &[2], 3),
        ],
        vec![0],
        3,
    );
    let canonical = [
        VmFusionHostOperation::Intrinsic(OperatorKind::UnaryMinus),
        VmFusionHostOperation::Builtin(binding),
        VmFusionHostOperation::Intrinsic(OperatorKind::UnaryMinus),
    ];
    let planned = plan_with(&chain, &canonical, real_slot, |index, candidate, arity| {
        (index == 1 && candidate == binding && arity == 1)
            .then_some(VmFusionHostInstruction::ElementwiseCall)
    })
    .expect("named exact-real liveness plan");
    assert_eq!(planned.slots.len(), 2);
    assert_eq!(
        planned
            .operations
            .iter()
            .map(|operation| operation.output_slot)
            .collect::<Vec<_>>(),
        [Some(0), Some(1), Some(0)]
    );
    assert_eq!(
        planned.operations[1].retirements[0],
        Some(HostOperandRetirement {
            operand_position: 0,
            slot: 0,
        })
    );
}

#[test]
fn whole_tensor_plan_reuses_only_after_last_read_and_coalesces_aliases() {
    let chain = source(
        vec![
            primitive(PrimitiveOp::UPlus, &[0], 1),
            primitive(PrimitiveOp::Neg, &[1], 2),
            primitive(PrimitiveOp::Mul, &[4, 2], 3),
            primitive(PrimitiveOp::Neg, &[3], 5),
        ],
        vec![0, 4],
        5,
    );
    let canonical = [
        VmFusionHostOperation::Intrinsic(OperatorKind::UnaryPlus),
        VmFusionHostOperation::Intrinsic(OperatorKind::UnaryMinus),
        VmFusionHostOperation::Intrinsic(OperatorKind::MatrixMultiply),
        VmFusionHostOperation::Intrinsic(OperatorKind::UnaryMinus),
    ];
    let planned =
        plan_with(&chain, &canonical, real_slot, |_, _, _| None).expect("intrinsic liveness plan");
    assert_eq!(planned.slots.len(), 2);
    assert_eq!(
        planned
            .operations
            .iter()
            .map(|operation| operation.output_slot)
            .collect::<Vec<_>>(),
        vec![None, Some(0), Some(1), Some(0)]
    );
    assert_eq!(
        planned.operations[2].retirements,
        [
            None,
            Some(HostOperandRetirement {
                operand_position: 1,
                slot: 0,
            }),
        ]
    );
    assert_eq!(
        planned.operations[3].retirements,
        [
            Some(HostOperandRetirement {
                operand_position: 0,
                slot: 1,
            }),
            None,
        ]
    );

    let mut pinned = chain.clone();
    pinned
        .materialized_stores
        .push(crate::fusion::FusionStoreMaterialization {
            value_id: 2,
            binding: crate::graph::VarBinding {
                kind: crate::graph::VarKind::Global,
                index: 0,
            },
        });
    let pinned =
        plan_with(&pinned, &canonical, real_slot, |_, _, _| None).expect("materialized pin plan");
    assert_eq!(pinned.slots.len(), 3);
    assert_eq!(
        pinned
            .operations
            .iter()
            .map(|operation| operation.output_slot)
            .collect::<Vec<_>>(),
        vec![None, Some(0), Some(1), Some(2)]
    );
    assert!(pinned.operations[2].retirements.iter().all(Option::is_none));

    let repeated = source(
        vec![
            primitive(PrimitiveOp::Neg, &[0], 1),
            primitive(PrimitiveOp::Mul, &[1, 1], 2),
        ],
        vec![0],
        2,
    );
    let repeated = plan_with(
        &repeated,
        &[
            VmFusionHostOperation::Intrinsic(OperatorKind::UnaryMinus),
            VmFusionHostOperation::Intrinsic(OperatorKind::MatrixMultiply),
        ],
        real_slot,
        |_, _, _| None,
    )
    .expect("repeated operand plan");
    assert_eq!(
        repeated.operations[1].retirements,
        [
            Some(HostOperandRetirement {
                operand_position: 0,
                slot: 0,
            }),
            Some(HostOperandRetirement {
                operand_position: 1,
                slot: 0,
            }),
        ]
    );
}
