use runmat_accelerate::{
    fusion::{FusionGroupPlan, FusionOp},
    FusionKind, ReductionAxes, ShapeInfo,
};
use runmat_value::{Tensor, Value};

fn compiled_plan() -> FusionGroupPlan {
    // The private A+1 producer is compiler-certified, not a published input.
    // Only the VM owner can project its private Store/Load binding before
    // preparing the exact retained reduction plan.
    let ast = runmat_parser::parse("A=rand(2,3); y=sum(A+1,1);").expect("authentic source");
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).expect("HIR");
    let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).expect("MIR");
    let bytecode = crate::bytecode::compile(&hir.assembly, &mir, hir.assembly.entrypoints[0].id)
        .expect("bytecode");
    let (plan, graph) = bytecode.prepare_vm_fusion_plan();
    assert!(graph.is_some(), "compiled reduction graph");
    let plan = plan.expect("VM-certified reduction plan");
    let plan = plan
        .groups
        .iter()
        .find(|group| group.group.kind == FusionKind::Reduction)
        .expect("mandatory reduction group")
        .clone();
    assert!(plan.kernel.supported);
    plan
}

pub(super) fn request_plan(shape: &[usize], axis: ReductionAxes, add: bool) -> FusionGroupPlan {
    let mut plan = compiled_plan();
    let data = plan.reduction_data.unwrap();
    // Request a reduction over an already materialized data value. Discard its
    // former producer; no unreachable operation may retain unbound operands.
    plan.operations.retain(
        |operation| matches!(operation, FusionOp::Builtin { output, .. } if *output == plan.output),
    );
    assert_eq!(plan.operations.len(), 1, "one compiled reduction binding");
    let FusionOp::Builtin {
        inputs: reduction_inputs,
        ..
    } = &mut plan.operations[0]
    else {
        unreachable!()
    };
    assert_eq!(reduction_inputs.first(), Some(&data));
    reduction_inputs.clear();
    reduction_inputs.push(data);
    plan.inputs = vec![data];
    plan.const_values.clear();
    plan.constants.clear();
    plan.reduction_dim = None;
    let dimension = data.max(plan.output.unwrap()) + 100;
    match &axis {
        ReductionAxes::Explicit(dims) => {
            plan.reduction_dim = Some(dimension);
            let control = match dims.as_slice() {
                [dimension] => Value::Num(*dimension as f64),
                _ => Value::Tensor(
                    Tensor::new(
                        dims.iter().map(|dimension| *dimension as f64).collect(),
                        vec![1, dims.len()],
                    )
                    .expect("dimension-vector fixture"),
                ),
            };
            plan.const_values.insert(dimension, control);
            reduction_inputs.push(dimension);
        }
        ReductionAxes::All => {
            plan.const_values
                .insert(dimension, Value::String("all".into()));
            reduction_inputs.push(dimension);
        }
        ReductionAxes::Unspecified => {}
    }
    plan.reduction_axes = Some(axis);
    // Deliberately stale cached shape must not decide actual runtime geometry.
    plan.group.shape = ShapeInfo::Tensor(vec![Some(99), Some(99)]);
    if add {
        let right = plan
            .operations
            .iter()
            .flat_map(|op| match op {
                FusionOp::Primitive { inputs, .. } | FusionOp::Builtin { inputs, .. } => inputs,
            })
            .copied()
            .max()
            .unwrap_or(data)
            + 100;
        let output = right + 1;
        let FusionOp::Builtin { inputs, .. } = &mut plan.operations[0] else {
            unreachable!()
        };
        inputs[0] = output;
        plan.inputs.push(right);
        plan.operations.insert(
            0,
            FusionOp::Primitive {
                op: runmat_accelerate::graph::PrimitiveOp::Add,
                inputs: vec![data, right],
                output: Some(output),
            },
        );
        plan.reduction_data = Some(output);
    }
    assert!(!shape.is_empty());
    plan
}

pub(super) fn tensor(shape: &[usize]) -> Value {
    let count = if shape.contains(&0) {
        0
    } else {
        shape.iter().product()
    };
    Value::Tensor(Tensor::new((1..=count).map(|n| n as f64).collect(), shape.to_vec()).unwrap())
}
