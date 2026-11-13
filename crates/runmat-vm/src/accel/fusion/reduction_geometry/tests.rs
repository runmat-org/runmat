use super::*;
use runmat_accelerate::fusion::{FusionGroup, FusionKernelSpec, FusionOp};
use runmat_accelerate::graph::PrimitiveOp;
use runmat_accelerate::{FusionKind, InstrSpan, ShapeInfo};
use runmat_value::Tensor;
use std::collections::HashMap;

fn plan() -> FusionGroupPlan {
    FusionGroupPlan {
        index: 0,
        group: FusionGroup {
            id: 0,
            kind: FusionKind::Reduction,
            nodes: vec![],
            shape: ShapeInfo::Tensor(vec![Some(99), Some(99)]),
            span: InstrSpan { start: 0, end: 0 },
            pattern: None,
            stack_layout: None,
        },
        operations: vec![],
        inputs: vec![7],
        stack_pattern: vec![],
        constants: HashMap::new(),
        const_values: HashMap::new(),
        materialized_stores: vec![],
        output: Some(8),
        kernel: FusionKernelSpec {
            kind: FusionKind::Reduction,
            supported: true,
        },
        reduction_data: Some(7),
        reduction_dim: None,
        reduction_flavor: None,
        reduction_axes: Some(ReductionAxes::Unspecified),
        pattern: None,
    }
}
fn input(shape: &[usize]) -> Value {
    Value::Tensor(Tensor::new(vec![1.; cardinality(shape).unwrap()], shape.to_vec()).unwrap())
}
fn geometry(plan: &FusionGroupPlan, inputs: Vec<Value>) -> Result<ReductionGeometry, RuntimeError> {
    resolve_reduction_geometry(
        plan,
        &FusionExecutionRequest {
            plan,
            inputs,
            placement: None,
            runtime: None,
        },
    )
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn unsupported_complex_geometry_deopts_before_effects_and_restores_stack() {
    let plan = plan();
    let operand = Value::ComplexTensor(
        runmat_value::ComplexTensor::new(vec![(1.0, 2.0), (2.0, 1.0)], vec![1, 2]).unwrap(),
    );
    let original = vec![Value::Num(17.0), operand.clone()];
    let mut stack = crate::interpreter::stack::InterpreterStack::from_unbounded(original.clone());
    let guard = super::super::StackSliceGuard::new(&mut stack, 1)
        .expect("the unbounded test stack has a unique operand slice");
    let result = futures::executor::block_on(super::super::execute_fusion_reduction(
        &plan,
        FusionExecutionRequest {
            plan: &plan,
            inputs: vec![operand],
            placement: None,
            runtime: None,
        },
        guard,
        None,
    ));
    assert!(matches!(
        result,
        Err(super::super::FusionAttemptError::Deopt(_))
    ));
    assert_eq!(stack.as_slice(), original.as_slice());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn actual_shape_overrides_cached_shape_and_binds_default_row_axis_without_mutation() {
    let plan = plan();
    let geom = geometry(&plan, vec![input(&[1, 3])]).unwrap();
    assert_eq!(
        geom,
        ReductionGeometry {
            axis: 1,
            reduce_len: 3,
            num_slices: 1,
            output_shape: vec![1, 1]
        }
    );
    let specialized = bind_axis(&plan, &geom).unwrap();
    assert_eq!(plan.reduction_dim, None);
    assert!(plan.const_values.is_empty());
    let dim = specialized.reduction_dim.unwrap();
    assert_ne!(dim, 7);
    assert_eq!(specialized.const_values[&dim], Value::Num(2.));
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn exact_data_identity_and_expression_broadcast_are_required() {
    let mut plan = plan();
    plan.inputs = vec![9, 7];
    assert!(geometry(&plan, vec![input(&[3, 2]), input(&[2, 3])]).is_err());
    plan.operations.push(FusionOp::Primitive {
        op: PrimitiveOp::Add,
        inputs: vec![7, 9],
        output: Some(10),
    });
    plan.reduction_data = Some(10);
    let geom = geometry(&plan, vec![input(&[2, 3]), input(&[2, 3])]).unwrap();
    assert_eq!((geom.reduce_len, geom.num_slices), (2, 3));
    // Inference can derive [2,3], but today's kernel cannot index these leaves.
    assert!(geometry(&plan, vec![input(&[1, 3]), input(&[2, 1])]).is_err());
    assert!(geometry(&plan, vec![input(&[4, 1]), input(&[2, 3])]).is_err());
    plan.reduction_data = Some(11);
    assert!(geometry(&plan, vec![input(&[2, 3]), input(&[2, 3])]).is_err());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn empty_overflow_and_axis_never_become_invented_singletons() {
    assert_eq!(cardinality(&[usize::MAX, usize::MAX, 0]), Some(0));
    assert_eq!(cardinality(&[usize::MAX, 2]), None);
    assert_eq!(cardinality(&[]), Some(1));
    let mut plan = plan();
    for shape in [vec![0, 3], vec![3, 0]] {
        assert!(geometry(&plan, vec![input(&shape)]).is_err());
    }
    plan.reduction_axes = Some(ReductionAxes::Explicit(vec![2]));
    assert_eq!(
        geometry(&plan, vec![input(&[])]).unwrap(),
        ReductionGeometry {
            axis: 1,
            reduce_len: 1,
            num_slices: 1,
            output_shape: vec![1, 1],
        }
    );
    assert_eq!(
        geometry(&plan, vec![input(&[4])]).unwrap(),
        ReductionGeometry {
            axis: 1,
            reduce_len: 1,
            num_slices: 4,
            output_shape: vec![4, 1],
        }
    );
    plan.reduction_axes = Some(ReductionAxes::Explicit(vec![3]));
    assert!(geometry(&plan, vec![input(&[])]).is_err());
    assert!(geometry(&plan, vec![input(&[4])]).is_err());
    let rank_three = geometry(&plan, vec![input(&[2, 3, 4])]).unwrap();
    assert_eq!(
        rank_three,
        ReductionGeometry {
            axis: 2,
            reduce_len: 4,
            num_slices: 6,
            output_shape: vec![2, 3, 1],
        }
    );
    plan.reduction_axes = Some(ReductionAxes::All);
    let geom = geometry(&plan, vec![input(&[2, 2, 2])]).unwrap();
    assert_eq!(
        (geom.reduce_len, geom.num_slices, geom.output_shape),
        (8, 1, vec![1, 1])
    );
    assert!(geometry(&plan, vec![input(&[0, 3])]).is_err());
    for dim in [0., -1., 1.5, f64::INFINITY, f64::NAN] {
        assert!(axis_from_value(&Value::Num(dim)).is_err());
    }
    assert_eq!(axis_from_value(&Value::Num(2.)).unwrap(), 1);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn cycles_duplicate_producers_and_conflicting_axis_fail_closed() {
    let mut plan = plan();
    plan.operations.push(FusionOp::Primitive {
        op: PrimitiveOp::Add,
        inputs: vec![7, 10],
        output: Some(10),
    });
    plan.reduction_data = Some(10);
    assert!(geometry(&plan, vec![input(&[2, 3])]).is_err());
    plan.operations[0] = FusionOp::Primitive {
        op: PrimitiveOp::UPlus,
        inputs: vec![7],
        output: Some(10),
    };
    plan.operations.push(plan.operations[0].clone());
    assert!(geometry(&plan, vec![input(&[2, 3])]).is_err());
    plan.operations.clear();
    plan.reduction_data = Some(7);
    plan.reduction_dim = Some(10);
    plan.const_values.insert(10, Value::Num(2.));
    plan.reduction_axes = Some(ReductionAxes::Explicit(vec![1]));
    assert!(geometry(&plan, vec![input(&[2, 3])]).is_err());
}

#[cfg(not(target_arch = "wasm32"))]
#[path = "tests/direct_execution/mod.rs"]
mod direct_execution;
