use super::*;
use crate::fusion::{FusionGroup, FusionKernelSpec, FusionKind};
use crate::graph::{InstrSpan, PrimitiveOp, ShapeInfo};
use crate::precision::PrecisionPolicy;
use runmat_accelerate_api::ProviderPrecision;
use runmat_value::Value;

#[path = "typed_producer_primitive_tests.rs"]
mod primitive_tests;

fn plan(operations: Vec<FusionOp>, output: ValueId) -> FusionGroupPlan {
    FusionGroupPlan {
        index: 0,
        group: FusionGroup {
            id: 0,
            kind: FusionKind::ElementwiseChain,
            nodes: vec![],
            shape: ShapeInfo::Scalar,
            span: InstrSpan { start: 0, end: 4 },
            pattern: None,
            stack_layout: None,
        },
        operations,
        inputs: vec![0, 1],
        stack_pattern: vec![0, 1],
        constants: HashMap::new(),
        const_values: HashMap::new(),
        materialized_stores: vec![],
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

fn add(left: ValueId, right: ValueId, output: ValueId) -> FusionOp {
    FusionOp::Primitive {
        op: PrimitiveOp::Add,
        inputs: vec![left, right],
        output: Some(output),
    }
}
fn leaves() -> HashMap<ValueId, String> {
    HashMap::from([(0, "input0.data[i0]".into()), (1, "input1.data[i1]".into())])
}
fn resolve(
    plan: &FusionGroupPlan,
    inputs: &[Value],
    precision: ProviderPrecision,
) -> ResolvedProgram {
    futures::executor::block_on(ResolvedProgram::resolve_with_policy(
        plan,
        inputs,
        precision,
        PrecisionPolicy::Exact,
    ))
    .unwrap()
}

#[test]
fn root_slice_keeps_shared_dependencies_once_and_original_temporary_indices() {
    let plan = plan(
        vec![add(0, 1, 2), add(0, 0, 3), add(2, 2, 4), add(2, 4, 5)],
        5,
    );
    let program = resolve(
        &plan,
        &[Value::Num(1.0), Value::Num(2.0)],
        ProviderPrecision::F64,
    );
    let sliced =
        TypedProducer::emit(&plan, &program, ScalarType::F64, leaves(), Some(&[4, 5])).unwrap();
    assert_eq!(sliced.body.matches("let typed0:").count(), 1);
    assert!(!sliced.body.contains("let typed1:"));
    assert!(sliced.body.contains("let typed2:"));
    assert!(sliced.body.contains("let typed3:"));
    assert_eq!(sliced.expression(4).unwrap(), "typed2");
    assert_eq!(sliced.expression(5).unwrap(), "typed3");
    let full = TypedProducer::emit(&plan, &program, ScalarType::F64, leaves(), None).unwrap();
    assert!(full.body.contains("let typed1:"));
    assert_eq!(
        full.preamble(&program).unwrap(),
        sliced.preamble(&program).unwrap()
    );
    // Slice identity conservatively retains every admitted fact, while shader
    // text independently captures the different selected producer body.
    assert_ne!(full.body, sliced.body);
}

#[test]
fn full_elementwise_source_and_rounding_are_unchanged() {
    let plan = plan(vec![add(0, 1, 2), add(2, 1, 3)], 3);
    let program = resolve(
        &plan,
        &[Value::Num(1.0), Value::Num(2.0)],
        ProviderPrecision::F64,
    );
    let producer = TypedProducer::emit(&plan, &program, ScalarType::F64, leaves(), None).unwrap();
    let expected_body = "    let typed0: f64 = (f64(input0.data[i0]) + f64(input1.data[i1]));\n    let typed1: f64 = (f64(typed0) + f64(input1.data[i1]));\n";
    assert_eq!(producer.body, expected_body);
    let expected = format!(
        "// {}\n\n{}",
        program.cache_identity().unwrap(),
        plan.build_wgsl_shader(
            "f64",
            "@group(0) @binding(2) var<storage, read_write> output0: Tensor;\n",
            3,
            expected_body,
            "    output0.data[g] = f64(typed1);\n"
        )
    );
    assert_eq!(
        plan.generate_typed_wgsl(&[3], &program, ScalarType::F64)
            .unwrap(),
        expected
    );

    let single = resolve(
        &plan,
        &[
            Value::Tensor(runmat_value::Tensor::from_f32(vec![16_777_216.0], vec![1, 1]).unwrap()),
            Value::Tensor(runmat_value::Tensor::from_f32(vec![1.0], vec![1, 1]).unwrap()),
        ],
        ProviderPrecision::F64,
    );
    let producer =
        TypedProducer::emit(&plan, &single, ScalarType::F64, leaves(), Some(&[3])).unwrap();
    assert!(producer.body.contains("let typed0: f32"));
    assert!(producer.body.contains("let typed1: f32"));
    assert_eq!((16_777_216.0_f32 + 1.0) + 1.0, 16_777_216.0);
    assert_eq!((16_777_216.0_f64 + 1.0) + 1.0, 16_777_218.0);
    let strict = resolve(
        &plan,
        &[Value::Num(1.0), Value::Num(2.0)],
        ProviderPrecision::F32,
    );
    assert!(TypedProducer::emit(&plan, &strict, ScalarType::F32, leaves(), Some(&[3])).is_err());
}

#[test]
fn input_root_keeps_supplied_physical_index_expression() {
    let plan = plan(vec![add(0, 1, 2)], 2);
    let program = resolve(
        &plan,
        &[Value::Num(1.0), Value::Num(2.0)],
        ProviderPrecision::F64,
    );
    let mut inputs = leaves();
    inputs.insert(0, "input0.data[broadcast_index]".into());
    let producer =
        TypedProducer::emit(&plan, &program, ScalarType::F64, inputs, Some(&[0])).unwrap();
    assert!(producer.body.is_empty());
    assert_eq!(
        producer.expression(0).unwrap(),
        "input0.data[broadcast_index]"
    );
    assert!(TypedProducer::emit(&plan, &program, ScalarType::F32, leaves(), Some(&[0])).is_err());
}

#[test]
fn canonical_helper_is_shared_by_full_and_sliced_emission() {
    let plan = plan(
        vec![FusionOp::Builtin {
            origin: Default::default(),
            name: "round".into(),
            inputs: vec![0],
            output: Some(2),
        }],
        2,
    );
    let program = resolve(
        &plan,
        &[Value::Num(2.5), Value::Num(0.0)],
        ProviderPrecision::F64,
    );
    let full = TypedProducer::emit(&plan, &program, ScalarType::F64, leaves(), None).unwrap();
    let sliced =
        TypedProducer::emit(&plan, &program, ScalarType::F64, leaves(), Some(&[2])).unwrap();
    assert_eq!(full.body, sliced.body);
    assert_eq!(
        full.preamble(&program).unwrap(),
        sliced.preamble(&program).unwrap()
    );
    let expected = crate::kernel_math::rounding::functions(ProviderPrecision::F64);
    assert_eq!(
        full.preamble(&program).unwrap(),
        format!("// {}\n{expected}\n", program.cache_identity().unwrap())
    );
    assert!(full.body.contains("rm_round_f64"));
}

#[test]
fn malformed_slice_dependencies_fail_closed() {
    let inputs = leaves();
    assert!(producer_indices(&plan(vec![add(0, 1, 2)], 2), &inputs, &[99]).is_err());
    assert!(producer_indices(&plan(vec![add(0, 99, 2)], 2), &inputs, &[2]).is_err());
    assert!(producer_indices(&plan(vec![add(0, 1, 2), add(1, 0, 2)], 2), &inputs, &[2]).is_err());
    assert!(producer_indices(&plan(vec![add(0, 1, 0)], 0), &inputs, &[0]).is_err());
    assert!(producer_indices(&plan(vec![add(3, 1, 2), add(2, 0, 3)], 3), &inputs, &[3]).is_err());
    assert!(producer_indices(&plan(vec![add(2, 1, 2)], 2), &inputs, &[2]).is_err());
}

#[test]
fn complete_emission_does_not_hide_errors_in_unrequested_operations() {
    let valid = plan(vec![add(0, 1, 2)], 2);
    let program = resolve(
        &valid,
        &[Value::Num(1.0), Value::Num(2.0)],
        ProviderPrecision::F64,
    );
    let malformed = plan(vec![add(0, 1, 2), add(99, 0, 3)], 2);
    // Full mode must still reject the later operation, even though only 2
    // would be written. It must not silently enable dead-code elimination.
    assert!(TypedProducer::emit(&malformed, &program, ScalarType::F64, leaves(), None).is_err());
}
