use super::*;
use crate::fusion::{FusionGroup, FusionKernelSpec, FusionKind};
use crate::graph::{InstrSpan, ShapeInfo};
use runmat_runtime::builtins::common::spec::ScalarType;

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

fn cast(name: &str, input: ValueId, output: ValueId) -> FusionOp {
    FusionOp::Builtin {
        name: name.into(),
        inputs: vec![input],
        output: Some(output),
    }
}

#[test]
fn interior_single_arithmetic_remains_f32_before_double_publication() {
    let plan = plan(
        vec![
            cast("single", 0, 2),
            cast("single", 1, 3),
            FusionOp::Primitive {
                op: PrimitiveOp::Add,
                inputs: vec![2, 3],
                output: Some(4),
            },
            cast("double", 4, 5),
        ],
        5,
    );
    let inputs = [Value::Num(16_777_216.0), Value::Num(1.0)];
    let program = ResolvedProgram::resolve(&plan, &inputs).unwrap();
    assert_eq!(program.floating_class(4).unwrap(), NumericClass::Single);
    assert_eq!(program.floating_class(5).unwrap(), NumericClass::Double);
    let shader = plan
        .generate_typed_wgsl(&[5], &program, ScalarType::F64)
        .unwrap();
    assert!(shader.contains("let typed2: f32"), "{shader}");
    assert!(shader.contains("let typed3: f64"), "{shader}");
    // Host oracle demonstrates why widening only the final result is wrong.
    assert_eq!((16_777_216.0_f32 + 1.0_f32) as f64, 16_777_216.0);
    assert_eq!(16_777_216.0_f64 + 1.0_f64, 16_777_217.0);
}

#[test]
fn equal_final_class_does_not_collapse_different_interior_program_identities() {
    let rounded = plan(vec![cast("single", 0, 2), cast("double", 2, 3)], 3);
    let direct = plan(vec![cast("double", 0, 3)], 3);
    let inputs = [Value::Num(16_777_217.0), Value::Num(-0.0)];
    let first = ResolvedProgram::resolve(&rounded, &inputs).unwrap();
    let second = ResolvedProgram::resolve(&direct, &inputs).unwrap();
    assert_eq!(
        first.floating_class(3).unwrap(),
        second.floating_class(3).unwrap()
    );
    assert_ne!(
        first.cache_identity().unwrap(),
        second.cache_identity().unwrap()
    );
    assert_eq!(
        first.cache_identity().unwrap(),
        ResolvedProgram::resolve(&rounded, &inputs)
            .unwrap()
            .cache_identity()
            .unwrap()
    );
}

#[test]
fn real_inputs_do_not_fabricate_power_domain_evidence() {
    let plan = plan(
        vec![FusionOp::Primitive {
            op: PrimitiveOp::ElemPow,
            inputs: vec![0, 1],
            output: Some(2),
        }],
        2,
    );
    assert!(ResolvedProgram::resolve(&plan, &[Value::Num(-1.0), Value::Num(0.5)]).is_err());
}
