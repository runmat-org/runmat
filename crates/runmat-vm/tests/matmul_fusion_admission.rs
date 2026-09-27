#![cfg(feature = "native-accel")]
#![allow(clippy::result_large_err)]

#[path = "support/mod.rs"]
mod test_helpers;

use futures::executor::block_on;
use once_cell::sync::Lazy;
use runmat_accelerate::{
    fusion::prepare_fusion_plan,
    fusion_exec::{execute_matmul_epilogue_placed, FusionExecutionOutcome, FusionExecutionRequest},
    simple_provider::InProcessProvider,
    FusionKind,
};
use runmat_accelerate_api::{AccelProvider, ThreadProviderGuard};
use runmat_value::{Tensor, Value};

static PROVIDER: Lazy<InProcessProvider> = Lazy::new(InProcessProvider::new);

fn tensor(data: &[f64], shape: &[usize]) -> Value {
    Value::Tensor(Tensor::new(data.to_vec(), shape.to_vec()).unwrap())
}

fn direct(expression: &str) -> FusionExecutionOutcome<Value> {
    direct_with_left(expression, None)
}

fn direct_with_left(expression: &str, left: Option<Value>) -> FusionExecutionOutcome<Value> {
    let bytecode = test_helpers::compile_source(&format!(
        "A=rand(2,2); B=rand(2,2); C=rand(2,2); r=rand(2,1); c=rand(1,2); y={expression};"
    ))
    .unwrap();
    let windows = bytecode.runtime_fusion_groups();
    let graph = bytecode
        .runtime_accel_graph_for_fusion(&windows)
        .expect("matrix graph");
    let groups = bytecode.runtime_fusion_groups_for_graph(&graph);
    let plan = prepare_fusion_plan(
        Some(&graph),
        &groups,
        bytecode.fusion_metadata.mir_fusion_candidate_group_count,
    )
    .expect("matrix plan");
    let group = plan
        .groups
        .iter()
        .find(|group| group.group.kind == FusionKind::MatmulEpilogue)
        .expect("actual matrix group");
    let inputs = group
        .inputs
        .iter()
        .enumerate()
        .map(|(index, id)| {
            if let Some(value) = group.constants.get(&index) {
                return value.clone();
            }
            let binding = graph.var_binding(*id).expect("fixture variable input");
            match bytecode.var_names[&binding.index].as_str() {
                "A" => left
                    .clone()
                    .unwrap_or_else(|| tensor(&[1., 3., 2., 4.], &[2, 2])),
                "B" => tensor(&[2., 0., 0., 2.], &[2, 2]),
                "C" => tensor(&[1., 1., 1., 1.], &[2, 2]),
                "r" => tensor(&[2., 4.], &[2, 1]),
                "c" => tensor(&[2., 4.], &[1, 2]),
                name => panic!("unexpected fixture input {name}"),
            }
        })
        .collect();
    block_on(execute_matmul_epilogue_placed(FusionExecutionRequest {
        plan: group,
        inputs,
        placement: None,
        runtime: None,
    }))
    .unwrap()
}

#[test]
fn actual_provider_keeps_ordered_affine_and_scale_orientations() {
    runmat_gc::gc_test_context(|| {
        let _provider = ThreadProviderGuard::set(Some(&*PROVIDER));
        for (expression, expected) in [
            ("A*B", [2., 6., 4., 8.]),
            ("2-(A*B)", [0., -4., -2., -6.]),
            ("(A*B)-2", [0., 4., 2., 6.]),
            ("(A*B)*3+2", [8., 20., 14., 26.]),
            ("((A*B).*r)./c", [2., 12., 2., 8.]),
        ] {
            let FusionExecutionOutcome::Executed(Value::GpuTensor(handle)) = direct(expression)
            else {
                panic!("supported provider route lost: {expression}");
            };
            let result = block_on(PROVIDER.download(&handle)).unwrap();
            assert_eq!(result.shape, [2, 2]);
            assert_eq!(result.data, expected, "{expression}");
            PROVIDER.free(&handle).unwrap();
        }
    });
}

#[test]
fn unrepresentable_operations_fall_back_before_fused_execution() {
    runmat_gc::gc_test_context(|| {
        let _provider = ThreadProviderGuard::set(Some(&*PROVIDER));
        for expression in [
            "sin(A*B)",
            "(A*B)+C",
            "((A*B)+2)*3",
            "2./(A*B)",
            "((A*B).*c).*r",
            "(A*B).^0.5",
        ] {
            assert!(
                matches!(direct(expression), FusionExecutionOutcome::SharedRuntime(_)),
                "{expression}"
            );
        }
        let source = "A=[1,2;3,4]; B=[2,0;0,2]; x=A*B; y=sin(x); assert(isequal(x,[2,4;6,8])); assert(max(abs(y(:)-sin([2;6;4;8])))<1e-12); z=((A*B)+2)*3; assert(isequal(z,[12,18;24,30]));";
        let bytecode = test_helpers::compile_source(source).unwrap();
        test_helpers::interpret(&bytecode).unwrap();
    });
}

#[test]
fn actual_representation_not_compile_shape_controls_matrix_admission() {
    runmat_gc::gc_test_context(|| {
        let _provider = ThreadProviderGuard::set(Some(&*PROVIDER));
        for value in [
            Value::Num(2.0),
            Value::Complex(2.0, 1.0),
            tensor(&[2.0], &[1, 1]),
            tensor(&[1.0, 2.0, 3.0], &[1, 3]),
            tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]),
        ] {
            assert!(matches!(
                direct_with_left("A*B", Some(value)),
                FusionExecutionOutcome::SharedRuntime(_)
            ));
        }
    });
}
