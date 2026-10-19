#![cfg(feature = "native-accel")]

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
#[path = "matmul_fusion_admission/fixture.rs"]
mod fixture;
#[path = "matmul_fusion_admission/no_upload.rs"]
mod no_upload;
#[path = "matmul_fusion_admission/selected_binding.rs"]
mod selected_binding;

fn tensor(data: &[f64], shape: &[usize]) -> Value {
    Value::Tensor(Tensor::new(data.to_vec(), shape.to_vec()).unwrap())
}

fn retire_result(handle: runmat_accelerate_api::GpuTensorHandle) {
    assert_eq!(
        handle.release().expect("fused result retirement"),
        runmat_accelerate_api::allocation::AllocationReleaseOutcome::Retired,
        "the fused result must retain no hidden aliases"
    );
}

fn direct(expression: &str) -> FusionExecutionOutcome<Value> {
    direct_with_left(expression, None)
}

fn direct_with_left(expression: &str, left: Option<Value>) -> FusionExecutionOutcome<Value> {
    let (group, inputs) = fixture::requested(expression, left);
    block_on(execute_matmul_epilogue_placed(FusionExecutionRequest {
        plan: &group,
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
            ("min(max(A*B,3),7)", [3., 6., 4., 7.]),
            ("(A*B).^2", [4., 36., 16., 64.]),
        ] {
            let FusionExecutionOutcome::Executed(Value::GpuTensor(handle)) = direct(expression)
            else {
                panic!("supported provider route lost: {expression}");
            };
            let result = block_on(PROVIDER.download(&handle)).unwrap();
            assert_eq!(result.shape, [2, 2]);
            assert_eq!(result.data, expected, "{expression}");
            retire_result(handle);
        }
    });
}

#[test]
fn admitted_diagonal_remains_on_provider_and_publishes_column() {
    runmat_gc::gc_test_context(|| {
        let _provider = ThreadProviderGuard::set(Some(&*PROVIDER));
        let FusionExecutionOutcome::Executed(Value::GpuTensor(handle)) = direct("diag(A*B)") else {
            panic!("diagonal provider route");
        };
        let result = block_on(PROVIDER.download(&handle)).unwrap();
        assert_eq!(result.shape, [2, 1]);
        assert_eq!(result.data, [2., 8.]);
        retire_result(handle);
    });
}

#[test]
fn power_proof_preserves_negative_integral_and_clamped_fractional_domains() {
    runmat_gc::gc_test_context(|| {
        let _provider = ThreadProviderGuard::set(Some(&*PROVIDER));
        for (expression, expected) in [
            ("(A*B).^2", [4., 36., 16., 64.]),
            ("max(A*B,0).^0.5", [0., 0., 2., 8.0f64.sqrt()]),
        ] {
            let FusionExecutionOutcome::Executed(Value::GpuTensor(handle)) =
                direct_with_left(expression, Some(tensor(&[-1., -3., 2., 4.], &[2, 2])))
            else {
                panic!("proved real domain");
            };
            let result = block_on(PROVIDER.download(&handle)).unwrap();
            assert_eq!(result.data, expected);
            retire_result(handle);
        }
    });
}

#[test]
fn actual_epilogue_keeps_nan_override_and_reverse_subtraction_zero_sign() {
    runmat_gc::gc_test_context(|| {
        let _provider = ThreadProviderGuard::set(Some(&*PROVIDER));
        let FusionExecutionOutcome::Executed(Value::GpuTensor(handle)) =
            direct_with_left("(A*B).^0", Some(tensor(&[f64::NAN, 1., 1., 1.], &[2, 2])))
        else {
            panic!("integral power route");
        };
        let result = block_on(PROVIDER.download(&handle)).unwrap();
        assert!(result.data[0].is_nan() && result.data[2].is_nan());
        assert_eq!(result.data[1], 1.);
        assert_eq!(result.data[3], 1.);
        retire_result(handle);
        let FusionExecutionOutcome::Executed(Value::GpuTensor(handle)) =
            direct_with_left("-0-(A*B)", Some(tensor(&[0.; 4], &[2, 2])))
        else {
            panic!("reverse subtract route");
        };
        let result = block_on(PROVIDER.download(&handle)).unwrap();
        assert!(result
            .data
            .iter()
            .all(|v| v.to_bits() == (-0.0f64).to_bits()));
        retire_result(handle);
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
