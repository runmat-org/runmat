use super::*;
use crate::builtins::math::elementwise::floating_conversions::test_support::{
    with_conversion_hook_provider, HookBehavior,
};
use runmat_accelerate_api::{AccelProvider, HostTensorView};

fn upload_input(provider: &dyn AccelProvider) -> runmat_accelerate_api::GpuTensorHandle {
    provider
        .upload(&HostTensorView {
            data: &[1.25, -2.5],
            shape: &[2, 1],
        })
        .expect("input upload")
}

#[test]
fn direct_double_conversion_preserves_input_and_provenance() {
    with_conversion_hook_provider(HookBehavior::Success, |provider| {
        let mut input = upload_input(provider);
        runmat_accelerate_api::set_handle_provenance(
            &mut input,
            runmat_accelerate_api::GpuHandleProvenance::Explicit,
        );

        let Value::GpuTensor(output) = double_builtin(Value::GpuTensor(input.clone()), Vec::new())
            .expect("direct double conversion")
        else {
            panic!("expected resident output")
        };
        assert_eq!(
            runmat_accelerate_api::handle_provenance(&output),
            Some(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        );
        assert!(block_on(provider.download(&input)).is_ok());
        provider.free(&input).expect("free input");
        provider.free(&output).expect("free output");
    });
}

#[test]
fn double_conversion_falls_back_only_for_typed_unsupported_hooks() {
    with_conversion_hook_provider(HookBehavior::Unsupported, |provider| {
        let input = upload_input(provider);
        let result = double_builtin(Value::GpuTensor(input.clone()), Vec::new())
            .expect("typed unsupported hook should fall back");
        assert_eq!(
            test_support::gather(result)
                .expect("fallback gather")
                .materialize_f64(),
            vec![1.25, -2.5]
        );
        assert!(block_on(provider.download(&input)).is_ok());
        provider.free(&input).expect("free input");
    });

    with_conversion_hook_provider(HookBehavior::Failure, |provider| {
        let input = upload_input(provider);
        let error = double_builtin(Value::GpuTensor(input.clone()), Vec::new())
            .expect_err("provider failure must remain visible");
        assert!(error
            .message()
            .contains("injected conversion kernel failure"));
        assert!(block_on(provider.download(&input)).is_ok());
        provider.free(&input).expect("free input");
    });
}

#[test]
fn malformed_double_output_is_freed_without_consuming_input() {
    with_conversion_hook_provider(HookBehavior::Malformed, |provider| {
        let input = upload_input(provider);
        let error = double_builtin(Value::GpuTensor(input.clone()), Vec::new())
            .expect_err("malformed output must be rejected");
        assert!(error.message().contains("malformed output"));
        let malformed = provider.take_malformed_output();
        assert!(block_on(provider.download(&malformed)).is_err());
        assert!(block_on(provider.download(&input)).is_ok());
        provider.free(&input).expect("free input");
    });
}

#[test]
fn single_and_complex_inputs_use_typed_preserving_fallbacks() {
    with_conversion_hook_provider(HookBehavior::Failure, |provider| {
        let single = Tensor::from_f32(vec![1.25, -2.5], vec![2, 1]).expect("single tensor");
        let single_input = gpu_helpers::upload_tensor(provider, &single).expect("single upload");
        let result = double_builtin(Value::GpuTensor(single_input.clone()), Vec::new())
            .expect("single conversion must bypass the double hook");
        let gathered = test_support::gather(result).expect("double gather");
        assert_eq!(gathered.numeric_dtype(), NumericDType::F64);
        assert_eq!(gathered.materialize_f64(), vec![1.25, -2.5]);
        assert!(block_on(provider.download_numeric(&single_input)).is_ok());
        provider.free(&single_input).expect("free single input");

        let complex = ComplexTensor::new(vec![(1.25, -2.5), (3.75, 4.5)], vec![2, 1])
            .expect("complex tensor");
        let complex_input =
            gpu_helpers::upload_complex_tensor(provider, &complex).expect("complex upload");
        let result = double_builtin(Value::GpuTensor(complex_input.clone()), Vec::new())
            .expect("complex conversion must use the preserving fallback");
        let gathered = block_on(gpu_helpers::gather_value_async(&result)).expect("complex gather");
        let Value::ComplexTensor(gathered) = gathered else {
            panic!("expected complex double tensor")
        };
        assert_eq!(gathered.numeric_dtype(), NumericDType::F64);
        assert_eq!(gathered.materialize_f64(), vec![(1.25, -2.5), (3.75, 4.5)]);
        assert!(block_on(provider.download_numeric(&complex_input)).is_ok());
        provider.free(&complex_input).expect("free complex input");
        if let Value::GpuTensor(output) = result {
            provider.free(&output).expect("free complex output");
        }
    });
}
