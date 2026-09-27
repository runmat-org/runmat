use super::*;
use crate::builtins::math::elementwise::floating_conversions::test_support::{
    with_conversion_hook_provider, HookBehavior,
};
use runmat_accelerate_api::{
    AccelProvider, HostIntegerDataView, HostIntegerTensorView, HostTensorView, NumericElementType,
};

#[test]
fn abandoned_direct_hook_restores_borrowed_metadata() {
    use std::future::Future;
    use std::task::{Context, Poll};
    with_conversion_hook_provider(HookBehavior::PendingMetadataMutation, |provider| {
        let input = upload_input(provider);
        let original = gpu_helpers::snapshot_handle_metadata(&input);
        let mut call = Box::pin(super::super::provider::single_from_gpu(input.clone(), &[]));
        let mut context = Context::from_waker(futures::task::noop_waker_ref());
        assert!(matches!(call.as_mut().poll(&mut context), Poll::Pending));
        assert!(runmat_accelerate_api::handle_is_logical(&input));
        drop(call);
        assert_eq!(gpu_helpers::snapshot_handle_metadata(&input), original);
        assert!(block_on(provider.download(&input)).is_ok());
        provider.free(&input).unwrap();
    });
}

fn upload_input(provider: &dyn AccelProvider) -> runmat_accelerate_api::GpuTensorHandle {
    provider
        .upload(&HostTensorView {
            data: &[1.25, -2.5],
            shape: &[2, 1],
        })
        .expect("input upload")
}

#[test]
fn dropping_template_download_releases_only_the_owned_conversion_result() {
    use std::future::Future;
    use std::task::{Context, Poll};
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    with_conversion_hook_provider(HookBehavior::PendingTemplateDownload, |provider| {
        let input = upload_input(provider);
        let before = gpu_helpers::snapshot_handle_metadata(&input);
        let mut call = Box::pin(super::super::single_builtin(
            Value::GpuTensor(input.clone()),
            vec![Value::from("like"), Value::Num(0.0)],
        ));
        let mut context = Context::from_waker(futures::task::noop_waker_ref());
        assert!(matches!(call.as_mut().poll(&mut context), Poll::Pending));
        let intermediate = provider.created_output();
        assert!(!provider.was_freed(&intermediate));
        drop(call);
        assert!(provider.was_freed(&intermediate));
        assert!(!provider.was_freed(&input));
        assert_eq!(gpu_helpers::snapshot_handle_metadata(&input), before);
        assert!(block_on(provider.download(&input)).is_ok());
        provider.free(&input).unwrap();
    });
}

#[test]
fn invalid_like_prototype_rejects_before_a_pending_conversion_hook() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    with_conversion_hook_provider(HookBehavior::PendingMetadataMutation, |provider| {
        let input = upload_input(provider);
        let before = gpu_helpers::snapshot_handle_metadata(&input);
        let error = single_builtin(
            Value::GpuTensor(input.clone()),
            vec![Value::from("like"), Value::from("not a prototype")],
        )
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            runmat_builtins::SINGLE_ERROR_INVALID_INPUT.identifier
        );
        assert_eq!(gpu_helpers::snapshot_handle_metadata(&input), before);
        provider.free(&input).unwrap();
    });
}

#[test]
fn unsupported_complex_like_input_rejects_without_gathering() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    with_conversion_hook_provider(HookBehavior::PendingTemplateDownload, |provider| {
        let prototype = upload_input(provider);
        let complex = runmat_value::ComplexTensor::new(vec![(1.0, 2.0)], vec![1, 1]).unwrap();
        let input = gpu_helpers::upload_complex_tensor(provider, &complex).unwrap();
        let error = single_builtin(
            Value::GpuTensor(input.clone()),
            vec![Value::from("like"), Value::GpuTensor(prototype.clone())],
        )
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            runmat_builtins::SINGLE_ERROR_INVALID_INPUT.identifier
        );
        assert!(!provider.was_freed(&input));
        assert!(!provider.was_freed(&prototype));
        provider.free(&input).unwrap();
        provider.free(&prototype).unwrap();
    });
}

#[test]
fn fixed_outputs_admit_discard_and_reject_multiple_before_provider_work() {
    for count in [None, Some(0), Some(1)] {
        let _outputs = crate::output_count::push_output_count(count);
        single_builtin(Value::Num(1.25), vec![]).expect("valid output count");
    }
    with_conversion_hook_provider(HookBehavior::PendingMetadataMutation, |provider| {
        let input = upload_input(provider);
        let before = gpu_helpers::snapshot_handle_metadata(&input);
        let _outputs = crate::output_count::push_output_count(Some(2));
        // This would never complete if output admission reached the hook.
        let error = single_builtin(Value::GpuTensor(input.clone()), vec![]).unwrap_err();
        assert_eq!(
            error.identifier(),
            runmat_builtins::SINGLE_ERROR_INVALID_ARGUMENT.identifier
        );
        assert_eq!(gpu_helpers::snapshot_handle_metadata(&input), before);
        provider.free(&input).unwrap();
    });
}

#[test]
fn direct_single_conversion_preserves_input_and_provenance() {
    with_conversion_hook_provider(HookBehavior::Success, |provider| {
        let mut input = upload_input(provider);
        runmat_accelerate_api::set_handle_provenance(
            &mut input,
            runmat_accelerate_api::GpuHandleProvenance::Explicit,
        );

        let Value::GpuTensor(output) = single_builtin(Value::GpuTensor(input.clone()), Vec::new())
            .expect("direct single conversion")
        else {
            panic!("expected resident output")
        };
        assert_eq!(
            output.descriptor.element_type,
            Some(NumericElementType::F32)
        );
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
fn single_conversion_falls_back_only_for_typed_unsupported_hooks() {
    with_conversion_hook_provider(HookBehavior::Unsupported, |provider| {
        let input = upload_input(provider);
        let result = single_builtin(Value::GpuTensor(input.clone()), Vec::new())
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
        let error = single_builtin(Value::GpuTensor(input.clone()), Vec::new())
            .expect_err("provider failure must remain visible");
        assert!(error
            .message()
            .contains("injected conversion kernel failure"));
        assert!(block_on(provider.download(&input)).is_ok());
        provider.free(&input).expect("free input");
    });
}

#[test]
fn malformed_single_output_is_freed_without_consuming_input() {
    with_conversion_hook_provider(HookBehavior::Malformed, |provider| {
        let input = upload_input(provider);
        let error = single_builtin(Value::GpuTensor(input.clone()), Vec::new())
            .expect_err("malformed output must be rejected");
        assert!(error.message().contains("malformed output"));
        let malformed = provider.take_malformed_output();
        assert!(block_on(provider.download(&malformed)).is_err());
        assert!(block_on(provider.download(&input)).is_ok());
        provider.free(&input).expect("free input");
    });
}

#[test]
fn integer_and_complex_inputs_use_typed_preserving_fallbacks() {
    with_conversion_hook_provider(HookBehavior::Failure, |provider| {
        let integer = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(&[1_u64 << 63, u64::MAX]),
                shape: &[2, 1],
            })
            .expect("integer upload");
        let result = single_builtin(Value::GpuTensor(integer.clone()), Vec::new())
            .expect("integer conversion must bypass the floating hook");
        let gathered = test_support::gather(result).expect("single gather");
        assert_eq!(gathered.numeric_dtype(), runmat_value::NumericDType::F32);
        assert_eq!(
            gathered.materialize_f64(),
            vec![(1_u64 << 63) as f32 as f64, u64::MAX as f32 as f64]
        );
        assert!(block_on(provider.download_integer(&integer)).is_ok());
        provider.free(&integer).expect("free integer input");

        let complex = ComplexTensor::new(vec![(1.25, -2.5), (3.75, 4.5)], vec![2, 1])
            .expect("complex tensor");
        let complex_input =
            gpu_helpers::upload_complex_tensor(provider, &complex).expect("complex upload");
        let result = single_builtin(Value::GpuTensor(complex_input.clone()), Vec::new())
            .expect("complex conversion must use the preserving fallback");
        let gathered = block_on(gpu_helpers::gather_value_async(&result)).expect("complex gather");
        let Value::ComplexTensor(gathered) = gathered else {
            panic!("expected complex single tensor")
        };
        assert_eq!(gathered.numeric_dtype(), runmat_value::NumericDType::F32);
        assert_eq!(gathered.materialize_f64(), vec![(1.25, -2.5), (3.75, 4.5)]);
        assert!(block_on(provider.download_numeric(&complex_input)).is_ok());
        provider.free(&complex_input).expect("free complex input");
        if let Value::GpuTensor(output) = result {
            provider.free(&output).expect("free complex output");
        }
    });
}
