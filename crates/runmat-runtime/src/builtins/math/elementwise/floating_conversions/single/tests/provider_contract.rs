use super::*;
use crate::builtins::math::elementwise::floating_conversions::test_support::{
    release_final_handles, with_conversion_hook_provider, HookBehavior,
};
use crate::context::RuntimeContext;
use crate::execution::RuntimeExecutionService;
use runmat_accelerate_api::{
    AccelProvider, HostIntegerDataView, HostIntegerTensorView, HostTensorView, NumericElementType,
};
use std::rc::Rc;

#[test]
fn abandoned_direct_hook_preserves_immutable_input_and_owner() {
    use std::future::Future;
    use std::task::{Context, Poll};
    with_conversion_hook_provider(HookBehavior::PendingDirectHook, |provider| {
        let input = upload_input(provider);
        let original = input.wire_descriptor();
        let mut call = Box::pin(super::super::provider::single_from_gpu(input.clone(), &[]));
        let mut context = Context::from_waker(futures::task::noop_waker_ref());
        assert!(matches!(call.as_mut().poll(&mut context), Poll::Pending));
        assert_eq!(provider.hook_calls(), 1);
        assert!(!runmat_accelerate_api::handle_is_logical(&input));
        assert_eq!(input.wire_descriptor(), original);
        assert_eq!(provider.numeric_download_calls(), 0);
        drop(call);
        assert_eq!(input.wire_descriptor(), original);
        assert!(block_on(provider.download(&input)).is_ok());
        release_final_handles([input]);
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
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let input = upload_input(provider);
        let before = input.wire_descriptor();
        let mut call = Box::pin(super::super::single_builtin(
            Some(runtime),
            Value::GpuTensor(input.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("like"),
                Value::Num(0.0),
            ]),
        ));
        let mut context = Context::from_waker(futures::task::noop_waker_ref());
        assert!(matches!(call.as_mut().poll(&mut context), Poll::Pending));
        assert_eq!(provider.numeric_download_calls(), 1);
        assert_eq!(provider.hook_calls(), 1);
        let intermediate = provider.take_created_output();
        assert!(intermediate.acquire().is_some());
        drop(call);
        assert!(intermediate.acquire().is_none());
        input
            .allocation_lease()
            .validate_live()
            .expect("caller input remains live");
        assert_eq!(input.wire_descriptor(), before);
        assert!(block_on(provider.download(&input)).is_ok());
        release_final_handles([input]);
    });
}

#[test]
fn invalid_like_prototype_rejects_before_a_pending_conversion_hook() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    with_conversion_hook_provider(HookBehavior::PendingDirectHook, |provider| {
        let input = upload_input(provider);
        let before = input.wire_descriptor();
        let error = single_builtin(
            Value::GpuTensor(input.clone()),
            vec![Value::from("like"), Value::from("not a prototype")],
        )
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            runmat_builtins::SINGLE_ERROR_INVALID_INPUT.identifier
        );
        assert_eq!(provider.hook_calls(), 0);
        assert_eq!(provider.numeric_download_calls(), 0);
        assert_eq!(input.wire_descriptor(), before);
        release_final_handles([input]);
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
        assert_eq!(provider.hook_calls(), 0);
        assert_eq!(provider.numeric_download_calls(), 0);
        input
            .allocation_lease()
            .validate_live()
            .expect("caller input remains live");
        prototype
            .allocation_lease()
            .validate_live()
            .expect("caller prototype remains live");
        release_final_handles([input]);
        release_final_handles([prototype]);
    });
}

#[test]
fn fixed_outputs_admit_discard_and_reject_multiple_before_provider_work() {
    for count in [None, Some(0), Some(1)] {
        let _outputs = crate::output_count::push_output_count(count).expect("output-count scope");
        single_builtin(Value::Num(1.25), vec![]).expect("valid output count");
    }
    with_conversion_hook_provider(HookBehavior::PendingDirectHook, |provider| {
        let input = upload_input(provider);
        let before = input.wire_descriptor();
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope"); // This would never complete if output admission reached the hook.
        let error = single_builtin(Value::GpuTensor(input.clone()), vec![]).unwrap_err();
        assert_eq!(
            error.identifier(),
            runmat_builtins::SINGLE_ERROR_INVALID_ARGUMENT.identifier
        );
        assert_eq!(provider.hook_calls(), 0);
        assert_eq!(provider.numeric_download_calls(), 0);
        assert_eq!(input.wire_descriptor(), before);
        release_final_handles([input]);
    });
}

#[test]
fn direct_single_conversion_preserves_input_and_provenance() {
    with_conversion_hook_provider(HookBehavior::Success, |provider| {
        let input = upload_input(provider)
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("explicit view");

        let Value::GpuTensor(output) = single_builtin(Value::GpuTensor(input.clone()), Vec::new())
            .expect("direct single conversion")
        else {
            panic!("expected resident output")
        };
        assert_eq!(
            output.descriptor().element_type,
            Some(NumericElementType::F32)
        );
        assert_eq!(
            runmat_accelerate_api::handle_provenance(&output),
            Some(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        );
        assert!(block_on(provider.download(&input)).is_ok());
        release_final_handles([input, output]);
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
        release_final_handles([input]);
    });

    with_conversion_hook_provider(HookBehavior::Failure, |provider| {
        let input = upload_input(provider);
        let error = single_builtin(Value::GpuTensor(input.clone()), Vec::new())
            .expect_err("provider failure must remain visible");
        assert!(error
            .message()
            .contains("injected conversion kernel failure"));
        assert!(block_on(provider.download(&input)).is_ok());
        release_final_handles([input]);
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
        drop(error);
        assert!(malformed.acquire().is_none());
        assert!(block_on(provider.download(&input)).is_ok());
        release_final_handles([input]);
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
        release_final_handles([integer]);

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
        let mut owned = vec![complex_input];
        if let Value::GpuTensor(output) = result {
            owned.push(output);
        }
        release_final_handles(owned);
    });
}
