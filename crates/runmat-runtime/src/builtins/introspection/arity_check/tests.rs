use super::*;
use runmat_value::{IntValue, IntegerStorage, Tensor};

fn runtime_with_arity(inputs: usize, outputs: usize) -> crate::context::RuntimeContext {
    let execution = std::rc::Rc::new(crate::execution::RuntimeExecutionService::new());
    crate::context::RuntimeContext::new(execution).with_invocation_arity(
        crate::context::RuntimeInvocationArity::function(inputs, outputs),
    )
}

#[test]
fn selected_dispatcher_admits_each_form_requested_output_count() {
    use crate::dispatcher::{
        call_builtin_async_in_context, call_builtin_async_with_outputs_in_context,
    };
    let runtime = runtime_with_arity(1, 1);
    let bounds = [Value::Num(0.0), Value::Num(2.0)];
    for name in [NARGINCHK_BUILTIN_NAME, NARGOUTCHK_BUILTIN_NAME] {
        let discarded = futures::executor::block_on(call_builtin_async_with_outputs_in_context(
            &runtime, name, &bounds, 0,
        ))
        .expect("zero-output form is discarded");
        assert_eq!(discarded, Value::output_list(Vec::new()));
        let implicit =
            futures::executor::block_on(call_builtin_async_in_context(&runtime, name, &bounds))
                .expect("implicit statement form");
        assert_eq!(implicit, Value::output_list(Vec::new()));
        let error = futures::executor::block_on(call_builtin_async_with_outputs_in_context(
            &runtime, name, &bounds, 1,
        ))
        .expect_err("zero-output form cannot be assigned");
        assert_eq!(error.identifier(), Some("RunMat:TooManyOutputs"));
    }

    for (arguments, expected_struct) in [
        (
            vec![Value::Num(0.0), Value::Num(1.0), Value::Num(2.0)],
            false,
        ),
        (
            vec![
                Value::Num(0.0),
                Value::Num(1.0),
                Value::Num(2.0),
                Value::from("struct"),
            ],
            true,
        ),
    ] {
        let one = futures::executor::block_on(call_builtin_async_with_outputs_in_context(
            &runtime,
            NARGOUTCHK_BUILTIN_NAME,
            &arguments,
            1,
        ))
        .expect("legacy result");
        assert_eq!(matches!(one.value(), Value::Struct(_)), expected_struct);
        let discarded = futures::executor::block_on(call_builtin_async_with_outputs_in_context(
            &runtime,
            NARGOUTCHK_BUILTIN_NAME,
            &arguments,
            0,
        ))
        .expect("discarded legacy result");
        assert_eq!(discarded, Value::output_list(Vec::new()));
        let error = futures::executor::block_on(call_builtin_async_with_outputs_in_context(
            &runtime,
            NARGOUTCHK_BUILTIN_NAME,
            &arguments,
            2,
        ))
        .expect_err("legacy form has one output");
        assert_eq!(error.identifier(), Some("RunMat:TooManyOutputs"));
    }
}

#[test]
fn form_output_admission_precedes_bounds_and_function_context() {
    let script = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ))
    .with_invocation_arity(crate::context::RuntimeInvocationArity::Script);
    let bad_bounds = vec![Value::Num(-1.0), Value::Num(0.0)];
    for (name, args) in [
        (NARGINCHK_BUILTIN_NAME, bad_bounds.clone()),
        (NARGOUTCHK_BUILTIN_NAME, bad_bounds),
    ] {
        let error = futures::executor::block_on(
            crate::dispatcher::call_builtin_async_with_outputs_in_context(&script, name, &args, 1),
        )
        .expect_err("requested result is rejected before bound inspection");
        assert_eq!(error.identifier(), Some("RunMat:TooManyOutputs"));
    }
}

#[test]
fn narginchk_uses_runtime_call_count_context() {
    let runtime = runtime_with_arity(2, 1);
    let value =
        dispatch_narginchk_in_context(Some(&runtime), vec![Value::Num(1.0), Value::Num(2.0)])
            .expect("narginchk succeeds");
    assert_eq!(value, Value::output_list(Vec::new()));
}

#[test]
fn nargoutchk_uses_runtime_call_count_context() {
    let runtime = runtime_with_arity(2, 1);
    let value =
        dispatch_nargoutchk_in_context(Some(&runtime), vec![Value::Num(1.0), Value::Num(1.0)])
            .expect("nargoutchk succeeds");
    assert_eq!(value, Value::output_list(Vec::new()));
}

#[test]
fn runtime_arity_helpers_report_context_unavailable_without_vm_context() {
    let err = dispatch_narginchk_in_context(None, vec![Value::Num(0.0), Value::Num(1.0)])
        .expect_err("missing context should fail");
    assert_eq!(err.identifier(), Some("RunMat:NarginchkContextUnavailable"));

    let err = dispatch_nargoutchk(vec![Value::Num(0.0), Value::Num(1.0)])
        .expect_err("missing context should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:NargoutchkContextUnavailable")
    );
}

#[test]
fn script_context_keeps_function_checks_unavailable_but_legacy_count_is_explicit() {
    let execution = std::rc::Rc::new(crate::execution::RuntimeExecutionService::new());
    let script = crate::context::RuntimeContext::new(execution)
        .with_invocation_arity(crate::context::RuntimeInvocationArity::Script);
    let error =
        dispatch_narginchk_in_context(Some(&script), vec![Value::Num(0.0), Value::Num(0.0)])
            .expect_err("scripts have no function input count");
    assert_eq!(
        error.identifier(),
        Some("RunMat:NarginchkContextUnavailable")
    );
    let error =
        dispatch_nargoutchk_in_context(Some(&script), vec![Value::Num(0.0), Value::Num(0.0)])
            .expect_err("scripts have no function output count");
    assert_eq!(
        error.identifier(),
        Some("RunMat:NargoutchkContextUnavailable")
    );
    let explicit = dispatch_nargoutchk_in_context(
        Some(&script),
        vec![Value::Num(0.0), Value::Num(0.0), Value::Num(0.0)],
    )
    .expect("legacy explicit count has no function-context dependency");
    assert!(matches!(explicit, Value::CharArray(value) if value.data().is_empty()));
}

#[test]
fn interleaved_sibling_call_futures_keep_their_selected_arity_after_drop() {
    use std::future::Future;
    use std::task::{Context, Poll};

    let execution = std::rc::Rc::new(crate::execution::RuntimeExecutionService::new());
    let parent = crate::context::RuntimeContext::new(execution);
    let first = parent
        .clone()
        .with_invocation_arity(crate::context::RuntimeInvocationArity::function(1, 0));
    let second = parent
        .clone()
        .with_invocation_arity(crate::context::RuntimeInvocationArity::function(3, 2));
    let (first_sender, first_gate) = futures::channel::oneshot::channel::<()>();
    let (second_sender, second_gate) = futures::channel::oneshot::channel::<()>();
    let first_selected = first.clone();
    let second_selected = second.clone();
    let mut first_future = Box::pin(first.scope(async move {
        let _ = first_gate.await;
        dispatch_narginchk_in_context(
            Some(&first_selected),
            vec![Value::Num(1.0), Value::Num(1.0)],
        )
    }));
    let mut second_future = Box::pin(second.scope(async move {
        let _ = second_gate.await;
        dispatch_nargoutchk_in_context(
            Some(&second_selected),
            vec![Value::Num(2.0), Value::Num(2.0)],
        )
    }));
    let mut context = Context::from_waker(futures::task::noop_waker_ref());
    assert!(first_future.as_mut().poll(&mut context).is_pending());
    assert!(second_future.as_mut().poll(&mut context).is_pending());
    drop(first_future);
    assert!(first_sender.send(()).is_err());
    second_sender.send(()).expect("second call remains live");
    assert!(matches!(
        second_future.as_mut().poll(&mut context),
        Poll::Ready(Ok(Value::OutputList(values))) if values.is_empty()
    ));
    assert_eq!(parent.invocation_arity(), None);
}

#[test]
fn arity_bounds_read_typed_integer_tensor_storage_exactly() {
    let runtime = runtime_with_arity(2, 1);
    let min = Tensor::new_integer(IntegerStorage::U64(vec![2]), vec![1, 1]).expect("integer min");
    let max = Tensor::new_integer(IntegerStorage::U64(vec![2]), vec![1, 1]).expect("integer max");

    let value =
        dispatch_narginchk_in_context(Some(&runtime), vec![Value::Tensor(min), Value::Tensor(max)])
            .expect("narginchk succeeds");
    assert_eq!(value, Value::output_list(Vec::new()));

    assert_eq!(
        parse_finite_arity_bound(
            &Value::Int(IntValue::U16(2)),
            "narginchk",
            &NARGINCHK_ERROR_BOUNDS_INVALID
        )
        .unwrap(),
        2
    );
    for value in [
        Value::Int(IntValue::I8(-1)),
        Value::Num(1.5),
        Value::Num(usize::MAX as f64),
        Value::Num(usize::MAX as f64 + 1.0),
    ] {
        assert!(
            parse_finite_arity_bound(&value, "narginchk", &NARGINCHK_ERROR_BOUNDS_INVALID).is_err()
        );
    }
}

#[test]
fn arity_bound_reads_u16_tensor_storage_exactly() {
    let tensor =
        Tensor::new_integer(IntegerStorage::U16(vec![7]), vec![1, 1]).expect("arity tensor");
    assert_eq!(
        parse_finite_arity_bound(
            &Value::Tensor(tensor),
            "narginchk",
            &NARGINCHK_ERROR_BOUNDS_INVALID,
        )
        .unwrap(),
        7,
    );
}

#[test]
fn arity_bound_checks_wide_scalars_and_negative_integer_tensors() {
    let scalar = Value::Int(IntValue::U64(u64::MAX));
    let decoded = parse_finite_arity_bound(&scalar, "narginchk", &NARGINCHK_ERROR_BOUNDS_INVALID);
    if usize::BITS == 64 {
        assert_eq!(decoded.unwrap(), usize::MAX);
    } else {
        assert!(decoded.is_err());
    }
    let tensor =
        Tensor::new_integer(IntegerStorage::I64(vec![-1]), vec![1, 1]).expect("arity tensor");
    assert!(parse_finite_arity_bound(
        &Value::Tensor(tensor),
        "narginchk",
        &NARGINCHK_ERROR_BOUNDS_INVALID,
    )
    .is_err());
}

#[test]
fn positive_single_infinity_is_an_admitted_maximum() {
    let runtime = runtime_with_arity(1, 1);
    let positive_single_infinity =
        Value::Tensor(Tensor::from_f32(vec![f32::INFINITY], vec![1, 1]).expect("single scalar"));
    for name in [NARGINCHK_BUILTIN_NAME, NARGOUTCHK_BUILTIN_NAME] {
        let value = futures::executor::block_on(
            crate::dispatcher::call_builtin_async_with_outputs_in_context(
                &runtime,
                name,
                &[Value::Num(0.0), positive_single_infinity.clone()],
                0,
            ),
        )
        .expect("single positive infinity admits an unbounded max");
        assert_eq!(value, Value::output_list(Vec::new()));
    }
}

#[test]
fn arity_checks_accept_every_documented_integer_class() {
    let runtime = runtime_with_arity(1, 1);
    let cases = [
        (IntValue::I8(0), IntValue::I8(2), IntValue::I8(3)),
        (IntValue::I16(0), IntValue::I16(2), IntValue::I16(3)),
        (IntValue::I32(0), IntValue::I32(2), IntValue::I32(3)),
        (IntValue::I64(0), IntValue::I64(2), IntValue::I64(3)),
        (IntValue::U8(0), IntValue::U8(2), IntValue::U8(3)),
        (IntValue::U16(0), IntValue::U16(2), IntValue::U16(3)),
        (IntValue::U32(0), IntValue::U32(2), IntValue::U32(3)),
        (IntValue::U64(0), IntValue::U64(2), IntValue::U64(3)),
    ];
    for (min, max, invalid_outputs) in cases {
        assert_eq!(
            dispatch_narginchk_in_context(
                Some(&runtime),
                vec![Value::Int(min.clone()), Value::Int(max.clone())]
            )
            .expect("documented integer narginchk bounds"),
            Value::output_list(Vec::new())
        );
        assert_eq!(
            dispatch_nargoutchk_in_context(
                Some(&runtime),
                vec![Value::Int(min.clone()), Value::Int(max.clone())]
            )
            .expect("documented integer nargoutchk bounds"),
            Value::output_list(Vec::new())
        );
        let text = dispatch_nargoutchk(vec![
            Value::Int(min.clone()),
            Value::Int(max.clone()),
            Value::Int(invalid_outputs.clone()),
        ])
        .expect("legacy text nargoutchk");
        assert!(
            matches!(text, Value::CharArray(array) if runmat_value::decode_utf16_code_units(array.data()).expect("valid UTF-16 fixture").contains("Too many"))
        );

        let structured = dispatch_nargoutchk(vec![
            Value::Int(min),
            Value::Int(max),
            Value::Int(invalid_outputs),
            Value::from("struct"),
        ])
        .expect("legacy structured nargoutchk");
        assert!(
            matches!(structured, Value::Struct(value) if value.fields.contains_key("message") && value.fields.contains_key("identifier"))
        );
    }
}

#[test]
fn legacy_nargoutchk_returns_documented_empty_result_classes() {
    let text = dispatch_nargoutchk(vec![
        Value::Int(IntValue::U8(0)),
        Value::Int(IntValue::U8(2)),
        Value::Int(IntValue::U8(1)),
    ])
    .expect("valid legacy text form");
    assert!(
        matches!(text, Value::CharArray(array) if array.shape == vec![0, 0] && array.data().is_empty())
    );

    let structured = dispatch_nargoutchk(vec![
        Value::Int(IntValue::U8(0)),
        Value::Int(IntValue::U8(2)),
        Value::Int(IntValue::U8(1)),
        Value::from("struct"),
    ])
    .expect("valid legacy structure form");
    assert!(matches!(structured, Value::Struct(value) if value.fields.is_empty()));
}

#[test]
fn arity_checks_reject_resident_bounds_before_provider_access() {
    let runtime = runtime_with_arity(1, 1);
    let resident = Value::GpuTensor(
        runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("valid unregistered resident fixture")
            .handle,
    );
    let nargin =
        dispatch_narginchk_in_context(Some(&runtime), vec![resident.clone(), Value::Num(2.0)])
            .expect_err("resident nargin bound must reject");
    assert_eq!(nargin.identifier(), Some("RunMat:NarginchkArgumentInvalid"));
    let nargout = dispatch_nargoutchk_in_context(Some(&runtime), vec![resident, Value::Num(2.0)])
        .expect_err("resident nargout bound must reject");
    assert_eq!(
        nargout.identifier(),
        Some("RunMat:NargoutchkArgumentInvalid")
    );
}
