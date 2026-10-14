use super::*;
use futures::{executor::block_on, task::noop_waker_ref};
use runmat_accelerate_api::{
    test_support::{TestAllocation, TestAllocationOwner},
    NumericElementType,
};
use std::task::Context;

fn allocate(owner: &TestAllocationOwner) -> TestAllocation {
    owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap()
}

#[test]
fn pending_like_adaptation_owns_fresh_intermediate_and_preserves_caller_alias() {
    let owner = TestAllocationOwner::new();
    let input = allocate(&owner);
    let output = allocate(&owner);
    let mut future = Box::pin(evaluate(
        Value::GpuTensor(input.handle.clone()),
        Some(&input.handle),
        &Template::Default,
        move |_value, _protected| async move {
            RootedValue::new(Value::GpuTensor(output.handle))
                .map_err(|cause| cause.into_runtime_error())
        },
        |value, protected| async move {
            std::future::pending::<()>().await;
            drop(protected);
            RootedValue::new(value).map_err(|cause| cause.into_runtime_error())
        },
    ));
    assert!(future
        .as_mut()
        .poll(&mut Context::from_waker(noop_waker_ref()))
        .is_pending());
    assert_eq!(output.probe.retirements(), 0);
    drop(future);
    assert_eq!(output.probe.retirements(), 1);
    assert_eq!(input.probe.retirements(), 0);
    drop(input.handle);
    assert_eq!(input.probe.retirements(), 1);
}

#[test]
fn like_output_adoption_and_replacement_use_actual_value_ownership() {
    let owner = TestAllocationOwner::new();
    let allocation = allocate(&owner);
    let output =
        block_on(evaluate(
            Value::Num(0.0),
            None,
            &Template::Default,
            move |_, _| async move {
                RootedValue::new(Value::GpuTensor(allocation.handle))
                    .map_err(|cause| cause.into_runtime_error())
            },
            |value, _| async move {
                RootedValue::new(value).map_err(|cause| cause.into_runtime_error())
            },
        ))
        .unwrap();
    assert_eq!(allocation.probe.retirements(), 0);
    drop(output);
    assert_eq!(allocation.probe.retirements(), 1);
    let allocation = allocate(&owner);
    let output = block_on(evaluate(
        Value::Num(0.0),
        None,
        &Template::Default,
        move |_, _| async move {
            RootedValue::new(Value::GpuTensor(allocation.handle))
                .map_err(|cause| cause.into_runtime_error())
        },
        |value, _| async move {
            drop(value);
            RootedValue::new(Value::Num(1.0)).map_err(|cause| cause.into_runtime_error())
        },
    ))
    .unwrap();
    assert!(matches!(output.value(), Value::Num(1.0)));
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn like_adaptation_error_releases_result_without_consuming_borrowed_caller() {
    let owner = TestAllocationOwner::new();
    let input = allocate(&owner);
    let output = allocate(&owner);
    let result = block_on(evaluate(
        Value::GpuTensor(input.handle.clone()),
        Some(&input.handle),
        &Template::Default,
        move |_, _| async move {
            RootedValue::new(Value::GpuTensor(output.handle))
                .map_err(|cause| cause.into_runtime_error())
        },
        |value, _| async move {
            drop(value);
            Err(crate::build_runtime_error("fixture adaptation failed").build())
        },
    ));
    assert!(result.is_err());
    assert_eq!(output.probe.retirements(), 1);
    assert_eq!(input.probe.retirements(), 0);
    drop(input.handle);
    assert_eq!(input.probe.retirements(), 1);
}
