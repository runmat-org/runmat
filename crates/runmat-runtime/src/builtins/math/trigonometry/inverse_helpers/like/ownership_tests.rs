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
        move |_value, _protected| async move { Ok(Value::GpuTensor(output.handle)) },
        |value, protected| async move {
            std::future::pending::<()>().await;
            drop(protected);
            Ok(value)
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
    let output = block_on(evaluate(
        Value::Num(0.0),
        None,
        &Template::Default,
        move |_, _| async move { Ok(Value::GpuTensor(allocation.handle)) },
        |value, _| async move { Ok(value) },
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
        move |_, _| async move { Ok(Value::GpuTensor(allocation.handle)) },
        |value, _| async move {
            drop(value);
            Ok(Value::Num(1.0))
        },
    ))
    .unwrap();
    assert!(matches!(output, Value::Num(1.0)));
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
        move |_, _| async move { Ok(Value::GpuTensor(output.handle)) },
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
