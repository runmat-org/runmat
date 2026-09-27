use super::*;
use runmat_accelerate_api::{GpuHandleProvenance, ThreadProviderGuard};
use std::{
    future::Future,
    sync::atomic::Ordering,
    task::{Context, Poll},
};
mod fixture;
use fixture::{all_retired_except, provider, Mode};

#[test]
fn broadcast_geometry_checks_overflow_and_preserves_empty_shapes() {
    assert!(broadcast_repeats(&[usize::MAX, 1], &[1, 2]).is_err());
    assert_eq!(
        broadcast_repeats(&[0, usize::MAX], &[1, usize::MAX])
            .unwrap()
            .unwrap()
            .0,
        vec![0, usize::MAX]
    );
    assert!(broadcast_repeats(&[2, 3], &[4, 3]).unwrap().is_none());
}

async fn full(name: &str, input: Value, other: Value) -> BuiltinResult<(Value, Value)> {
    match name {
        "min" => crate::builtins::math::reduction::min::evaluate(input, &[other])
            .await
            .map(|result| result.into_pair()),
        _ => crate::builtins::math::reduction::max::evaluate(input, &[other])
            .await
            .map(|result| result.into_pair()),
    }
}
#[test]
fn builtin_values_only_skips_origin_work_but_public_evaluate_preserves_exact_origins() {
    for name in ["min", "max"] {
        let provider = provider(Mode::Normal);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let left = provider.allocate(vec![0.0, -0.0, f64::NAN, 3.0], vec![4, 1]);
        let right = provider.allocate(vec![-0.0, 0.0, 2.0, f64::NAN], vec![4, 1]);
        let source = left
            .handle
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap();
        let args = [
            Value::GpuTensor(source.clone()),
            Value::GpuTensor(right.handle.clone()),
        ];
        let output = futures::executor::block_on(crate::call_builtin_async(name, &args)).unwrap();
        assert_eq!(provider.values.load(Ordering::SeqCst), 1);
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        let Value::GpuTensor(ref handle) = output else {
            panic!("device values")
        };
        assert!(runmat_accelerate_api::handle_is_explicit(handle));
        let data = provider.read(handle).unwrap();
        for zero in &data[..2] {
            assert_eq!(
                zero.to_bits(),
                if name == "min" {
                    (-0.0_f64).to_bits()
                } else {
                    0.0_f64.to_bits()
                }
            );
        }
        assert!(data[2].is_nan() && data[3].is_nan());
        drop(output);
        for count in [0, 1] {
            let output = futures::executor::block_on(crate::call_builtin_async_with_outputs(
                name, &args, count,
            ))
            .unwrap();
            let Value::OutputList(outputs) = output else {
                panic!("explicit output-count result")
            };
            assert_eq!(outputs.len(), count);
            assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
            drop(outputs);
        }
        assert_eq!(
            provider.values.load(Ordering::SeqCst),
            3,
            "zero-output form still executes and validates values"
        );
        // An ambient zero-output count must not weaken the public Rust API.
        let _zero = crate::output_count::push_output_count(Some(0));
        let (values, indices) =
            futures::executor::block_on(full(name, args[0].clone(), args[1].clone())).unwrap();
        assert!(matches!(values, Value::GpuTensor(_)));
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 2);
        let expected = if name == "min" {
            vec![2.0, 1.0, 1.0, 2.0]
        } else {
            vec![1.0, 2.0, 1.0, 2.0]
        };
        assert_eq!(
            indices,
            Value::Tensor(runmat_value::Tensor::new(expected, vec![4, 1]).unwrap())
        );
        drop((values, indices, args, source));
        all_retired_except(
            provider,
            &[left.handle.buffer_id(), right.handle.buffer_id()],
        );
        drop((left.handle, right.handle));
        all_retired_except(provider, &[]);
    }
}

#[test]
fn malformed_stages_and_pending_futures_preserve_inputs_and_retire_owned_intermediates() {
    for mode in [
        Mode::WrongValues,
        Mode::WrongFill,
        Mode::WrongRepmat,
        Mode::PendingValues,
        Mode::PendingDownload,
        Mode::FailedValues,
    ] {
        let provider = provider(mode);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let left = provider.allocate(vec![1.0, 2.0], vec![2, 1]);
        let right = provider.allocate(
            vec![3.0, 4.0],
            if matches!(mode, Mode::WrongRepmat) {
                vec![1, 2]
            } else {
                vec![2, 1]
            },
        );
        let other = if matches!(mode, Mode::WrongFill) {
            Value::Num(3.0)
        } else {
            Value::GpuTensor(right.handle.clone())
        };
        let mut future = Box::pin(full("max", Value::GpuTensor(left.handle.clone()), other));
        if matches!(mode, Mode::PendingValues | Mode::PendingDownload) {
            let mut context = Context::from_waker(futures::task::noop_waker_ref());
            assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
            assert_eq!(
                provider
                    .records
                    .lock()
                    .unwrap()
                    .last()
                    .unwrap()
                    .1
                    .retirements(),
                0
            );
            drop(future);
        } else {
            let error = futures::executor::block_on(future).unwrap_err();
            assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
            assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
            if matches!(mode, Mode::WrongFill | Mode::WrongRepmat) {
                assert_eq!(provider.values.load(Ordering::SeqCst), 0);
            }
        }
        left.handle.allocation_lease().validate_live().unwrap();
        right.handle.allocation_lease().validate_live().unwrap();
        all_retired_except(
            provider,
            &[left.handle.buffer_id(), right.handle.buffer_id()],
        );
        drop((left.handle, right.handle));
        all_retired_except(provider, &[]);
    }
}

#[test]
fn valid_immutable_result_alias_is_adopted_without_retiring_caller() {
    let provider = provider(Mode::AliasValues);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let left = provider.allocate(vec![1.0, 2.0], vec![2, 1]);
    let right = provider.allocate(vec![3.0, 4.0], vec![2, 1]);
    let output = futures::executor::block_on(crate::call_builtin_async(
        "min",
        &[
            Value::GpuTensor(left.handle.clone()),
            Value::GpuTensor(right.handle.clone()),
        ],
    ))
    .unwrap();
    let Value::GpuTensor(ref output_handle) = output else {
        panic!("resident alias")
    };
    assert!(output_handle.shares_allocation_with(&left.handle));
    drop(output);
    assert_eq!(left.probe.retirements(), 0);
    drop((left.handle, right.handle));
    all_retired_except(provider, &[]);
}

#[test]
fn broadcast_and_scalar_routes_keep_validated_values_and_exact_host_origins() {
    for name in ["min", "max"] {
        let provider = provider(Mode::Normal);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let left = provider.allocate(vec![0.0, -0.0], vec![2, 1]);
        let right = provider.allocate(vec![-0.0, 0.0], vec![1, 2]);
        let (values, indices) = futures::executor::block_on(full(
            name,
            Value::GpuTensor(left.handle.clone()),
            Value::GpuTensor(right.handle.clone()),
        ))
        .unwrap();
        let Value::GpuTensor(ref handle) = values else {
            panic!("broadcast values")
        };
        assert_eq!(handle.shape(), &[2, 2]);
        assert_eq!(provider.repeats.load(Ordering::SeqCst), 2);
        assert_eq!(
            provider.downloads.load(Ordering::SeqCst),
            2,
            "gather originals, not expanded temporaries"
        );
        let expected = if name == "min" {
            vec![2.0, 1.0, 1.0, 1.0]
        } else {
            vec![1.0, 1.0, 1.0, 2.0]
        };
        assert_eq!(
            indices,
            Value::Tensor(runmat_value::Tensor::new(expected, vec![2, 2]).unwrap())
        );
        drop((values, indices));
        for scalar_left in [false, true] {
            let (input, other) = if scalar_left {
                (Value::Num(-0.0), Value::GpuTensor(left.handle.clone()))
            } else {
                (Value::GpuTensor(left.handle.clone()), Value::Num(-0.0))
            };
            let before = provider.downloads.load(Ordering::SeqCst);
            let (values, indices) = futures::executor::block_on(full(name, input, other)).unwrap();
            assert!(matches!(values, Value::GpuTensor(_)));
            assert_eq!(provider.downloads.load(Ordering::SeqCst), before + 1);
            let expected = match (name, scalar_left) {
                ("min", false) | ("max", true) => vec![2.0, 1.0],
                _ => vec![1.0, 1.0],
            };
            assert_eq!(
                indices,
                Value::Tensor(runmat_value::Tensor::new(expected, vec![2, 1]).unwrap())
            );
            drop((values, indices));
        }
        all_retired_except(
            provider,
            &[left.handle.buffer_id(), right.handle.buffer_id()],
        );
        drop((left.handle, right.handle));
        all_retired_except(provider, &[]);
    }
}
