use super::*;

#[test]
fn unary_and_complex_dispatch_preserve_exact_terminal_causes() {
    for (name, hook, probe) in [
        ("erf", "unary_erf", false),
        ("erfcinv", "unary_erfcinv", false),
        ("factorial", "unary_factorial", false),
        ("gamma", "unary_gamma", false),
        ("gammaln", "unary_gammaln", true),
        ("sqrt", "unary_sqrt", true),
        ("realsqrt", "unary_sqrt", true),
        ("pow2", "unary_pow2", false),
        ("nextpow2", "unary_nextpow2", false),
        ("log", "unary_log", true),
        ("log2", "unary_log2", true),
        ("log10", "unary_log10", true),
        ("log1p", "unary_log1p", true),
        ("real", "unary_real", false),
        ("imag", "unary_imag", false),
        ("conj", "unary_conj", false),
        ("complex", "complex_from_real", false),
    ] {
        check(name, hook, probe, false);
    }
    check("complex", "complex_from_real_imag", false, true);
}

#[test]
fn matrix_dispatch_preserves_each_provider_boundary_and_owned_transpose() {
    for (name, hook) in [
        ("cond", "cond"),
        ("rank", "rank"),
        ("inv", "inv"),
        ("pinv", "pinv"),
        ("norm", "norm"),
        ("rcond", "rcond"),
    ] {
        check_case(
            name,
            hook,
            &[hook],
            vec![2, 2],
            GpuTensorStorage::Real,
            |left, _| vec![Value::GpuTensor(left.clone())],
        );
    }
    for (name, hook, shape) in [
        ("dot", "dot", vec![3, 1]),
        ("cross", "cross", vec![3, 1]),
        ("mtimes", "matmul", vec![2, 2]),
    ] {
        check_case(
            name,
            hook,
            &[hook],
            shape,
            GpuTensorStorage::Real,
            |left, right| {
                vec![
                    Value::GpuTensor(left.clone()),
                    Value::GpuTensor(right.clone()),
                ]
            },
        );
    }
    for scalar_first in [false, true] {
        check_case(
            "mtimes",
            "scalar_mul",
            &["scalar_mul"],
            vec![2, 2],
            GpuTensorStorage::Real,
            |left, _| {
                let resident = Value::GpuTensor(left.clone());
                if scalar_first {
                    vec![Value::Num(2.0), resident]
                } else {
                    vec![resident, Value::Num(2.0)]
                }
            },
        );
    }
    for (hook, shape, storage, trace) in [
        (
            "transpose",
            vec![2, 3],
            GpuTensorStorage::Real,
            vec!["transpose"],
        ),
        (
            "permute",
            vec![2, 3, 1],
            GpuTensorStorage::Real,
            vec!["permute"],
        ),
        (
            "unary_conj",
            vec![2, 3],
            GpuTensorStorage::ComplexInterleaved,
            vec!["transpose", "unary_conj"],
        ),
    ] {
        check_case("ctranspose", hook, &trace, shape, storage, |left, _| {
            vec![Value::GpuTensor(left.clone())]
        });
    }
}

#[test]
fn malformed_scalar_hooks_retire_unpublished_output_without_host_replay() {
    for (name, failure, hook) in [
        ("cond", "cond_malformed", "cond"),
        ("rank", "rank_malformed", "rank"),
    ] {
        let provider = Box::leak(Box::new(Provider {
            owner: TestAllocationOwner::new(),
            failure,
            trace: Mutex::new(Vec::new()),
            probes: Mutex::new(Vec::new()),
        }));
        let _selected = ThreadProviderGuard::set(Some(provider));
        let source = provider
            .owner
            .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let arguments = [Value::GpuTensor(source.handle.clone())];
        let error =
            futures::executor::block_on(runmat_runtime::call_builtin_async(name, &arguments))
                .expect_err("malformed provider scalar must be rejected");
        assert_eq!(
            error.gpu_gather_retry,
            runmat_runtime::GpuGatherRetry::Never
        );
        assert_eq!(*provider.trace.lock().unwrap(), [hook]);
        let probes = provider.probes.lock().unwrap();
        assert_eq!(probes.len(), 1);
        assert_eq!(probes[0].1.retirements(), 1);
        source.handle.allocation_lease().validate_live().unwrap();
    }
}

#[test]
fn valid_scalar_hooks_publish_one_owned_resident_result() {
    for (name, failure, hook) in [
        ("cond", "cond_valid", "cond"),
        ("rank", "rank_valid", "rank"),
    ] {
        let provider = Box::leak(Box::new(Provider {
            owner: TestAllocationOwner::new(),
            failure,
            trace: Mutex::new(Vec::new()),
            probes: Mutex::new(Vec::new()),
        }));
        let _selected = ThreadProviderGuard::set(Some(provider));
        let source = provider
            .owner
            .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let arguments = [Value::GpuTensor(source.handle.clone())];
        let rooted_result =
            futures::executor::block_on(runmat_runtime::call_builtin_async(name, &arguments))
                .expect("validated scalar publication");
        let Value::GpuTensor(result) = rooted_result.value() else {
            panic!("{name} must retain the provider-owned scalar");
        };
        assert_eq!(result.shape(), [1, 1]);
        assert_eq!(result.device_id(), source.handle.device_id());
        assert_eq!(*provider.trace.lock().unwrap(), [hook]);
        assert_eq!(provider.probes.lock().unwrap().len(), 1);
        source.handle.allocation_lease().validate_live().unwrap();
    }
}

#[test]
fn after_effect_unsupported_scalar_hooks_never_replay() {
    for (name, failure, hook) in [
        ("cond", "cond_after_unsupported", "cond"),
        ("rank", "rank_after_unsupported", "rank"),
        ("inv", "inv_after_unsupported", "inv"),
        ("pinv", "pinv_after_unsupported", "pinv"),
        ("norm", "norm_after_unsupported", "norm"),
        ("rcond", "rcond_after_unsupported", "rcond"),
    ] {
        let provider = Box::leak(Box::new(Provider {
            owner: TestAllocationOwner::new(),
            failure,
            trace: Mutex::new(Vec::new()),
            probes: Mutex::new(Vec::new()),
        }));
        let _selected = ThreadProviderGuard::set(Some(provider));
        let source = provider
            .owner
            .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let arguments = [Value::GpuTensor(source.handle.clone())];
        let error =
            futures::executor::block_on(runmat_runtime::call_builtin_async(name, &arguments))
                .expect_err("after-effect Unsupported is terminal");
        assert_eq!(
            error.gpu_gather_retry,
            runmat_runtime::GpuGatherRetry::Never
        );
        assert_eq!(*provider.trace.lock().unwrap(), [hook]);
        let mut cause: &(dyn std::error::Error + 'static) = &error;
        let mut retained_phase = false;
        while let Some(next) = cause.source() {
            retained_phase |= next
                .downcast_ref::<ProviderExecutionError>()
                .is_some_and(ProviderExecutionError::is_after_effect);
            cause = next;
        }
        assert!(retained_phase, "{name}: phase-bearing source was lost");
        source.handle.allocation_lease().validate_live().unwrap();
    }
}
