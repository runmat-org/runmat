use super::*;

mod complex;
mod mixed_add;

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires an actual WGPU adapter with native integer transfer"]
fn plus_explicit_integer_public_route_exercises_host_fallback() {
    use runmat_accelerate_api::AccelProvider;

    let _guard = test_support::accel_test_lock();
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("required WGPU adapter");
    use runmat_accelerate_api::IntegerElementType as Class;

    // Keep the original I32 oracle and exercise every native integer class.
    // Wide cases stay above 2^53 without overflowing, so an f64 round trip
    // cannot accidentally satisfy their exact-value assertion.
    let cases = [
        (
            IntegerStorage::I8(vec![-128, 126]),
            IntegerStorage::I8(vec![1]),
            IntegerStorage::I8(vec![-127, 127]),
            Class::I8,
        ),
        (
            IntegerStorage::I16(vec![-32768, 32766]),
            IntegerStorage::I16(vec![1]),
            IntegerStorage::I16(vec![-32767, 32767]),
            Class::I16,
        ),
        (
            IntegerStorage::I32(vec![2, 4]),
            IntegerStorage::I32(vec![2]),
            IntegerStorage::I32(vec![4, 6]),
            Class::I32,
        ),
        (
            IntegerStorage::I64(vec![i64::MIN, i64::MIN + 2]),
            IntegerStorage::I64(vec![1]),
            IntegerStorage::I64(vec![i64::MIN + 1, i64::MIN + 3]),
            Class::I64,
        ),
        (
            IntegerStorage::U8(vec![0, 254]),
            IntegerStorage::U8(vec![1]),
            IntegerStorage::U8(vec![1, 255]),
            Class::U8,
        ),
        (
            IntegerStorage::U16(vec![0, 65534]),
            IntegerStorage::U16(vec![1]),
            IntegerStorage::U16(vec![1, 65535]),
            Class::U16,
        ),
        (
            IntegerStorage::U32(vec![0, u32::MAX - 1]),
            IntegerStorage::U32(vec![1]),
            IntegerStorage::U32(vec![1, u32::MAX]),
            Class::U32,
        ),
        (
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX - 2]),
            IntegerStorage::U64(vec![1]),
            IntegerStorage::U64(vec![9_007_199_254_740_994, u64::MAX - 1]),
            Class::U64,
        ),
    ];
    for (input, scalar, expected, class) in cases {
        let input = Tensor::new_integer(input, vec![2, 1]).unwrap();
        let source = gpu_helpers::upload_tensor(provider, &input)
            .unwrap()
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .unwrap();
        let scalar = Tensor::new_integer(scalar, vec![1, 1]).unwrap();
        let before = provider.telemetry_snapshot().download_bytes;
        // Same-class host integer Tensor takes PLUS's pre-effect fallback
        // branch. A Num scalar could instead use native provider arithmetic.
        let Value::GpuTensor(output) = plus_builtin(
            Value::GpuTensor(source.clone()),
            Value::Tensor(scalar),
            Vec::new(),
        )
        .unwrap() else {
            panic!("public PLUS must restore an explicit resident result")
        };
        assert!(
            provider.telemetry_snapshot().download_bytes > before,
            "{class:?} source must be gathered for host evaluation"
        );
        assert!(runmat_accelerate_api::handle_is_explicit(&output));
        assert_eq!(output.shape(), &[2, 1]);
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&output),
            Some(class)
        );
        assert!(!output.shares_allocation_with(&source));
        let gathered = test_support::gather(Value::GpuTensor(output)).unwrap();
        assert_eq!(
            gathered.integer_storage(),
            Some(&expected),
            "{class:?} exact value"
        );
        source.allocation_lease().validate_live().unwrap();
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn plus_wgpu_matches_cpu_elementwise() {
    let _guard = test_support::accel_test_lock();
    if !register_wgpu_provider_available() {
        return;
    }
    let lhs = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let rhs = Tensor::new(vec![4.0, 3.0, 2.0, 1.0], vec![2, 2]).unwrap();
    let cpu = plus_host(Value::Tensor(lhs.clone()), Value::Tensor(rhs.clone())).unwrap();
    let provider = runmat_accelerate_api::provider().unwrap();
    let ha = gpu_helpers::upload_tensor(provider, &lhs).unwrap();
    let hb = gpu_helpers::upload_tensor(provider, &rhs).unwrap();
    let gpu = block_on(plus_gpu_pair(None, ha, hb)).unwrap();
    let gathered = test_support::gather(gpu).expect("gather");
    match cpu {
        Value::Tensor(t) => assert_eq!(
            gathered.as_f64_slice().expect("double GPU output"),
            t.as_f64_slice().expect("double CPU output")
        ),
        Value::Num(n) => {
            assert_eq!(gathered.as_f64_slice().expect("double output"), &[n])
        }
        other => panic!("unexpected cpu result {other:?}"),
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires an actual F32 WGPU adapter"]
fn plus_wgpu_mixed_single_resident_double_scalar_preserves_both_tie_sides_and_orders() {
    use runmat_accelerate_api::AccelProvider;

    let _guard = test_support::accel_test_lock();
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("required actual WGPU adapter");
    assert_eq!(
        provider.precision(),
        runmat_accelerate_api::ProviderPrecision::F32
    );
    let below_one = f32::from_bits(0x3f7f_ffff);
    let input = Tensor::from_f32(vec![below_one; 2], vec![1, 2]).unwrap();
    let resident = gpu_helpers::upload_tensor(provider, &input)
        .unwrap()
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        .unwrap();
    assert!(runmat_accelerate_api::handle_is_explicit(&resident));
    let half_ulp = f64::from(f32::EPSILON) / 4.0;
    for scalar in [half_ulp - f64::EPSILON, half_ulp + f64::EPSILON] {
        for reverse in [false, true] {
            let operands = if reverse {
                [Value::Num(scalar), Value::GpuTensor(resident.clone())]
            } else {
                [Value::GpuTensor(resident.clone()), Value::Num(scalar)]
            };
            let [left, right] = operands;
            let before = provider.telemetry_snapshot();
            let error = plus_builtin(left, right, Vec::new())
                .expect_err("explicit resident arithmetic cannot silently run on host");
            assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
            assert_eq!(
                error.identifier(),
                Some("RunMat:gpu:ExplicitProviderRequired")
            );
            assert_eq!(
                provider.telemetry_snapshot().download_bytes,
                before.download_bytes
            );
            resident.allocation_lease().validate_live().unwrap();
        }
    }
}
