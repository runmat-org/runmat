#[cfg(feature = "wgpu")]
use super::*;
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn power_wgpu_matches_cpu_elementwise() {
    let _guard = test_support::accel_test_lock();
    let Some(provider) = test_support::wgpu_provider_if_available() else {
        return;
    };
    let base = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    let exp = Tensor::new(vec![2.0, 0.5, -1.0], vec![1, 3]).unwrap();
    let cpu = power_host(Value::Tensor(base.clone()), Value::Tensor(exp.clone())).unwrap();
    let hb = gpu_helpers::upload_tensor(provider, &base).unwrap();
    let he = gpu_helpers::upload_tensor(provider, &exp).unwrap();
    let base_lease = hb.allocation_lease().downgrade();
    let exponent_lease = he.allocation_lease().downgrade();
    let gpu = block_on(power_gpu_pair(None, hb.clone(), he.clone())).unwrap();
    let gathered = test_support::gather(gpu).expect("gather");
    let cpu_tensor = match cpu {
        Value::Tensor(t) => t,
        Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).unwrap(),
        other => panic!("unexpected cpu result {other:?}"),
    };
    let tol = match provider.precision() {
        runmat_accelerate_api::ProviderPrecision::F64 => 1e-9,
        runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
    };
    assert_eq!(gathered.shape(), cpu_tensor.shape());
    assert_eq!(gathered.len(), cpu_tensor.len());
    for (a, b) in gathered
        .as_f64_slice()
        .expect("gathered double")
        .iter()
        .zip(cpu_tensor.as_f64_slice().expect("CPU double"))
    {
        assert!((a - b).abs() < tol);
    }
    hb.release().expect("release base");
    he.release().expect("release exponent");
    assert!(base_lease.acquire().is_none());
    assert!(exponent_lease.acquire().is_none());
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
#[test]
#[ignore = "requires an actual shader-f64 WGPU adapter; run in the F64 qualification gate"]
fn ordinary_native_single_power_uses_f64_owner_without_reclassifying_control() {
    let _guard = test_support::accel_test_lock();
    let provider = test_support::wgpu_provider_if_available()
        .expect("F64 qualification gate requires an actual WGPU adapter");
    assert_eq!(
        provider.precision(),
        runmat_accelerate_api::ProviderPrecision::F64
    );
    let base = Tensor::from_f32(vec![1.25; 40], vec![2, 4, 5]).unwrap();
    let control = Tensor::from_f32(vec![1.8], vec![1, 1]).unwrap();
    let resident = gpu_helpers::upload_tensor(provider, &base).unwrap();
    let output = power_builtin(
        Value::GpuTensor(resident.clone()),
        Value::Tensor(control),
        Vec::new(),
    )
    .expect("ordinary NativeSingle POWER");
    let Value::GpuTensor(output) = output else {
        panic!("NativeSingle POWER must remain resident on the F64 owner")
    };
    assert_eq!(output.shape(), [2, 4, 5]);
    assert_eq!(
        runmat_accelerate_api::handle_precision(&output),
        Some(runmat_accelerate_api::ProviderPrecision::F32)
    );
    assert!(!output.shares_allocation_with(&resident));
    resident.release().unwrap();
    assert!(output.allocation_lease().validate_live().is_ok());
    let gathered = test_support::gather(Value::GpuTensor(output)).unwrap();
    assert_eq!(gathered.shape(), [2, 4, 5]);
    assert!(gathered.as_f32_slice().is_some());
    let expected = 1.25_f32.powf(1.8_f32);
    assert!(gathered
        .as_f32_slice()
        .unwrap()
        .iter()
        .all(|value| (*value - expected).abs() < 2e-5));
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
#[test]
#[ignore = "requires an actual shader-f64 WGPU adapter; run in the F64 qualification gate"]
fn ordinary_native_single_controls_keep_original_bits_and_domain_on_f64_owner() {
    let _guard = test_support::accel_test_lock();
    let provider = test_support::wgpu_provider_if_available()
        .expect("F64 qualification gate requires an actual WGPU adapter");
    assert_eq!(
        provider.precision(),
        runmat_accelerate_api::ProviderPrecision::F64
    );
    let real_cases = [
        (1000.0_f32, f32::from_bits(1.0_f32.to_bits() + 1)),
        (-0.0_f32, 1.0_f32),
        (2.0_f32, f32::NAN),
    ];
    for (base, exponent) in real_cases {
        let resident = gpu_helpers::upload_tensor(
            provider,
            &Tensor::from_f32(vec![base], vec![1, 1]).unwrap(),
        )
        .unwrap();
        let output = power_builtin(
            Value::GpuTensor(resident.clone()),
            Value::Tensor(Tensor::from_f32(vec![exponent], vec![1, 1]).unwrap()),
            Vec::new(),
        )
        .expect("ordinary Single control must remain resident");
        let Value::GpuTensor(output) = output else {
            panic!("Single control must retain provider residency")
        };
        assert_eq!(
            runmat_accelerate_api::handle_precision(&output),
            Some(runmat_accelerate_api::ProviderPrecision::F32)
        );
        let actual = test_support::gather(Value::GpuTensor(output)).unwrap();
        let actual = actual.as_f32_slice().expect("Single result")[0];
        if exponent.is_nan() {
            assert!(actual.is_nan());
        } else if base == 0.0 {
            assert_eq!(actual.to_bits(), (-0.0_f32).to_bits());
        } else {
            let exact_control = base.powf(exponent);
            assert!((actual - exact_control).abs() < 0.01);
            assert_ne!(actual.to_bits(), base.to_bits());
        }
        resident.release().unwrap();
    }

    let resident =
        gpu_helpers::upload_tensor(provider, &Tensor::from_f32(vec![-4.0], vec![1, 1]).unwrap())
            .unwrap();
    let output = power_builtin(
        Value::GpuTensor(resident.clone()),
        Value::Tensor(Tensor::from_f32(vec![0.5], vec![1, 1]).unwrap()),
        Vec::new(),
    )
    .expect("negative base fractional Single POWER");
    let Value::GpuTensor(output) = output else {
        panic!("complex Single output must remain resident")
    };
    assert_eq!(
        runmat_accelerate_api::handle_storage(&output),
        runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
    );
    let gathered = block_on(crate::dispatcher::gather_if_needed_async(
        &Value::GpuTensor(output),
    ))
    .unwrap();
    let Value::ComplexTensor(gathered) = gathered else {
        panic!("negative base fractional power must publish complex Single")
    };
    assert!(gathered.as_f32_slice().is_some());
    let materialized = gathered.materialize_f64();
    let [(real, imag)] = materialized.as_slice() else {
        panic!("single complex scalar")
    };
    assert!(real.abs() < 2e-5 && (*imag - 2.0).abs() < 2e-5);
    resident.release().unwrap();
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
mod mixed;
