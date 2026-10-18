//! Actual-WGPU registered std/var witnesses; the native test requires an adapter.
use crate::builtins::common::gpu_helpers;
use runmat_accelerate_api::{AccelProvider, ProviderPrecision, ThreadProviderGuard};
use runmat_value::{NumericDType, Tensor, Value};

async fn public_value(name: &str, arguments: &[Value]) -> crate::execution::RootedValue {
    let outputs = crate::call_builtin_sequence_async_with_outputs(name, arguments, 1)
        .await
        .expect("registered dispersion call");
    outputs
        .try_map_owned(
            |outputs| {
                let value = outputs
                    .into_output_storage()
                    .try_into_first()
                    .expect("unique dispersion output")
                    .expect("one dispersion output");
                Ok::<Value, crate::RuntimeError>(value)
            },
            |error| error.into_runtime_error(),
        )
        .expect("root dispersion output")
}

fn release_resident_output(value: crate::execution::RootedValue) {
    value.with_owned(|value| {
        let Value::GpuTensor(output) = value else {
            panic!("expected the inspected resident dispersion output");
        };
        output.release().expect("successful WGPU output release");
    });
}

async fn compare_host_and_resident(provider: &'static dyn AccelProvider) {
    assert_eq!(
        provider.precision(),
        ProviderPrecision::F32,
        "Single WGPU owner"
    );
    let _selected = ThreadProviderGuard::set(Some(provider));
    for name in ["std", "var"] {
        // Retain the numerical inputs of the former direct std/var WGPU tests.
        let data: [f32; 4] = if name == "std" {
            [1.0, 4.0, 2.0, 6.0]
        } else {
            [1.0, 4.0, 2.0, 5.0]
        };
        let tensor = Tensor::from_f32(data.to_vec(), vec![2, 2]).expect("Single input");
        let input = gpu_helpers::upload_tensor(provider, &tensor).expect("Single WGPU upload");
        assert_eq!(
            runmat_accelerate_api::handle_precision(&input),
            Some(ProviderPrecision::F32)
        );
        for controls in [
            vec![],
            vec![Value::Num(0.0), Value::Num(2.0)],
            vec![Value::Num(1.0), Value::Num(2.0)],
            vec![Value::Num(0.0), Value::from("all")],
            vec![Value::Num(1.0), Value::from("all")],
        ] {
            let host_arguments: Vec<_> = std::iter::once(Value::Tensor(tensor.clone()))
                .chain(controls.iter().cloned())
                .collect();
            let expected = public_value(name, &host_arguments).await;
            let Value::Tensor(expected) = expected.value() else {
                panic!("{name}: Single host result must remain a tensor");
            };
            assert_eq!(expected.numeric_dtype(), NumericDType::F32);

            let gpu_arguments: Vec<_> = std::iter::once(Value::GpuTensor(input.clone()))
                .chain(controls.iter().cloned())
                .collect();
            let actual = public_value(name, &gpu_arguments).await;
            let Value::GpuTensor(output) = actual.value() else {
                panic!("{name}: supported Single reduction must remain resident");
            };
            assert_eq!(output.shape(), expected.shape(), "{name} shape");
            assert_eq!(
                runmat_accelerate_api::handle_precision(output),
                Some(ProviderPrecision::F32)
            );
            assert!(
                !output.shares_allocation_with(&input),
                "{name} output allocation"
            );
            assert!(
                gpu_helpers::exact_provider_for_handle(output).is_some_and(|owner| {
                    runmat_accelerate_api::same_provider_instance(owner, provider)
                }),
                "{name} output owner"
            );
            let gathered = gpu_helpers::gather_tensor_async(output)
                .await
                .expect("owning WGPU readback");
            assert_eq!(gathered.numeric_dtype(), NumericDType::F32, "{name} class");
            assert_eq!(gathered.shape(), expected.shape(), "{name} gathered shape");
            let gpu_values = gathered.materialize_f64();
            let host_values = expected.materialize_f64();
            assert_eq!(gpu_values.len(), host_values.len(), "{name} value count");
            for (gpu, host) in gpu_values.iter().zip(host_values) {
                assert!(
                    (*gpu - host).abs() < 1e-4,
                    "{name} numerical mismatch: gpu={gpu} host={host}"
                );
            }
            let weak_output = output.downgrade();
            release_resident_output(actual);
            assert!(weak_output.acquire().is_err(), "{name} output view retired");
            input
                .allocation_lease()
                .validate_live()
                .expect("source survives output drop");
        }
        let retained = gpu_helpers::gather_tensor_async(&input)
            .await
            .expect("source readback after reductions");
        assert_eq!(retained.numeric_dtype(), NumericDType::F32);
        assert_eq!(retained.shape(), tensor.shape());
        let retained_bits: Vec<_> = retained
            .as_f32_slice()
            .expect("Single source payload")
            .iter()
            .map(|value| value.to_bits())
            .collect();
        assert_eq!(retained_bits, data.map(f32::to_bits).to_vec());
        input
            .allocation_lease()
            .validate_live()
            .expect("source remains live");
        let weak_input = input.downgrade();
        input.release().expect("successful WGPU source release");
        assert!(weak_input.acquire().is_err(), "{name} source view retired");
    }
}

#[cfg(not(target_arch = "wasm32"))]
async fn compare_required_f64_dim_one(provider: &'static dyn AccelProvider) {
    let _selected = ThreadProviderGuard::set(Some(provider));
    for name in ["std", "var"] {
        let data = if name == "std" {
            vec![1.0, 4.0, 2.0, 6.0]
        } else {
            vec![1.0, 4.0, 2.0, 5.0]
        };
        let tensor = Tensor::new(data, vec![2, 2]).expect("Double input");
        let input = gpu_helpers::upload_tensor(provider, &tensor).expect("Double WGPU upload");
        assert_eq!(
            runmat_accelerate_api::handle_precision(&input),
            Some(ProviderPrecision::F64)
        );
        let controls = [Value::Num(0.0), Value::Num(1.0)];
        let host = public_value(
            name,
            &[
                Value::Tensor(tensor),
                controls[0].clone(),
                controls[1].clone(),
            ],
        )
        .await;
        let Value::Tensor(expected) = host.value() else {
            panic!("{name}: expected Double dimensional host tensor");
        };
        let actual = public_value(
            name,
            &[
                Value::GpuTensor(input.clone()),
                controls[0].clone(),
                controls[1].clone(),
            ],
        )
        .await;
        let Value::GpuTensor(output) = actual.value() else {
            panic!("{name}: required F64 dimensional result must remain resident");
        };
        assert_eq!(output.shape(), expected.shape());
        assert_eq!(
            runmat_accelerate_api::handle_precision(output),
            Some(ProviderPrecision::F64)
        );
        assert!(
            gpu_helpers::exact_provider_for_handle(output).is_some_and(|owner| {
                runmat_accelerate_api::same_provider_instance(owner, provider)
            }),
            "{name} Double output owner"
        );
        let gathered = gpu_helpers::gather_tensor_async(output)
            .await
            .expect("owning Double readback");
        assert_eq!(gathered.numeric_dtype(), NumericDType::F64);
        assert_eq!(gathered.shape(), expected.shape());
        let gpu_values = gathered.materialize_f64();
        let host_values = expected.materialize_f64();
        assert_eq!(gpu_values.len(), host_values.len());
        for (gpu, host) in gpu_values.iter().zip(host_values) {
            assert!((*gpu - host).abs() < 1e-12, "{name} Double mismatch");
        }
        let weak_output = output.downgrade();
        release_resident_output(actual);
        assert!(
            weak_output.acquire().is_err(),
            "{name} Double output retired"
        );
        input
            .allocation_lease()
            .validate_live()
            .expect("Double source live");
        input.release().expect("successful Double source release");
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires selected F32 WGPU adapter; run explicitly as required hardware qualification"]
fn actual_wgpu_registered_single_std_var() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider =
        runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(Default::default())
            .expect("required std/var WGPU qualification needs a live adapter");
    if let Ok(requested) = std::env::var("RUNMAT_WGPU_FORCE_PRECISION") {
        let actual = match provider.precision() {
            ProviderPrecision::F32 => "f32",
            ProviderPrecision::F64 => "f64",
        };
        assert_eq!(requested.to_ascii_lowercase(), actual);
    }
    futures::executor::block_on(compare_host_and_resident(provider));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires RUNMAT_WGPU_FORCE_PRECISION=f64 and actual enabled SHADER_F64"]
fn actual_wgpu_registered_double_std_var_dim_one() {
    assert_eq!(
        std::env::var("RUNMAT_WGPU_FORCE_PRECISION").as_deref(),
        Ok("f64")
    );
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider =
        runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(Default::default())
            .expect("required F64 WGPU qualification needs a live adapter");
    assert_eq!(provider.precision(), ProviderPrecision::F64);
    let context = provider
        .export_context(runmat_accelerate_api::AccelContextKind::Plotting)
        .expect("selected WGPU context");
    let wgpu = context.as_wgpu().expect("WGPU context");
    assert!(wgpu.features.contains(wgpu::Features::SHADER_F64));
    futures::executor::block_on(compare_required_f64_dim_one(provider));
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn actual_browser_wgpu_registered_single_std_var() {
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider_async(
        Default::default(),
    )
    .await
    .expect("required browser WGPU qualification needs a live adapter");
    compare_host_and_resident(provider).await;
}
