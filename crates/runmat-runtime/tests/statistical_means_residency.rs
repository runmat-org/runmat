use futures::executor::block_on;
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{
    AccelProvider, GpuHandleProvenance, GpuTensorHandle, GpuTensorStorage, HostNumericDataOwned,
    HostNumericDataView, HostNumericTensorView, ThreadProviderGuard,
};
use runmat_value::Value;

fn upload(
    provider: &'static InProcessProvider,
    values: &[f64],
    shape: &[usize],
) -> GpuTensorHandle {
    runmat_accelerate_api::upload_numeric_protecting(
        provider,
        &HostNumericTensorView {
            data: HostNumericDataView::F64(values),
            storage: GpuTensorStorage::Real,
            shape,
        },
        &[],
    )
    .unwrap()
    .with_provenance(GpuHandleProvenance::Explicit)
    .unwrap()
}

fn upload_single(
    provider: &'static InProcessProvider,
    values: &[f32],
    shape: &[usize],
) -> GpuTensorHandle {
    runmat_accelerate_api::upload_numeric_protecting(
        provider,
        &HostNumericTensorView {
            data: HostNumericDataView::F32(values),
            storage: GpuTensorStorage::Real,
            shape,
        },
        &[],
    )
    .unwrap()
    .with_provenance(GpuHandleProvenance::Explicit)
    .unwrap()
}

fn verify_mean_residency(name: &str, paired: bool, single: bool, omitnan: bool, expected: f64) {
    let provider = Box::leak(Box::new(InProcessProvider::new()));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let (input, other) = match (single, omitnan) {
        (true, true) => (
            upload_single(provider, &[1.0, f32::NAN, 4.0], &[3, 1]),
            upload_single(provider, &[2.0, 0.0, 2.0], &[3, 1]),
        ),
        (false, true) => (
            upload(provider, &[1.0, f64::NAN, 4.0], &[3, 1]),
            upload(provider, &[2.0, 0.0, 2.0], &[3, 1]),
        ),
        (true, false) => (
            upload_single(provider, &[1.0, 4.0], &[2, 1]),
            upload_single(provider, &[2.0, 2.0], &[2, 1]),
        ),
        (false, false) => (
            upload(provider, &[1.0, 4.0], &[2, 1]),
            upload(provider, &[2.0, 2.0], &[2, 1]),
        ),
    };
    let mut arguments = vec![Value::GpuTensor(input.clone())];
    if paired {
        arguments.push(Value::GpuTensor(other));
    }
    if omitnan {
        arguments.push(Value::from("omitnan"));
    }
    let result = block_on(runmat_runtime::call_builtin_async(name, &arguments))
        .unwrap_or_else(|error| panic!("{name} failed: {error}"));
    let Value::GpuTensor(result) = result.value() else {
        panic!("{name} should preserve same-owner floating input residency");
    };
    assert_eq!(result.shape(), &[1, 1], "{name}");
    assert_eq!(result.device_id(), input.device_id(), "{name}");
    assert!(!result.shares_allocation_with(&input), "{name}");
    assert!(input.allocation_lease().validate_live().is_ok());
    assert!(result.allocation_lease().validate_live().is_ok());
    let owner = runmat_accelerate_api::provider_for_handle(result)
        .expect("result must retain its exact provider owner");
    assert!(runmat_accelerate_api::same_provider_instance(
        owner, provider
    ));
    let host = block_on(provider.download_numeric(result)).unwrap();
    let actual = match host.data {
        HostNumericDataOwned::F32(values) if single => f64::from(values[0]),
        HostNumericDataOwned::F64(values) if !single => values[0],
        other => panic!("{name} changed output precision: {other:?}"),
    };
    assert!(
        (actual - expected).abs() < if single { 1e-5 } else { 1e-12 },
        "{name}"
    );
}

macro_rules! resident_mean_case {
    ($test:ident, $name:literal, $paired:literal, $single:literal, $omitnan:literal, $expected:expr) => {
        #[test]
        fn $test() {
            verify_mean_residency($name, $paired, $single, $omitnan, $expected);
        }
    };
}

resident_mean_case!(geomean_double_resident, "geomean", false, false, false, 2.0);
resident_mean_case!(
    harmmean_double_resident,
    "harmmean",
    false,
    false,
    false,
    1.6
);
resident_mean_case!(
    rms_double_resident,
    "rms",
    false,
    false,
    false,
    8.5_f64.sqrt()
);
resident_mean_case!(
    rmse_double_resident,
    "rmse",
    true,
    false,
    false,
    2.5_f64.sqrt()
);
resident_mean_case!(geomean_single_resident, "geomean", false, true, false, 2.0);
resident_mean_case!(
    harmmean_single_resident,
    "harmmean",
    false,
    true,
    false,
    1.6
);
resident_mean_case!(
    rms_single_resident,
    "rms",
    false,
    true,
    false,
    8.5_f64.sqrt()
);
resident_mean_case!(
    rmse_single_resident,
    "rmse",
    true,
    true,
    false,
    2.5_f64.sqrt()
);
resident_mean_case!(
    geomean_double_omitnan_resident,
    "geomean",
    false,
    false,
    true,
    2.0
);
resident_mean_case!(
    harmmean_double_omitnan_resident,
    "harmmean",
    false,
    false,
    true,
    1.6
);
resident_mean_case!(
    rms_double_omitnan_resident,
    "rms",
    false,
    false,
    true,
    8.5_f64.sqrt()
);
resident_mean_case!(
    rmse_double_omitnan_resident,
    "rmse",
    true,
    false,
    true,
    2.5_f64.sqrt()
);
resident_mean_case!(
    geomean_single_omitnan_resident,
    "geomean",
    false,
    true,
    true,
    2.0
);
resident_mean_case!(
    harmmean_single_omitnan_resident,
    "harmmean",
    false,
    true,
    true,
    1.6
);
resident_mean_case!(
    rms_single_omitnan_resident,
    "rms",
    false,
    true,
    true,
    8.5_f64.sqrt()
);
resident_mean_case!(
    rmse_single_omitnan_resident,
    "rmse",
    true,
    true,
    true,
    2.5_f64.sqrt()
);

#[test]
fn geometric_negative_domain_is_local_to_each_resident_slice() {
    let provider = Box::leak(Box::new(InProcessProvider::new()));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = upload(provider, &[1.0, 4.0, -4.0, 9.0], &[2, 2]);
    let result = block_on(runmat_runtime::call_builtin_async(
        "geomean",
        &[Value::GpuTensor(input.clone())],
    ))
    .unwrap();
    let Value::GpuTensor(result) = result.value() else {
        panic!("supported real geomean should remain resident");
    };
    assert_eq!(result.shape(), &[1, 2]);
    assert!(!result.shares_allocation_with(&input));
    assert!(input.allocation_lease().validate_live().is_ok());
    let host = block_on(provider.download_numeric(result)).unwrap();
    let HostNumericDataOwned::F64(values) = host.data else {
        panic!("geomean must preserve double precision");
    };
    assert_eq!(values.len(), 2);
    assert!((values[0] - 2.0).abs() < 1e-12);
    assert!(values[1].is_nan());
}
