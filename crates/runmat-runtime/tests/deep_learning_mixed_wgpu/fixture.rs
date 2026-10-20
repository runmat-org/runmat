use futures::executor::block_on;
use runmat_accelerate::backend::wgpu::provider::WgpuProvider;
use runmat_accelerate_api::{
    handle_precision, handle_provenance, handle_storage, AccelProvider, GpuHandleProvenance,
    GpuTensorHandle, GpuTensorStorage, HostNumericDataOwned, HostNumericDataView,
    HostNumericTensorView, ProviderPrecision,
};
use runmat_runtime::call_builtin;
use runmat_runtime::execution::RootedValues;
use runmat_value::{CellArray, Value};

pub(super) fn device() -> &'static WgpuProvider {
    let provider = Box::leak(Box::new(
        WgpuProvider::new(Default::default()).expect("actual WGPU device required"),
    ));
    if let Ok(requested) = std::env::var("RUNMAT_WGPU_FORCE_PRECISION") {
        let expected = match requested.trim().to_ascii_lowercase().as_str() {
            "f32" | "float32" | "32" => ProviderPrecision::F32,
            "f64" | "float64" | "64" => ProviderPrecision::F64,
            other => panic!("unsupported hardware-test precision: {other}"),
        };
        assert_eq!(
            provider.precision(),
            expected,
            "hardware qualification must not pass by substituting another precision"
        );
    }
    provider
}

pub(super) fn upload(
    provider: &WgpuProvider,
    values: &[f64],
    shape: &[usize],
    provenance: GpuHandleProvenance,
) -> GpuTensorHandle {
    let single = values.iter().map(|value| *value as f32).collect::<Vec<_>>();
    provider
        .upload_numeric(&HostNumericTensorView {
            data: match provider.precision() {
                ProviderPrecision::F32 => HostNumericDataView::F32(&single),
                ProviderPrecision::F64 => HostNumericDataView::F64(values),
            },
            shape,
            storage: GpuTensorStorage::Real,
        })
        .unwrap()
        .with_provenance(provenance)
        .unwrap()
}

pub(super) fn read(provider: &WgpuProvider, handle: &GpuTensorHandle) -> Vec<f64> {
    match (
        provider.precision(),
        block_on(provider.download_numeric(handle)).unwrap().data,
    ) {
        (ProviderPrecision::F32, HostNumericDataOwned::F32(values)) => {
            values.into_iter().map(f64::from).collect()
        }
        (ProviderPrecision::F64, HostNumericDataOwned::F64(values)) => values,
        _ => panic!("provider changed the physical numerical class"),
    }
}

pub(super) fn forward(
    provider: &WgpuProvider,
    input: &GpuTensorHandle,
    input_format: &str,
    output_format: &str,
    layers: RootedValues,
    expected_shape: &[usize],
    expected_provenance: GpuHandleProvenance,
) -> GpuTensorHandle {
    let input_value = call_builtin(
        "dlarray",
        &[
            Value::GpuTensor(input.clone()),
            Value::String(input_format.into()),
        ],
    )
    .unwrap();
    let count = layers.values().len();
    let network = layers
        .with_unbounded_values(|layers| {
            call_builtin(
                "dlnetwork",
                &[Value::Cell(CellArray::new(layers, count, 1).unwrap())],
            )
            .unwrap()
        })
        .expect("fixture layer batch is unbounded");
    provider.reset_telemetry();
    let output = network.with_owned(|network| {
        input_value
            .with_owned(|input_value| call_builtin("forward", &[network, input_value]).unwrap())
    });
    let Value::Object(output) = output.value() else {
        panic!("formatted forward output must be a dlarray")
    };
    assert_eq!(
        output.properties().get("Format"),
        Some(&Value::String(output_format.into()))
    );
    assert_eq!(
        output.properties().get("Labels"),
        Some(&Value::String(output_format.into()))
    );
    let Some(Value::GpuTensor(data)) = output.properties().get("Data") else {
        panic!("supported resident route must not gather")
    };
    assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    assert_eq!(data.shape(), expected_shape);
    assert_eq!(data.device_id(), input.device_id());
    assert_eq!(handle_precision(data), Some(provider.precision()));
    assert_eq!(handle_storage(data), GpuTensorStorage::Real);
    assert_eq!(handle_provenance(data), Some(expected_provenance));
    data.clone()
}

pub(super) fn assert_values(provider: &WgpuProvider, handle: &GpuTensorHandle, expected: &[f64]) {
    let actual = read(provider, handle);
    assert_eq!(actual.len(), expected.len());
    let tolerance = match provider.precision() {
        ProviderPrecision::F32 => 3.0e-4,
        ProviderPrecision::F64 => 1.0e-9,
    };
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "element {index}: {actual} differs from independent oracle {expected}"
        );
    }
}
