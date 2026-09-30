//! Public forward witness for a composed resident normalization/dense route.
#![cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]

use futures::executor::block_on;
use runmat_accelerate::backend::wgpu::provider::WgpuProvider;
use runmat_accelerate_api::{
    handle_precision, handle_provenance, handle_storage, AccelProvider, GpuHandleProvenance,
    GpuTensorStorage, HostNumericDataOwned, HostNumericDataView, HostNumericTensorView,
    ProviderPrecision, ThreadProviderGuard,
};
use runmat_runtime::call_builtin;
use runmat_value::{CellArray, Tensor, Value};

#[test]
#[ignore = "requires an actual WGPU device; run explicitly for both native precisions"]
fn public_nonsquare_cb_normalization_then_dense_stays_resident_and_matches_oracle() {
    let provider = Box::leak(Box::new(
        WgpuProvider::new(Default::default()).expect("actual WGPU device required"),
    ));
    let _selected = ThreadProviderGuard::set(Some(provider));
    for lazy_source in [false, true] {
        let source = [1.0, 3.0, 2.0, 5.0, 5.0, 7.0];
        let backing = if lazy_source {
            [1.0, 2.0, 5.0, 3.0, 5.0, 7.0]
        } else {
            source
        };
        let backing_shape: &[usize] = if lazy_source { &[3, 2] } else { &[2, 3] };
        let single = backing.map(|value| value as f32);
        let uploaded = provider
            .upload_numeric(&HostNumericTensorView {
                data: match provider.precision() {
                    ProviderPrecision::F32 => HostNumericDataView::F32(&single),
                    ProviderPrecision::F64 => HostNumericDataView::F64(&backing),
                },
                shape: backing_shape,
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        let handle = (if lazy_source {
            provider.transpose(&uploaded)
        } else {
            Ok(uploaded)
        })
        .unwrap()
        .with_provenance(GpuHandleProvenance::Explicit)
        .unwrap();
        let retained_input = handle.clone();
        let input = call_builtin(
            "dlarray",
            &[Value::GpuTensor(handle), Value::String("CB".into())],
        )
        .unwrap();
        let layers = [
            call_builtin("featureInputLayer", &[Value::Num(2.0)]).unwrap(),
            call_builtin("layerNormalizationLayer", &[]).unwrap(),
            call_builtin(
                "fullyConnectedLayer",
                &[
                    Value::Num(1.0),
                    Value::String("Weights".into()),
                    Value::Tensor(Tensor::new(vec![2.0, -1.0], vec![1, 2]).unwrap()),
                    Value::String("Bias".into()),
                    Value::Tensor(Tensor::new(vec![0.5], vec![1, 1]).unwrap()),
                ],
            )
            .unwrap(),
        ];
        let network = call_builtin(
            "dlnetwork",
            &[Value::Cell(CellArray::new(layers.to_vec(), 3, 1).unwrap())],
        )
        .unwrap();
        provider.reset_telemetry();
        let output = call_builtin("forward", &[network, input]).unwrap();
        let Value::Object(output) = output else {
            panic!("formatted dlarray output expected")
        };
        let Some(Value::GpuTensor(data)) = output.properties.get("Data") else {
            panic!("explicit input must retain resident output")
        };
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        assert_eq!(data.device_id(), retained_input.device_id());
        assert_eq!(handle_precision(data), Some(provider.precision()));
        assert_eq!(handle_storage(data), GpuTensorStorage::Real);
        assert_eq!(handle_provenance(data), Some(GpuHandleProvenance::Explicit));
        assert_eq!(data.shape(), [1, 3]);
        assert_eq!(
            output.properties.get("Format"),
            Some(&Value::String("CB".into()))
        );
        let gathered = block_on(provider.download_numeric(data)).unwrap();
        let actual = match (provider.precision(), gathered.data) {
            (ProviderPrecision::F32, HostNumericDataOwned::F32(values)) => {
                values.into_iter().map(f64::from).collect::<Vec<_>>()
            }
            (ProviderPrecision::F64, HostNumericDataOwned::F64(values)) => values,
            _ => panic!("resident route changed numeric class"),
        };
        let expected = [1.0, 1.5, 1.0].map(|difference| {
            let centered = difference / (difference * difference + 1.0e-5).sqrt();
            0.5 - 3.0 * centered
        });
        assert_eq!(actual.len(), expected.len());
        let tolerance = if provider.precision() == ProviderPrecision::F32 {
            2.0e-4
        } else {
            1.0e-9
        };
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < tolerance,
                "{actual} vs {expected}"
            );
        }
        let retained = block_on(provider.download_numeric(&retained_input)).unwrap();
        let retained = match (provider.precision(), retained.data) {
            (ProviderPrecision::F32, HostNumericDataOwned::F32(values)) => {
                values.into_iter().map(f64::from).collect::<Vec<_>>()
            }
            (ProviderPrecision::F64, HostNumericDataOwned::F64(values)) => values,
            _ => panic!("retained input changed numeric class"),
        };
        assert_eq!(retained, source);
    }
}
