use super::*;
use futures::executor::block_on;
use runmat_accelerate_api::{AccelProvider, HostNumericDataOwned, HostNumericTensorView};

#[test]
fn actual_wgpu_typed_sequences_preserve_class_and_exact_endpoints() {
    use runmat_accelerate_api::{ProviderPrecision, ProviderSequenceRequest};
    let provider = crate::backend::wgpu::provider::register_wgpu_provider(
        crate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("actual WGPU adapter required for typed sequences");
    let precision = provider.precision();
    let opposite = match precision {
        ProviderPrecision::F32 => ProviderPrecision::F64,
        ProviderPrecision::F64 => ProviderPrecision::F32,
    };
    assert!(provider
        .sequence(&ProviderSequenceRequest::linear(-0.0, 1.0, 2, opposite))
        .unwrap()
        .is_none());
    let linear = provider
        .sequence(&ProviderSequenceRequest::linear(-0.0, 1.0, 2, precision))
        .unwrap()
        .expect("native typed linear sequence");
    let logarithmic = provider
        .sequence(&ProviderSequenceRequest::log10(
            0.0,
            std::f64::consts::PI.log10(),
            3,
            precision,
            1.0,
            std::f64::consts::PI,
        ))
        .unwrap()
        .expect("native typed logarithmic sequence");
    assert_eq!(linear.shape(), [1, 2]);
    assert_eq!(logarithmic.shape(), [1, 3]);
    assert_eq!(
        runmat_accelerate_api::handle_precision(&linear),
        Some(precision)
    );
    assert_eq!(
        runmat_accelerate_api::handle_precision(&logarithmic),
        Some(precision)
    );
    match block_on(provider.download_numeric(&linear)).unwrap().data {
        HostNumericDataOwned::F32(values) => {
            assert_eq!(values[0].to_bits(), (-0.0f32).to_bits());
            assert_eq!(values[1].to_bits(), 1.0f32.to_bits());
        }
        HostNumericDataOwned::F64(values) => {
            assert_eq!(values[0].to_bits(), (-0.0f64).to_bits());
            assert_eq!(values[1].to_bits(), 1.0f64.to_bits());
        }
        other => panic!("unexpected linear class {other:?}"),
    }
    match block_on(provider.download_numeric(&logarithmic))
        .unwrap()
        .data
    {
        HostNumericDataOwned::F32(values) => {
            assert_eq!(values[2].to_bits(), (std::f64::consts::PI as f32).to_bits())
        }
        HostNumericDataOwned::F64(values) => {
            assert_eq!(values[2].to_bits(), std::f64::consts::PI.to_bits())
        }
        other => panic!("unexpected logarithmic class {other:?}"),
    }
    linear.release().unwrap();
    logarithmic.release().unwrap();
}

#[test]
fn actual_wgpu_fill_like_retains_negative_zero_sign() {
    let provider = crate::backend::wgpu::provider::register_wgpu_provider(
        crate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("actual WGPU adapter required for signed-zero fill");
    let prototype_data = match provider.precision() {
        runmat_accelerate_api::ProviderPrecision::F32 => HostNumericDataOwned::F32(vec![1.0, 1.0]),
        runmat_accelerate_api::ProviderPrecision::F64 => HostNumericDataOwned::F64(vec![1.0, 1.0]),
    };
    let prototype = provider
        .upload_numeric(&HostNumericTensorView {
            data: prototype_data.as_view(),
            shape: &[2],
            storage: GpuTensorStorage::Real,
        })
        .expect("native prototype");
    let filled = provider
        .fill_like(&prototype, -0.0)
        .expect("negative zero fill")
        .expect("admitted optional constructor");
    assert_eq!(filled.shape(), [2]);
    assert!(!filled.shares_allocation_with(&prototype));
    let gathered = block_on(provider.download_numeric(&filled)).unwrap();
    match gathered.data {
        HostNumericDataOwned::F32(values) => {
            assert!(values
                .iter()
                .all(|value| value.to_bits() == (-0.0_f32).to_bits()));
        }
        HostNumericDataOwned::F64(values) => {
            assert!(values
                .iter()
                .all(|value| value.to_bits() == (-0.0_f64).to_bits()));
        }
        other => panic!("unexpected native fill storage {other:?}"),
    }
    prototype.release().unwrap();
    filled.release().unwrap();
}

#[test]
fn actual_wgpu_typed_zeros_and_logical_fill_preserve_producer_facts() {
    let provider = crate::backend::wgpu::provider::register_wgpu_provider(
        crate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("typed constructors require an actual WGPU adapter");
    let fixtures = [
        (
            HostNumericDataOwned::F32(vec![7.0, 8.0]),
            GpuTensorStorage::Real,
        ),
        (
            HostNumericDataOwned::F32(vec![7.0, 8.0, 9.0, 10.0]),
            GpuTensorStorage::ComplexInterleaved,
        ),
        (HostNumericDataOwned::U8(vec![7, 8]), GpuTensorStorage::Real),
        (
            HostNumericDataOwned::U64(vec![7, u64::MAX]),
            GpuTensorStorage::Real,
        ),
    ];
    for (data, storage) in fixtures {
        let input = provider
            .upload_numeric(&HostNumericTensorView {
                data: data.as_view(),
                shape: &[1, 2],
                storage,
            })
            .unwrap();
        for shape in [vec![2, 1, 3], vec![usize::MAX, usize::MAX, 0]] {
            let output = provider
                .zeros_like(&input, &shape)
                .unwrap()
                .expect("admitted optional constructor");
            assert_eq!(output.shape(), shape);
            assert_eq!(
                output.physical().element_type(),
                input.physical().element_type()
            );
            assert_eq!(output.physical().storage(), storage);
            assert!(!output.shares_allocation_with(&input));
            let actual = block_on(provider.download_numeric(&output)).unwrap();
            let count = if shape.contains(&0) {
                0
            } else if storage == GpuTensorStorage::Real {
                6
            } else {
                12
            };
            let expected = match &data {
                HostNumericDataOwned::F32(_) => HostNumericDataOwned::F32(vec![0.0; count]),
                HostNumericDataOwned::U8(_) => HostNumericDataOwned::U8(vec![0; count]),
                HostNumericDataOwned::U64(_) => HostNumericDataOwned::U64(vec![0; count]),
                _ => unreachable!("fixture types are exhaustive above"),
            };
            assert_eq!(actual.data, expected);
        }
    }
    for value in [false, true] {
        let input = provider
            .logical_fill(&[1, 3], value)
            .unwrap()
            .expect("admitted optional constructor");
        assert!(input.view().is_logical());
        assert_eq!(
            block_on(provider.download(&input)).unwrap().data,
            vec![if value { 1.0 } else { 0.0 }; 3]
        );
        let output = provider
            .zeros_like(&input, &[2, 2])
            .unwrap()
            .expect("admitted optional constructor");
        assert!(output.view().is_logical());
        assert_eq!(
            block_on(provider.download(&output)).unwrap().data,
            vec![0.0; 4]
        );
        let empty = provider
            .logical_fill(&[usize::MAX, usize::MAX, 0], value)
            .unwrap()
            .expect("admitted optional constructor");
        assert!(empty.view().is_logical());
        assert_eq!(
            block_on(provider.download(&empty)).unwrap().data,
            Vec::<f64>::new()
        );
    }
}

#[test]
#[ignore = "requires an actual shader-f64 adapter; run in the F64 qualification gate"]
fn actual_wgpu_f64_zero_prototype_preserves_complex_storage() {
    let provider = crate::backend::wgpu::provider::register_wgpu_provider(
        crate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("F64 constructor gate requires an actual adapter");
    assert_eq!(
        provider.precision(),
        runmat_accelerate_api::ProviderPrecision::F64
    );
    let input = provider
        .upload_numeric(&HostNumericTensorView {
            data: runmat_accelerate_api::HostNumericDataView::F64(&[1.0, 2.0, 3.0, 4.0]),
            shape: &[1, 2],
            storage: GpuTensorStorage::ComplexInterleaved,
        })
        .unwrap();
    let output = provider
        .zeros_like(&input, &[3, 1])
        .unwrap()
        .expect("admitted optional constructor");
    assert_eq!(
        output.physical().element_type(),
        runmat_accelerate_api::NumericElementType::F64
    );
    assert_eq!(
        output.physical().storage(),
        GpuTensorStorage::ComplexInterleaved
    );
    assert_eq!(
        block_on(provider.download_numeric(&output)).unwrap().data,
        HostNumericDataOwned::F64(vec![0.0; 6])
    );
}

#[test]
fn constructor_shape_counts_are_zero_first_and_checked() {
    assert_eq!(constructor_elements(&[usize::MAX, usize::MAX, 0]), Some(0));
    assert_eq!(constructor_elements(&[usize::MAX, 2]), None);
    assert_eq!(constructor_elements(&[2, 3, 4]), Some(24));
}

#[test]
#[ignore = "requires actual WGPU adapter and completion worker; run in the image-filter hardware gate"]
fn fspecial_exact_admission_observes_completed_average_dispatch() {
    use runmat_accelerate_api::observation::{
        with_provider_observation, ProviderObservationStatus, ProviderObservationSupport,
    };
    use runmat_accelerate_api::{
        FspecialFilter, FspecialRequest, ProviderElementType, ProviderFeasibilityQuery,
        ProviderFspecialFilter, ProviderLayout, ProviderOperationFamily, ProviderOperationIdentity,
        ProviderRepresentation, ProviderResidency, ProviderStorage, ProviderWorkload,
    };
    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU adapter required");
    let request = FspecialRequest {
        filter: FspecialFilter::Average { rows: 2, cols: 3 },
    };
    let query = ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new("fspecial"),
        family: ProviderOperationFamily::Library,
        inputs: Vec::new(),
        outputs: vec![ProviderRepresentation {
            element_type: ProviderElementType::from(provider.precision()),
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape: vec![2, 3],
            residency: ProviderResidency::Device,
        }],
        workload: ProviderWorkload {
            fspecial: Some(ProviderFspecialFilter::from(&request)),
            requested_outputs: Some(1),
            ..Default::default()
        },
    };
    let runmat_accelerate_api::ProviderFeasibility::Supported { estimate } =
        provider.query_feasibility(&query)
    else {
        panic!("actual WGPU average constructor must have exact admission")
    };
    assert_eq!(estimate.dispatches, Some(1));
    let ProviderObservationSupport::Observed(mut receipt) =
        provider.begin_operation_observation(&query)
    else {
        panic!("actual WGPU average constructor must have a completion fence")
    };
    receipt.require_provider_execution(provider.device_id(), query.operation.clone());
    let output =
        with_provider_observation(Some(receipt.recorder()), || provider.fspecial(&request))
            .expect("observed average constructor");
    assert_eq!(output.shape(), &[2, 3]);
    assert_eq!(output.device_id(), provider.device_id());
    receipt.publish().expect("validated provider publication");
    let native = block_on(provider.download_numeric(&output)).expect("actual readback");
    let values = match native.data {
        HostNumericDataOwned::F32(values) => values.into_iter().map(f64::from).collect::<Vec<_>>(),
        HostNumericDataOwned::F64(values) => values,
        other => panic!("unexpected constructor payload: {other:?}"),
    };
    assert_eq!(values.len(), 6);
    assert!(values.iter().all(|value| (*value - 1.0 / 6.0).abs() < 1e-6));
    let completed = (0..1000)
        .find_map(|_| {
            let observation = receipt.take_completed();
            if observation.is_none() {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            observation
        })
        .expect("provider completion fence must resolve");
    assert!(matches!(
        completed.status,
        ProviderObservationStatus::Completed { submissions: 1, .. }
    ));
}
