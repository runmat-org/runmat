use super::*;
use runmat_accelerate_api::{
    AccelDownloadFuture, GpuTensorHandle, HostTensorView, ProviderConvolutionGeometry,
    ProviderConvolutionInputBackings, ProviderWorkload,
};

struct GeometryProvider {
    memory_limit: Option<u64>,
    precision: ProviderPrecision,
}

impl GeometryProvider {
    const fn unbounded() -> Self {
        Self {
            memory_limit: None,
            precision: ProviderPrecision::F64,
        }
    }
}

impl AccelProvider for GeometryProvider {
    fn precision(&self) -> ProviderPrecision {
        self.precision
    }
    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("feasibility fixture never uploads")
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("feasibility fixture never downloads") })
    }

    fn device_info(&self) -> String {
        "convolution geometry fixture".into()
    }

    fn device_info_struct(&self) -> runmat_accelerate_api::ApiDeviceInfo {
        runmat_accelerate_api::ApiDeviceInfo {
            device_id: 0,
            name: self.device_info(),
            vendor: "RunMat".into(),
            memory_bytes: self.memory_limit,
            backend: Some("fixture".into()),
        }
    }

    fn convolution_geometry(&self) -> Option<ProviderConvolutionGeometry> {
        Some(ProviderConvolutionGeometry {
            max_buffer_bytes: 4096,
            max_binding_bytes: 4096,
            max_workgroups_per_dimension: 4,
            workgroup_size: 64,
        })
    }
}

fn repr(shape: &[u64], element_type: ProviderElementType) -> ProviderRepresentation {
    ProviderRepresentation {
        element_type,
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape: shape.to_vec(),
        residency: ProviderResidency::Device,
    }
}

fn query(
    form: ProviderConvolutionForm,
    mode: ConvolutionMode,
    a: &[u64],
    b: &[u64],
) -> ProviderFeasibilityQuery {
    let identity = match form {
        ProviderConvolutionForm::Vector { .. } => "conv",
        ProviderConvolutionForm::Matrix => "conv2",
    };
    ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new(identity),
        family: ProviderOperationFamily::Library,
        inputs: vec![
            repr(a, ProviderElementType::F64),
            repr(b, ProviderElementType::F64),
        ],
        outputs: vec![],
        workload: ProviderWorkload {
            convolution: Some(ProviderConvolutionWorkload {
                form,
                mode,
                input_backings: None,
            }),
            ..Default::default()
        },
    }
}

fn vector() -> ProviderConvolutionForm {
    ProviderConvolutionForm::Vector {
        orientation: ProviderConvOrientation::Row,
    }
}

fn code(result: ProviderFeasibility) -> ProviderRejectionCode {
    match result {
        ProviderFeasibility::Rejected { rejection } => rejection.code,
        other => panic!("expected rejection, got {other:?}"),
    }
}

#[test]
fn exact_vector_mode_and_physical_limits() {
    let provider = GeometryProvider::unbounded();
    let full = query(vector(), ConvolutionMode::Full, &[1, 5], &[1, 3]);
    let same = query(vector(), ConvolutionMode::Same, &[1, 5], &[1, 3]);
    let valid = query(vector(), ConvolutionMode::Valid, &[1, 5], &[1, 3]);
    for (query, bytes) in [(full, 56), (same, 40), (valid, 24)] {
        let result = validate(&provider, true, &query).expect("recognized convolution");
        assert_eq!(
            result,
            ProviderFeasibility::supported(ProviderResourceEstimate {
                transient_bytes: None,
                output_bytes: Some(bytes),
                dispatches: Some(1),
            })
        );
    }
    let mut too_many = query(vector(), ConvolutionMode::Full, &[1, 300], &[1, 1]);
    assert_eq!(
        code(validate(&provider, true, &too_many).unwrap()),
        ProviderRejectionCode::ResourceLimit
    );
    too_many.inputs[0].element_type = ProviderElementType::F32;
    assert_eq!(
        code(validate(&provider, true, &too_many).unwrap()),
        ProviderRejectionCode::UnsupportedElementType
    );
}

#[test]
fn matrix_hook_is_in_process_only_and_empty_geometry_is_typed() {
    let provider = GeometryProvider::unbounded();
    let matrix = query(
        ProviderConvolutionForm::Matrix,
        ConvolutionMode::Same,
        &[2, 3],
        &[2, 2],
    );
    assert_eq!(
        code(validate(&provider, true, &matrix).unwrap()),
        ProviderRejectionCode::UnsupportedOperation
    );
    let supported = validate(&provider, false, &matrix).unwrap();
    assert_eq!(
        supported,
        ProviderFeasibility::supported(ProviderResourceEstimate {
            transient_bytes: None,
            output_bytes: Some(48),
            dispatches: Some(0),
        })
    );
    let empty = query(vector(), ConvolutionMode::Valid, &[1, 2], &[1, 4]);
    let result = validate(&provider, true, &empty).unwrap();
    assert_eq!(
        result,
        ProviderFeasibility::supported(ProviderResourceEstimate {
            transient_bytes: None,
            output_bytes: Some(8),
            dispatches: Some(0),
        })
    );
}

#[test]
fn output_and_route_claims_must_match_real_hook() {
    let provider = GeometryProvider::unbounded();
    let mut q = query(vector(), ConvolutionMode::Same, &[1, 5], &[1, 3]);
    q.outputs.push(repr(&[1, 7], ProviderElementType::F64));
    assert_eq!(
        code(validate(&provider, true, &q).unwrap()),
        ProviderRejectionCode::InvalidShape
    );
    q.outputs[0] = repr(&[1, 5], ProviderElementType::F64);
    q.workload.scalar_work_units = Some(14);
    assert_eq!(
        code(validate(&provider, true, &q).unwrap()),
        ProviderRejectionCode::InvalidShape
    );
    q.workload.scalar_work_units = Some(15);
    assert!(validate(&provider, true, &q).unwrap().is_supported());
    q.operation = ProviderOperationIdentity::new("deconv");
    assert!(validate(&provider, true, &q).is_none());
    assert_eq!(
        code(super::super::in_process_feasibility(&provider, &q)),
        ProviderRejectionCode::UnsupportedOperation
    );
}

#[test]
fn capability_inventory_matches_implemented_hooks() {
    let provider = GeometryProvider::unbounded();
    let host = super::super::in_process_capability_snapshot(&provider);
    assert!(host.supports_operation(
        &ProviderOperationIdentity::new("conv"),
        ProviderOperationFamily::Library
    ));
    assert!(host.supports_operation(
        &ProviderOperationIdentity::new("conv2"),
        ProviderOperationFamily::Library
    ));
    assert!(!host.supports_operation(
        &ProviderOperationIdentity::new("deconv"),
        ProviderOperationFamily::Library
    ));
    #[cfg(feature = "wgpu")]
    {
        let device = super::super::wgpu_capability_snapshot(&provider, 4096);
        assert_eq!(device.max_allocation_bytes, Some(4096));
        assert_eq!(device.device.memory_bytes, provider.memory_limit);
        assert!(device.supports_operation(
            &ProviderOperationIdentity::new("conv"),
            ProviderOperationFamily::Library
        ));
        assert!(!device.supports_operation(
            &ProviderOperationIdentity::new("conv2"),
            ProviderOperationFamily::Library
        ));
        assert!(!device.supports_operation(
            &ProviderOperationIdentity::new("deconv"),
            ProviderOperationFamily::Library
        ));
    }
}

#[test]
fn checked_peak_does_not_misuse_a_per_allocation_limit() {
    let source = crate::simple_provider::InProcessProvider::new();
    let signal = source
        .upload(&HostTensorView {
            data: &[1.0; 5],
            shape: &[1, 5],
        })
        .unwrap();
    let kernel = source
        .upload(&HostTensorView {
            data: &[1.0; 3],
            shape: &[1, 3],
        })
        .unwrap();
    let backing = ProviderConvolutionInputBackings::from_full_live_handles([&signal, &kernel])
        .unwrap()
        .expect("full registered input views");
    assert_eq!(backing.live_requested_bytes(), Some([40, 24]));
    assert_eq!(
        backing.unique_live_bytes([&signal, &kernel]).unwrap(),
        Some(64)
    );
    assert!(backing.matches_live_handles([&signal, &kernel]).unwrap());
    assert!(!backing.matches_live_handles([&kernel, &signal]).unwrap());
    assert_eq!(backing.unique_live_bytes([&kernel, &signal]).unwrap(), None);
    let alias = ProviderConvolutionInputBackings::from_full_live_handles([&signal, &signal])
        .unwrap()
        .expect("same allocation is an admissible ordered pair");
    assert_eq!(
        alias.unique_live_bytes([&signal, &signal]).unwrap(),
        Some(40)
    );
    let mut request = query(vector(), ConvolutionMode::Full, &[1, 5], &[1, 3]);
    request
        .workload
        .convolution
        .as_mut()
        .unwrap()
        .input_backings = Some(backing);
    let exact = GeometryProvider {
        // Each allocation is at most 56 bytes, while the route has 136 bytes
        // live in aggregate. The latter is not a single-buffer limit.
        memory_limit: Some(100),
        precision: ProviderPrecision::F64,
    };
    assert_eq!(
        validate(&exact, true, &request),
        Some(ProviderFeasibility::supported(ProviderResourceEstimate {
            transient_bytes: Some(136),
            output_bytes: Some(56),
            dispatches: Some(1),
        }))
    );
    // Transport retains stable byte facts for cache identity but cannot mint
    // a process-local live receipt.
    let wire = serde_json::to_vec(&request).unwrap();
    let decoded: ProviderFeasibilityQuery = serde_json::from_slice(&wire).unwrap();
    assert_eq!(
        decoded
            .workload
            .convolution
            .unwrap()
            .input_backings
            .unwrap()
            .live_requested_bytes(),
        None
    );
    let ProviderFeasibility::Supported { estimate } = validate(&exact, true, &decoded).unwrap()
    else {
        panic!("untrusted wire claim should remain feasible with unknown peak");
    };
    assert_eq!(estimate.transient_bytes, None);
    let output_too_large = GeometryProvider {
        memory_limit: Some(55),
        precision: ProviderPrecision::F64,
    };
    assert_eq!(
        code(validate(&output_too_large, true, &decoded).unwrap()),
        ProviderRejectionCode::ResourceLimit
    );
}

#[test]
fn in_process_convolution_receipt_completes_only_after_hook_publication() {
    use runmat_accelerate_api::observation::{
        with_provider_observation, ProviderObservationStatus, ProviderObservationSupport,
    };
    let provider = crate::simple_provider::InProcessProvider::new();
    let signal = provider
        .upload(&HostTensorView {
            data: &[1.0; 5],
            shape: &[1, 5],
        })
        .unwrap();
    let kernel = provider
        .upload(&HostTensorView {
            data: &[1.0; 3],
            shape: &[1, 3],
        })
        .unwrap();
    let backing = ProviderConvolutionInputBackings::from_full_live_handles([&signal, &kernel])
        .unwrap()
        .expect("full handles");
    let mut request = query(vector(), ConvolutionMode::Full, &[1, 5], &[1, 3]);
    request
        .workload
        .convolution
        .as_mut()
        .unwrap()
        .input_backings = Some(backing);
    let ProviderObservationSupport::Observed(mut receipt) =
        provider.begin_operation_observation(&request)
    else {
        panic!("in-process convolution is synchronous");
    };
    receipt.require_provider_execution(provider.device_id(), request.operation.clone());
    let result = with_provider_observation(Some(receipt.recorder()), || {
        provider.conv1d(
            &signal,
            &kernel,
            runmat_accelerate_api::ProviderConv1dOptions {
                mode: runmat_accelerate_api::ProviderConvMode::Full,
                orientation: ProviderConvOrientation::Row,
            },
        )
    });
    assert!(result.is_ok());
    assert_eq!(receipt.status(), ProviderObservationStatus::Pending);
    receipt.publish().unwrap();
    assert!(matches!(
        receipt.status(),
        ProviderObservationStatus::Completed { .. }
    ));
}

#[test]
fn in_process_host_upload_receipt_requires_the_actual_restore_hook() {
    use runmat_accelerate_api::observation::{
        with_provider_observation, ProviderObservationFailure, ProviderObservationStatus,
        ProviderObservationSupport,
    };
    let provider = crate::simple_provider::InProcessProvider::new();
    let output = ProviderRepresentation {
        element_type: ProviderElementType::F64,
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape: vec![1, 3],
        residency: ProviderResidency::Device,
    };
    let request = ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new("transfer.upload"),
        family: ProviderOperationFamily::Upload,
        inputs: vec![ProviderRepresentation {
            residency: ProviderResidency::Host,
            ..output.clone()
        }],
        outputs: vec![output],
        workload: ProviderWorkload::default(),
    };
    let ProviderObservationSupport::Observed(mut unproven) =
        provider.begin_operation_observation(&request)
    else {
        panic!("in-process upload should be observable");
    };
    unproven.require_provider_execution(provider.device_id(), request.operation.clone());
    assert_eq!(
        unproven.publish(),
        Err(ProviderObservationFailure::ExecutionUnproven)
    );

    let ProviderObservationSupport::Observed(mut receipt) =
        provider.begin_operation_observation(&request)
    else {
        panic!("in-process upload should be observable");
    };
    receipt.require_provider_execution(provider.device_id(), request.operation.clone());
    let result = with_provider_observation(Some(receipt.recorder()), || {
        provider.upload_numeric_raw(&runmat_accelerate_api::HostNumericTensorView {
            data: runmat_accelerate_api::HostNumericDataView::F64(&[1.0, 2.0, 3.0]),
            shape: &[1, 3],
            storage: runmat_accelerate_api::GpuTensorStorage::Real,
        })
    });
    assert!(result.is_ok());
    receipt.publish().unwrap();
    assert!(matches!(
        receipt.status(),
        ProviderObservationStatus::Completed { submissions: 0, .. }
    ));
}

#[cfg(feature = "wgpu")]
#[test]
fn only_the_admitted_wgpu_vector_route_claims_completion_fences() {
    use runmat_accelerate_api::observation::{
        ProviderObservationCapability, ProviderObservationOverhead,
    };
    let provider = GeometryProvider {
        memory_limit: None,
        precision: ProviderPrecision::F64,
    };
    let vector = query(vector(), ConvolutionMode::Same, &[1, 5], &[1, 3]);
    assert_eq!(
        observation_capability(&provider, &vector),
        ProviderObservationCapability::Supported(ProviderObservationOverhead {
            submissions: 1,
            requested_peak_buffer_bytes: 8,
            clear_commands: 1,
            map_callbacks: 1,
            wgpu_compute_query:
                crate::backend::wgpu::dispatch::observation::prospective_encoder_query(1, 1),
        })
    );
    let matrix = query(
        ProviderConvolutionForm::Matrix,
        ConvolutionMode::Full,
        &[3, 4],
        &[2, 2],
    );
    assert_eq!(
        observation_capability(&provider, &matrix),
        ProviderObservationCapability::Unsupported
    );
}

#[test]
fn in_process_full_crop_and_early_empty_routes_have_distinct_peaks() {
    let provider = GeometryProvider::unbounded();
    let vector_full = query(vector(), ConvolutionMode::Full, &[1, 5], &[1, 3]);
    let vector_same = query(vector(), ConvolutionMode::Same, &[1, 5], &[1, 3]);
    let vector_valid = query(vector(), ConvolutionMode::Valid, &[1, 2], &[1, 4]);
    for request in [vector_full, vector_same, vector_valid] {
        let ProviderFeasibility::Supported { estimate } =
            validate(&provider, false, &request).expect("recognized convolution")
        else {
            panic!("physical route rejected");
        };
        assert_eq!(estimate.transient_bytes, None);
    }
    let matrix = query(
        ProviderConvolutionForm::Matrix,
        ConvolutionMode::Valid,
        &[2, 3],
        &[2, 2],
    );
    let ProviderFeasibility::Supported { estimate } = validate(&provider, false, &matrix).unwrap()
    else {
        panic!("in-process matrix route rejected");
    };
    assert_eq!(estimate.transient_bytes, None);
    // These are checked physical-route calculations, not proof of borrowed
    // allocation backing when no live receipt accompanies the query.
    for (request, output_elements, output_bytes, backings, expected) in [
        (
            query(vector(), ConvolutionMode::Full, &[1, 5], &[1, 3]),
            7,
            56,
            [40, 24],
            240,
        ),
        (
            query(vector(), ConvolutionMode::Same, &[1, 5], &[1, 3]),
            5,
            40,
            [40, 24],
            224,
        ),
        (
            query(vector(), ConvolutionMode::Valid, &[1, 2], &[1, 4]),
            0,
            0,
            [16, 32],
            48,
        ),
    ] {
        let result = resources::estimate(
            ConvolutionIdentity::Vector,
            request.workload.convolution.unwrap(),
            &request.inputs,
            output_elements,
            output_bytes,
            false,
            Some(backings),
        )
        .unwrap();
        assert_eq!(result.0.transient_bytes, Some(expected));
    }
    let result = resources::estimate(
        ConvolutionIdentity::Matrix,
        matrix.workload.convolution.unwrap(),
        &matrix.inputs,
        2,
        16,
        false,
        Some([48, 32]),
    )
    .unwrap();
    assert_eq!(result.0.transient_bytes, Some(272));
    assert!(resources::estimate(
        ConvolutionIdentity::Matrix,
        matrix.workload.convolution.unwrap(),
        &matrix.inputs,
        2,
        16,
        false,
        Some([u64::MAX, 32]),
    )
    .is_none());
}

#[test]
fn f32_shader_uniform_and_empty_view_are_accounted_separately() {
    let provider = GeometryProvider {
        memory_limit: None,
        precision: ProviderPrecision::F32,
    };
    let mut full = query(vector(), ConvolutionMode::Full, &[1, 5], &[1, 3]);
    for input in &mut full.inputs {
        input.element_type = ProviderElementType::F32;
    }
    assert_eq!(
        validate(&provider, true, &full),
        Some(ProviderFeasibility::supported(ProviderResourceEstimate {
            transient_bytes: None,
            output_bytes: Some(28),
            dispatches: Some(1),
        }))
    );
    let mut empty = query(vector(), ConvolutionMode::Full, &[1, 0], &[1, 3]);
    for input in &mut empty.inputs {
        input.element_type = ProviderElementType::F32;
    }
    assert_eq!(
        validate(&provider, true, &empty),
        Some(ProviderFeasibility::supported(ProviderResourceEstimate {
            // A zero-length view does not identify its retained backing buffer.
            transient_bytes: None,
            output_bytes: Some(4),
            dispatches: Some(0),
        }))
    );
    assert_eq!(
        resources::estimate(
            ConvolutionIdentity::Vector,
            full.workload.convolution.unwrap(),
            &full.inputs,
            7,
            28,
            true,
            Some([20, 12]),
        )
        .unwrap()
        .0
        .transient_bytes,
        Some(76)
    );
}
