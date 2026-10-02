use super::*;
use runmat_accelerate_api::{
    AccelDownloadFuture, GpuTensorHandle, HostTensorView, ProviderConvolutionGeometry,
    ProviderWorkload,
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
            convolution: Some(ProviderConvolutionWorkload { form, mode }),
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
    for (query, bytes, peak) in [(full, 56, 136), (same, 40, 120), (valid, 24, 104)] {
        let result = validate(&provider, true, &query).expect("recognized convolution");
        assert_eq!(
            result,
            ProviderFeasibility::supported(ProviderResourceEstimate {
                transient_bytes: Some(peak),
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
            transient_bytes: Some(304),
            output_bytes: Some(48),
            dispatches: Some(0),
        })
    );
    let empty = query(vector(), ConvolutionMode::Valid, &[1, 2], &[1, 4]);
    let result = validate(&provider, true, &empty).unwrap();
    assert_eq!(
        result,
        ProviderFeasibility::supported(ProviderResourceEstimate {
            transient_bytes: Some(56),
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
        let device = super::super::wgpu_capability_snapshot(&provider);
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
fn checked_peak_obeys_a_known_provider_memory_limit() {
    let request = query(vector(), ConvolutionMode::Full, &[1, 5], &[1, 3]);
    let insufficient = GeometryProvider {
        memory_limit: Some(135),
        precision: ProviderPrecision::F64,
    };
    assert_eq!(
        code(validate(&insufficient, true, &request).unwrap()),
        ProviderRejectionCode::ResourceLimit
    );
    let exact = GeometryProvider {
        memory_limit: Some(136),
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
}

#[test]
fn in_process_full_crop_and_early_empty_routes_have_distinct_peaks() {
    let provider = GeometryProvider::unbounded();
    let vector_full = query(vector(), ConvolutionMode::Full, &[1, 5], &[1, 3]);
    let vector_same = query(vector(), ConvolutionMode::Same, &[1, 5], &[1, 3]);
    let vector_valid = query(vector(), ConvolutionMode::Valid, &[1, 2], &[1, 4]);
    for (request, expected) in [(vector_full, 240), (vector_same, 224), (vector_valid, 48)] {
        let ProviderFeasibility::Supported { estimate } =
            validate(&provider, false, &request).expect("recognized convolution")
        else {
            panic!("physical route rejected");
        };
        assert_eq!(estimate.transient_bytes, Some(expected));
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
    // Registered inputs 80, their clones 80, full 96, valid crop 16.
    assert_eq!(estimate.transient_bytes, Some(272));
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
            transient_bytes: Some(76),
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
}
