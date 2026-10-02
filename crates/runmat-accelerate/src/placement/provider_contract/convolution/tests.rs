use super::*;
use runmat_accelerate_api::{
    AccelDownloadFuture, GpuTensorHandle, HostTensorView, ProviderConvolutionGeometry,
    ProviderWorkload,
};

struct GeometryProvider;

impl AccelProvider for GeometryProvider {
    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("feasibility fixture never uploads")
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("feasibility fixture never downloads") })
    }

    fn device_info(&self) -> String {
        "convolution geometry fixture".into()
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
    let provider = GeometryProvider;
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
    let provider = GeometryProvider;
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
    let provider = GeometryProvider;
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
    let provider = GeometryProvider;
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
