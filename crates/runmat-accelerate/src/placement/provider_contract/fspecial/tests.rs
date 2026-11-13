use super::*;
use runmat_accelerate_api::{
    AccelDownloadFuture, GpuTensorHandle, HostTensorView, ProviderConvolutionGeometry,
    ProviderWorkload,
};

struct GeometryProvider {
    precision: ProviderPrecision,
    binding_bytes: u64,
    max_workgroups: u32,
}

impl AccelProvider for GeometryProvider {
    fn precision(&self) -> ProviderPrecision {
        self.precision
    }
    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        panic!("admission must not upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        panic!("admission must not download")
    }
    fn device_info(&self) -> String {
        "fspecial geometry fixture".into()
    }
    fn convolution_geometry(&self) -> Option<ProviderConvolutionGeometry> {
        Some(ProviderConvolutionGeometry {
            max_buffer_bytes: 4096,
            max_binding_bytes: self.binding_bytes,
            max_workgroups_per_dimension: self.max_workgroups,
            workgroup_size: 256,
        })
    }
}

fn query(filter: ProviderFspecialFilter, precision: ProviderPrecision) -> ProviderFeasibilityQuery {
    ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new("fspecial"),
        family: ProviderOperationFamily::Library,
        inputs: Vec::new(),
        outputs: vec![ProviderRepresentation {
            element_type: ProviderElementType::from(precision),
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape: filter.output_shape().unwrap().into(),
            residency: ProviderResidency::Device,
        }],
        workload: ProviderWorkload {
            fspecial: Some(filter),
            requested_outputs: Some(1),
            ..ProviderWorkload::default()
        },
    }
}

fn rejection(result: ProviderFeasibility) -> ProviderRejectionCode {
    match result {
        ProviderFeasibility::Rejected { rejection } => rejection.code,
        other => panic!("expected rejection, got {other:?}"),
    }
}

#[test]
fn exact_constructor_roles_output_and_physical_peak() {
    let provider = GeometryProvider {
        precision: ProviderPrecision::F32,
        binding_bytes: 4096,
        max_workgroups: 65535,
    };
    let mut exact = query(
        ProviderFspecialFilter::Average { rows: 2, cols: 3 },
        provider.precision(),
    );
    assert_eq!(
        validate(&provider, true, &exact),
        Some(ProviderFeasibility::supported(ProviderResourceEstimate {
            transient_bytes: Some(24 + FSPECIAL_F32_UNIFORM_BYTES),
            output_bytes: Some(24),
            dispatches: Some(1),
        }))
    );
    assert!(matches!(observation_capability(&provider, &exact),
        runmat_accelerate_api::observation::ProviderObservationCapability::Supported(overhead)
        if overhead.submissions == 1 && overhead.map_callbacks == 1));
    exact.outputs[0].shape = vec![3, 2];
    assert_eq!(
        rejection(validate(&provider, true, &exact).unwrap()),
        ProviderRejectionCode::InvalidShape
    );
    exact.outputs[0].shape = vec![2, 3];
    exact.outputs[0].element_type = ProviderElementType::F64;
    assert_eq!(
        rejection(validate(&provider, true, &exact).unwrap()),
        ProviderRejectionCode::InvalidShape
    );
    exact.outputs[0].element_type = ProviderElementType::F32;
    exact.inputs.push(exact.outputs[0].clone());
    assert_eq!(
        rejection(validate(&provider, true, &exact).unwrap()),
        ProviderRejectionCode::InvalidShape
    );
    exact.inputs.clear();
    exact.outputs[0].residency = ProviderResidency::Host;
    assert_eq!(
        rejection(validate(&provider, true, &exact).unwrap()),
        ProviderRejectionCode::InvalidShape
    );
    exact.outputs[0].residency = ProviderResidency::Device;
    exact.workload.requested_outputs = Some(2);
    assert_eq!(
        rejection(validate(&provider, true, &exact).unwrap()),
        ProviderRejectionCode::InvalidShape
    );
    exact.workload.requested_outputs = Some(1);
    exact.workload.fspecial = None;
    assert_eq!(
        rejection(validate(&provider, true, &exact).unwrap()),
        ProviderRejectionCode::UnknownResourceEstimate
    );
}

#[test]
fn exact_filter_and_single_buffer_limits_reject_before_provider_effects() {
    let provider = GeometryProvider {
        precision: ProviderPrecision::F64,
        binding_bytes: 32,
        max_workgroups: 65535,
    };
    let valid = query(
        ProviderFspecialFilter::Average { rows: 2, cols: 2 },
        provider.precision(),
    );
    assert_eq!(
        validate(&provider, true, &valid).unwrap(),
        ProviderFeasibility::supported(ProviderResourceEstimate {
            transient_bytes: Some(32 + FSPECIAL_F64_UNIFORM_BYTES),
            output_bytes: Some(32),
            dispatches: Some(1),
        })
    );
    let too_large = query(
        ProviderFspecialFilter::Average { rows: 2, cols: 3 },
        provider.precision(),
    );
    assert_eq!(
        rejection(validate(&provider, true, &too_large).unwrap()),
        ProviderRejectionCode::ResourceLimit
    );
    let unsupported = query(
        ProviderFspecialFilter::Disk {
            radius_bits: 1.0_f64.to_bits(),
            size: 3,
        },
        provider.precision(),
    );
    assert_eq!(
        rejection(validate(&provider, true, &unsupported).unwrap()),
        ProviderRejectionCode::UnsupportedOperation
    );
    let degenerate = query(
        ProviderFspecialFilter::Gaussian {
            rows: 3,
            cols: 3,
            sigma_bits: 0.0_f64.to_bits(),
        },
        provider.precision(),
    );
    assert_eq!(
        rejection(validate(&provider, true, &degenerate).unwrap()),
        ProviderRejectionCode::UnsupportedOperation
    );

    let launch_limited = GeometryProvider {
        precision: ProviderPrecision::F32,
        binding_bytes: 4096,
        max_workgroups: 1,
    };
    let too_many_groups = query(
        ProviderFspecialFilter::Average { rows: 17, cols: 16 },
        launch_limited.precision(),
    );
    assert_eq!(
        rejection(validate(&launch_limited, true, &too_many_groups).unwrap()),
        ProviderRejectionCode::ResourceLimit
    );
}
