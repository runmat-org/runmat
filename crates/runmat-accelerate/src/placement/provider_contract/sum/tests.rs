use super::*;
use runmat_accelerate_api::{
    AccelDownloadFuture, GpuTensorHandle, HostTensorView, ProviderSumGeometry, ProviderWorkload,
};

struct GeometryProvider {
    precision: ProviderPrecision,
}

impl AccelProvider for GeometryProvider {
    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("geometry fixture never uploads")
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("geometry fixture never downloads") })
    }

    fn device_info(&self) -> String {
        "sum geometry fixture".into()
    }

    fn precision(&self) -> ProviderPrecision {
        self.precision
    }

    fn sum_geometry(&self) -> Option<ProviderSumGeometry> {
        Some(ProviderSumGeometry {
            max_buffer_bytes: 1 << 20,
            max_binding_bytes: 1 << 20,
        })
    }
}

fn query(class: ProviderElementType, shape: &[u64], axes: &[u64]) -> ProviderFeasibilityQuery {
    let numeric = match class {
        ProviderElementType::F32 => NumericClass::Single,
        ProviderElementType::F64 => NumericClass::Double,
        ProviderElementType::I64 => NumericClass::Int64,
        _ => panic!("fixture class"),
    };
    ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new(runmat_builtins::SUM_CATALOG_ENTRY.identity.name),
        family: ProviderOperationFamily::Reduction,
        inputs: vec![ProviderRepresentation {
            element_type: class,
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape: shape.to_vec(),
            residency: ProviderResidency::Host,
        }],
        outputs: vec![],
        workload: ProviderWorkload {
            elements: shape.iter().copied().reduce(|a, b| a * b),
            sum: Some(ProviderSumWorkload {
                axes: axes.to_vec(),
                missing: ReductionNaN::Include,
                output: SumOutputClass::Double,
                input_class: numeric,
                input_domain: NumericDomain::Real,
                accumulation_class: NumericClass::Double,
                publication_class: NumericClass::Double,
            }),
            ..Default::default()
        },
    }
}

fn rejection_code(result: ProviderFeasibility) -> ProviderRejectionCode {
    match result {
        ProviderFeasibility::Rejected { rejection } => rejection.code,
        other => panic!("expected rejection, got {other:?}"),
    }
}

#[test]
fn resolved_axis_changes_output_bytes_and_workload_identity() {
    let provider = GeometryProvider {
        precision: ProviderPrecision::F64,
    };
    let row = query(ProviderElementType::F64, &[2, 3], &[0]);
    let column = query(ProviderElementType::F64, &[2, 3], &[1]);
    let row_work = work(&provider, true, &row).expect("2-D row kernel recipe");
    let column_work = work(&provider, true, &column).expect("2-D column kernel recipe");
    assert_eq!(row_work.estimate.output_bytes, Some(24));
    assert_eq!(column_work.estimate.output_bytes, Some(16));
    assert_eq!(row_work.estimate.dispatches, Some(1));
    assert_eq!(row_work.estimate.transient_bytes, None);
    assert_ne!(
        serde_json::to_vec(&row.workload).unwrap(),
        serde_json::to_vec(&column.workload).unwrap()
    );
}

#[test]
fn conversion_omit_nan_nd_and_view_do_not_borrow_simple_kernel_cost() {
    let provider = GeometryProvider {
        precision: ProviderPrecision::F64,
    };
    let base = query(ProviderElementType::F64, &[2, 3], &[0]);
    let mut changed = base.clone();
    changed.workload.sum.as_mut().unwrap().missing = ReductionNaN::Omit;
    assert_eq!(
        rejection_code(validate(&provider, true, &changed).unwrap()),
        ProviderRejectionCode::UnknownResourceEstimate
    );
    changed = query(ProviderElementType::F32, &[2, 3], &[0]);
    assert_eq!(
        rejection_code(validate(&provider, true, &changed).unwrap()),
        ProviderRejectionCode::UnknownResourceEstimate
    );
    changed = query(ProviderElementType::I64, &[2, 3], &[0]);
    assert_eq!(
        rejection_code(validate(&provider, true, &changed).unwrap()),
        ProviderRejectionCode::UnknownResourceEstimate
    );
    changed = query(ProviderElementType::F64, &[2, 3, 4], &[2]);
    assert_eq!(
        rejection_code(validate(&provider, true, &changed).unwrap()),
        ProviderRejectionCode::UnknownResourceEstimate
    );
    changed = base;
    changed.inputs[0].layout = ProviderLayout::Strided;
    assert_eq!(
        rejection_code(validate(&provider, true, &changed).unwrap()),
        ProviderRejectionCode::UnknownResourceEstimate
    );
    changed = query(ProviderElementType::F64, &[2, 3], &[0]);
    changed.inputs[0].residency = ProviderResidency::Device;
    assert_eq!(
        rejection_code(validate(&provider, true, &changed).unwrap()),
        ProviderRejectionCode::UnknownResourceEstimate
    );
}

#[test]
fn typed_controls_and_provider_precision_are_checked_before_estimation() {
    let provider = GeometryProvider {
        precision: ProviderPrecision::F64,
    };
    let original = query(ProviderElementType::F64, &[2, 3], &[0]);
    let mut invalid = original.clone();
    invalid.workload.sum.as_mut().unwrap().axes = vec![2];
    assert_eq!(
        rejection_code(validate(&provider, true, &invalid).unwrap()),
        ProviderRejectionCode::InvalidShape
    );
    invalid = original.clone();
    invalid.workload.sum.as_mut().unwrap().publication_class = NumericClass::Single;
    assert_eq!(
        rejection_code(validate(&provider, true, &invalid).unwrap()),
        ProviderRejectionCode::InvalidShape
    );
    invalid = original.clone();
    invalid.workload.elements = Some(7);
    assert_eq!(
        rejection_code(validate(&provider, true, &invalid).unwrap()),
        ProviderRejectionCode::InvalidShape
    );
    let single_only = GeometryProvider {
        precision: ProviderPrecision::F32,
    };
    assert_eq!(
        rejection_code(validate(&single_only, true, &original).unwrap()),
        ProviderRejectionCode::UnsupportedElementType
    );
    let mut unadmitted = original.clone();
    unadmitted.workload.sum = None;
    assert_eq!(
        rejection_code(validate(&provider, true, &unadmitted).unwrap()),
        ProviderRejectionCode::UnknownResourceEstimate
    );
    let too_large = query(ProviderElementType::F64, &[1024, 1024], &[0]);
    assert_eq!(
        rejection_code(validate(&provider, true, &too_large).unwrap()),
        ProviderRejectionCode::ResourceLimit
    );
}

#[test]
fn legacy_workload_roundtrips_without_inventing_sum_controls() {
    let old: ProviderWorkload =
        serde_json::from_str(r#"{"elements":4,"flops":null,"batch":null}"#).unwrap();
    assert!(old.sum.is_none());
    let route = query(ProviderElementType::F64, &[2, 3], &[0]);
    let serialized = serde_json::to_string(&route).unwrap();
    assert_eq!(
        serde_json::from_str::<ProviderFeasibilityQuery>(&serialized).unwrap(),
        route
    );
}

#[cfg(feature = "wgpu")]
#[test]
fn provider_cost_uses_the_same_resolved_axis_recipe() {
    use runmat_accelerate_api::ProviderCostQuery;
    use runmat_execution::CandidatePreparationState;

    let provider = GeometryProvider {
        precision: ProviderPrecision::F64,
    };
    let cost = |axis| {
        let operation = query(ProviderElementType::F64, &[200, 300], &[axis]);
        crate::placement::provider_cost::wgpu_cost_estimate(
            &provider,
            &ProviderCostQuery {
                operation,
                preparation: CandidatePreparationState::Cold,
                context: Default::default(),
                required_upload_bytes: 0,
                required_download_bytes: 0,
                downstream_materialization: false,
            },
        )
        .cost
    };
    let first = cost(0);
    let second = cost(1);
    assert!(first.components.execution_ns > second.components.execution_ns);
    assert_eq!(first.components.queue_ns, second.components.queue_ns);
    assert_eq!(
        first.confidence,
        runmat_execution::EstimateConfidence::Prior
    );
}
