use super::*;
use runmat_accelerate_api::{
    AccelDownloadFuture, GpuTensorHandle, HostTensorView, ProviderShaderRequirements,
    ProviderSumGeometry, ProviderWorkload,
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

    fn supports_shader_requirements(&self, required: ProviderShaderRequirements) -> bool {
        match required {
            ProviderShaderRequirements::Basic => true,
            ProviderShaderRequirements::F64IeeeWords => self.precision == ProviderPrecision::F64,
        }
    }

    fn sum_geometry(&self) -> Option<ProviderSumGeometry> {
        Some(ProviderSumGeometry {
            max_buffer_bytes: 1 << 20,
            max_binding_bytes: 1 << 20,
            nd_chunk_elements: 128,
            cast_chunk_elements: 128,
            integer_cast_chunk_elements: 128,
            dimension_2_max_output_elements: 512 * 65535,
            global_chunk_elements: 1024,
            global_group_elements: 1024,
        })
    }
}

fn query(class: ProviderElementType, shape: &[u64], axes: &[u64]) -> ProviderFeasibilityQuery {
    let numeric = match class {
        ProviderElementType::F32 => NumericClass::Single,
        ProviderElementType::F64 => NumericClass::Double,
        ProviderElementType::I64 => NumericClass::Int64,
        ProviderElementType::I8 => NumericClass::Int8,
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
                invocation: runmat_types::SumInvocation::Sum,
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
    assert!(matches!(
        validate(&provider, true, &changed),
        Some(ProviderFeasibility::Supported { .. })
    ));
    changed = query(ProviderElementType::F32, &[2, 3], &[0]);
    changed.workload.sum.as_mut().unwrap().output = SumOutputClass::Native;
    changed.workload.sum.as_mut().unwrap().publication_class = NumericClass::Single;
    assert!(matches!(
        validate(&provider, true, &changed),
        Some(ProviderFeasibility::Supported { .. })
    ));
    changed = query(ProviderElementType::I64, &[2, 3], &[0]);
    assert!(matches!(
        validate(&provider, true, &changed),
        Some(ProviderFeasibility::Supported { .. })
    ));
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
fn unspecified_logical_physical_storage_never_borrows_f64_upload_cost() {
    let provider = GeometryProvider {
        precision: ProviderPrecision::F64,
    };
    let mut request = query(ProviderElementType::F64, &[2, 3], &[0]);
    request.inputs[0].element_type = ProviderElementType::Logical;
    assert_eq!(
        rejection_code(validate(&provider, true, &request).unwrap()),
        ProviderRejectionCode::UnknownResourceEstimate
    );
}

#[test]
fn integer_to_double_uses_word_storage_and_its_own_cast_geometry() {
    let provider = GeometryProvider {
        precision: ProviderPrecision::F64,
    };
    let wide = query(ProviderElementType::I64, &[2, 3], &[0]);
    let narrow = query(ProviderElementType::I8, &[2, 3], &[0]);
    assert_eq!(
        crate::backend::wgpu_storage::checked_allocation_bytes(&wide.inputs[0]),
        Some(48)
    );
    assert_eq!(
        crate::backend::wgpu_storage::checked_allocation_bytes(&narrow.inputs[0]),
        Some(24)
    );
    for request in [&wide, &narrow] {
        let work = work(&provider, true, request).unwrap();
        assert_eq!(work.estimate.output_bytes, Some(24));
        assert_eq!(work.estimate.dispatches, Some(2));
        assert_eq!(work.estimate.transient_bytes, None);
        #[cfg(feature = "wgpu")]
        assert!(work.integer_input_cast && !work.input_cast && !work.output_cast);
    }
    // Omit-NaN is a no-op on integer values, so it uses the same route.
    let mut omit = wide.clone();
    omit.workload.sum.as_mut().unwrap().missing = ReductionNaN::Omit;
    assert_eq!(
        work(&provider, true, &omit).unwrap().estimate.dispatches,
        work(&provider, true, &wide).unwrap().estimate.dispatches
    );
    let mut native = wide.clone();
    let controls = native.workload.sum.as_mut().unwrap();
    controls.output = SumOutputClass::Native;
    controls.accumulation_class = NumericClass::Int64;
    controls.publication_class = NumericClass::Int64;
    assert_eq!(
        rejection_code(validate(&provider, true, &native).unwrap()),
        ProviderRejectionCode::UnknownResourceEstimate
    );
}

#[test]
fn integer_cast_dispatches_use_integer_shader_geometry_not_floating_cast_geometry() {
    let geometry = ProviderSumGeometry {
        max_buffer_bytes: 1 << 20,
        max_binding_bytes: 1 << 20,
        nd_chunk_elements: 128,
        cast_chunk_elements: 128,
        integer_cast_chunk_elements: 2,
        dimension_2_max_output_elements: 1024,
        global_chunk_elements: 1024,
        global_group_elements: 1024,
    };
    let mut work = floating::dimension_work(2, 1, 6, 3, 24, geometry).unwrap();
    floating::add_integer_input_cast(&mut work, 6, geometry).unwrap();
    assert_eq!(work.estimate.dispatches, Some(4));
    #[cfg(feature = "wgpu")]
    assert_eq!(work.queue_submissions, 4);
}

#[test]
fn nansum_binding_is_distinct_but_uses_the_same_typed_physical_route() {
    let provider = GeometryProvider {
        precision: ProviderPrecision::F64,
    };
    let mut alias = query(ProviderElementType::F64, &[2, 3], &[0]);
    alias.operation =
        ProviderOperationIdentity::new(runmat_builtins::NANSUM_CATALOG_ENTRY.identity.name);
    let route = alias.workload.sum.as_mut().unwrap();
    route.invocation = runmat_types::SumInvocation::NanSum;
    route.missing = ReductionNaN::Omit;
    assert!(matches!(
        validate(&provider, true, &alias),
        Some(ProviderFeasibility::Supported { .. })
    ));
    let sum = query(ProviderElementType::F64, &[2, 3], &[0]);
    assert_ne!(
        serde_json::to_vec(&alias.workload).unwrap(),
        serde_json::to_vec(&sum.workload).unwrap()
    );
    alias.workload.sum.as_mut().unwrap().invocation = runmat_types::SumInvocation::Sum;
    assert_eq!(
        rejection_code(validate(&provider, true, &alias).unwrap()),
        ProviderRejectionCode::InvalidShape
    );
}

#[test]
fn omission_counts_map_warmup_and_each_dimensional_reduction() {
    let provider = GeometryProvider {
        precision: ProviderPrecision::F64,
    };
    let mut one = query(ProviderElementType::F64, &[2, 3], &[0]);
    one.workload.sum.as_mut().unwrap().missing = ReductionNaN::Omit;
    let one_work = work(&provider, true, &one).unwrap();
    assert_eq!(one_work.estimate.output_bytes, Some(24));
    assert_eq!(one_work.estimate.dispatches, Some(2));
    assert_eq!(one_work.estimate.transient_bytes, None);
    #[cfg(feature = "wgpu")]
    {
        use runmat_accelerate_api::observation::ProviderObservationCapability;
        assert!(one_work.nan_map);
        assert_eq!(one_work.queue_submissions, 3);
        assert_eq!(one_work.allocations, 5);
        let ProviderObservationCapability::Supported(overhead) =
            observation_capability(&provider, &one)
        else {
            panic!("audited omit-NaN route");
        };
        assert_eq!(overhead.submissions, 3);
        assert_eq!(overhead.wgpu_compute_query.unwrap().max_query_sets, 3);
    }
    let mut all = query(ProviderElementType::F64, &[2, 3], &[0, 1]);
    all.workload.sum.as_mut().unwrap().missing = ReductionNaN::Omit;
    let all_work = work(&provider, true, &all).unwrap();
    assert_eq!(all_work.estimate.output_bytes, Some(8));
    assert_eq!(all_work.estimate.dispatches, Some(3));
    #[cfg(feature = "wgpu")]
    {
        assert_eq!(all_work.queue_submissions, 4);
        assert_eq!(all_work.allocations, 7);
    }
    let mut single = query(ProviderElementType::F32, &[2, 3], &[0]);
    let route = single.workload.sum.as_mut().unwrap();
    route.missing = ReductionNaN::Omit;
    route.output = SumOutputClass::Native;
    route.publication_class = NumericClass::Single;
    let single_work = work(&provider, true, &single).unwrap();
    assert_eq!(single_work.estimate.output_bytes, Some(12));
    assert_eq!(single_work.estimate.dispatches, Some(4));
    #[cfg(feature = "wgpu")]
    {
        assert!(single_work.nan_map && single_work.input_cast && single_work.output_cast);
        assert_eq!(single_work.queue_submissions, 5);
        assert_eq!(single_work.allocations, 9);
    }
}

#[test]
fn single_input_cast_and_publication_are_separate_physical_work() {
    let provider = GeometryProvider {
        precision: ProviderPrecision::F64,
    };
    let mut native = query(ProviderElementType::F32, &[2, 3], &[0]);
    let route = native.workload.sum.as_mut().unwrap();
    route.output = SumOutputClass::Native;
    route.publication_class = NumericClass::Single;
    let native_work = work(&provider, true, &native).unwrap();
    assert_eq!(native_work.estimate.output_bytes, Some(12));
    assert_eq!(native_work.estimate.dispatches, Some(3));
    #[cfg(feature = "wgpu")]
    {
        assert!(native_work.input_cast && native_work.output_cast);
        assert_eq!(native_work.queue_submissions, 3);
        assert_eq!(native_work.allocations, 7);
    }
    let mut double = native.clone();
    let route = double.workload.sum.as_mut().unwrap();
    route.output = SumOutputClass::Double;
    route.publication_class = NumericClass::Double;
    let double_work = work(&provider, true, &double).unwrap();
    assert_eq!(double_work.estimate.output_bytes, Some(24));
    assert_eq!(double_work.estimate.dispatches, Some(2));
    #[cfg(feature = "wgpu")]
    assert_eq!(double_work.allocations, 5);
    let single_only = GeometryProvider {
        precision: ProviderPrecision::F32,
    };
    assert_eq!(
        rejection_code(validate(&single_only, true, &native).unwrap()),
        ProviderRejectionCode::UnsupportedElementType
    );
}

#[test]
fn two_dimensional_launch_geometry_admits_exact_adapter_boundary() {
    let geometry = ProviderSumGeometry {
        max_buffer_bytes: u64::MAX,
        max_binding_bytes: u64::MAX,
        nd_chunk_elements: 128,
        cast_chunk_elements: 128,
        integer_cast_chunk_elements: 128,
        dimension_2_max_output_elements: 1024,
        global_chunk_elements: 1024,
        global_group_elements: 1024,
    };
    assert!(floating::dimension_work(2, 1, 2048, 1024, 8192, geometry).is_ok());
    match floating::dimension_work(2, 1, 2050, 1025, 8200, geometry) {
        Err(reason) => assert_eq!(rejection_code(reason), ProviderRejectionCode::ResourceLimit),
        Ok(_) => panic!("dispatch beyond the adapter limit was admitted"),
    }
}

#[test]
fn nd_chunks_follow_selected_provider_geometry_and_reduced_axis() {
    let provider = GeometryProvider {
        precision: ProviderPrecision::F64,
    };
    let first = query(ProviderElementType::F64, &[3, 5, 40], &[0]);
    let last = query(ProviderElementType::F64, &[3, 5, 40], &[2]);
    let first_work = work(&provider, true, &first).unwrap();
    let last_work = work(&provider, true, &last).unwrap();
    assert_eq!(first_work.estimate.output_bytes, Some(1600));
    assert_eq!(last_work.estimate.output_bytes, Some(120));
    assert_eq!(first_work.estimate.dispatches, Some(2));
    assert_eq!(last_work.estimate.dispatches, Some(1));
    assert_eq!(first_work.estimate.transient_bytes, None);
    assert_eq!(last_work.estimate.transient_bytes, None);
}

#[test]
fn complex_host_output_accounts_for_both_physical_lanes() {
    let provider = GeometryProvider {
        precision: ProviderPrecision::F64,
    };
    for (class, element, expected_bytes) in [
        (NumericClass::Double, ProviderElementType::ComplexF64, 48),
        (NumericClass::Single, ProviderElementType::ComplexF32, 24),
    ] {
        let mut request = query(ProviderElementType::F64, &[2, 3], &[0]);
        request.inputs[0].element_type = element;
        request.inputs[0].storage = ProviderStorage::DenseComplexInterleaved;
        let route = request.workload.sum.as_mut().unwrap();
        route.input_class = class;
        route.input_domain = NumericDomain::Complex;
        route.output = SumOutputClass::Native;
        route.publication_class = class;
        assert_eq!(
            work(&provider, false, &request)
                .unwrap()
                .estimate
                .output_bytes,
            Some(expected_bytes)
        );
    }
}

#[test]
fn global_sum_counts_partial_passes_and_not_one_dimensional_dispatch() {
    let provider = GeometryProvider {
        precision: ProviderPrecision::F64,
    };
    let request = query(ProviderElementType::F64, &[2, 1024], &[0, 1]);
    let work = work(&provider, true, &request).unwrap();
    assert_eq!(work.estimate.output_bytes, Some(8));
    assert_eq!(work.estimate.dispatches, Some(3));
    assert_eq!(work.estimate.transient_bytes, None);
    #[cfg(feature = "wgpu")]
    {
        assert_eq!(work.queue_submissions, 3);
        assert_eq!(work.allocations, 6);
        assert_eq!(work.uniform_bytes, 48);
    }
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
    assert!(!serde_json::to_string(&old).unwrap().contains("\"sum\""));
    let route = query(ProviderElementType::F64, &[2, 3], &[0]);
    let serialized = serde_json::to_string(&route).unwrap();
    assert!(!serialized.contains("\"invocation\""));
    assert_eq!(
        serde_json::from_str::<ProviderFeasibilityQuery>(&serialized).unwrap(),
        route
    );
    let mut prior = serde_json::to_value(&route).unwrap();
    let admitted = prior["workload"]["sum"]
        .as_object_mut()
        .expect("sum workload object");
    admitted.remove("invocation");
    assert_eq!(
        serde_json::from_value::<ProviderFeasibilityQuery>(prior).unwrap(),
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
