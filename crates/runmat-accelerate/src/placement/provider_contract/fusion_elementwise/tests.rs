use super::fixture::GeometryProvider;
use super::*;

#[test]
fn provider_instances_keep_independent_launch_geometry() {
    let first = GeometryProvider(64);
    let mut input = value(ProviderElementType::F64, ProviderResidency::Device);
    input.shape = vec![1, 65_535 * 64 + 1];
    let query = ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new("fusion.elementwise"),
        family: ProviderOperationFamily::Fusion,
        inputs: vec![input.clone()],
        outputs: vec![input],
        workload: Default::default(),
    };
    let before = physical_work(&first, &query).unwrap().estimate.dispatches;
    let second = GeometryProvider(512);
    assert_eq!(
        physical_work(&second, &query).unwrap().estimate.dispatches,
        Some(1)
    );
    assert_eq!(before, Some(2));
    assert_eq!(
        physical_work(&first, &query).unwrap().estimate.dispatches,
        before
    );
}

#[cfg(feature = "wgpu")]
#[test]
fn observation_accounts_for_complete_physical_chain_not_one_dispatch() {
    use runmat_accelerate_api::observation::ProviderObservationCapability;
    let provider = GeometryProvider::new(); // geometry only, not execution proof
    let mut query = ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new("fusion.elementwise"),
        family: ProviderOperationFamily::Fusion,
        inputs: vec![value(ProviderElementType::F32, ProviderResidency::Device)],
        outputs: vec![value(ProviderElementType::F32, ProviderResidency::Device)],
        workload: Default::default(),
    };
    let work = physical_work(&provider, &query).unwrap();
    assert!(work.queue_submissions > 1);
    let ProviderObservationCapability::Supported(overhead) =
        observation_capability(&provider, &query)
    else {
        panic!("audited route");
    };
    assert_eq!(overhead.submissions, work.queue_submissions);
    assert_eq!(overhead.clear_commands, work.queue_submissions);
    assert_eq!(overhead.map_callbacks, work.queue_submissions);
    assert_eq!(
        overhead.requested_peak_buffer_bytes,
        u64::from(work.queue_submissions) * 8
    );
    assert_eq!(
        overhead.wgpu_compute_query.unwrap().max_query_sets,
        work.queue_submissions
    );
    assert_eq!(work.compute_encoders, work.estimate.dispatches.unwrap() + 1);
    let mut strided = query.clone();
    strided.inputs[0].layout = ProviderLayout::Strided;
    let strided_work = physical_work(&provider, &strided).unwrap();
    let ProviderObservationCapability::Supported(strided_overhead) =
        observation_capability(&provider, &strided)
    else {
        panic!("audited strided route");
    };
    assert!(strided_work.queue_submissions > strided_work.estimate.dispatches.unwrap());
    assert_eq!(strided_overhead.submissions, strided_work.queue_submissions);
    assert_eq!(
        strided_overhead.wgpu_compute_query.unwrap().max_query_sets,
        strided_work.queue_submissions
    );
    assert_eq!(
        strided_work.queue_submissions,
        strided_work.compute_encoders + 1
    );
    let mut empty = query.clone();
    empty.inputs[0].shape = vec![1, 0];
    empty.outputs[0].shape = vec![1, 0];
    let ProviderObservationCapability::Supported(empty_overhead) =
        observation_capability(&provider, &empty)
    else {
        panic!("audited empty route");
    };
    assert_eq!(empty_overhead.submissions, 0);
    assert!(empty_overhead.wgpu_compute_query.is_none());
    query.operation = ProviderOperationIdentity::new("zeros");
    assert_eq!(
        observation_capability(&provider, &query),
        ProviderObservationCapability::Unsupported
    );
}

fn value(
    element_type: ProviderElementType,
    residency: ProviderResidency,
) -> ProviderRepresentation {
    ProviderRepresentation {
        element_type,
        residency,
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape: vec![1, 8],
    }
}

#[test]
fn uniform_single_abi_costs_no_widening_or_publication_adapters_on_double_owner() {
    use runmat_accelerate_api::{ProviderFusedElementwiseAbi, ProviderFusedElementwiseWorkload};
    let provider = GeometryProvider::new();
    let abi = ProviderFusedElementwiseAbi::BroadcastReal {
        precision: ProviderPrecision::F32,
    };
    for (residency, visits) in [
        (ProviderResidency::Device, 0),
        (ProviderResidency::Host, 0),
        (ProviderResidency::Host, 8),
    ] {
        let mut input = value(ProviderElementType::F32, residency);
        if residency == ProviderResidency::Device {
            input.layout = ProviderLayout::Strided;
        }
        let mut query = ProviderFeasibilityQuery {
            operation: ProviderOperationIdentity::new("fusion.elementwise"),
            family: ProviderOperationFamily::Fusion,
            inputs: vec![input],
            outputs: vec![value(ProviderElementType::F32, ProviderResidency::Device)],
            workload: ProviderWorkload {
                fused_elementwise: Some(ProviderFusedElementwiseWorkload {
                    abi,
                    host_materialization_elements: visits,
                    host_workspace_bytes: visits * 4,
                }),
                ..Default::default()
            },
        };
        let work = physical_work(&provider, &query).unwrap();
        assert_eq!(work.estimate.transient_bytes, Some(64));
        assert_eq!(work.estimate.output_bytes, Some(32));
        assert_eq!(work.estimate.dispatches, Some(1));
        assert_eq!(work.queue_submissions, 2);
        assert_eq!(work.adapter_element_visits, visits);
        assert_eq!(
            work.scratch_bytes,
            if residency == ProviderResidency::Host {
                32 + visits * 4
            } else {
                0
            }
        );
        query.inputs[0].element_type = ProviderElementType::F64;
        assert!(
            physical_work(&provider, &query).is_none(),
            "Double data is not authorized to narrow by a Single output ABI"
        );
    }
}

#[test]
fn output_cast_and_resident_widening_are_both_in_peak_and_dispatch_estimates() {
    let provider = GeometryProvider::new(); // F64 physical geometry; not an execution witness.
    let query = ProviderFeasibilityQuery {
        operation: runmat_accelerate_api::ProviderOperationIdentity::new("fusion.elementwise"),
        family: ProviderOperationFamily::Fusion,
        inputs: vec![value(ProviderElementType::F32, ProviderResidency::Device)],
        outputs: vec![
            value(ProviderElementType::F32, ProviderResidency::Device),
            value(ProviderElementType::F64, ProviderResidency::Device),
        ],
        workload: Default::default(),
    };
    let ProviderFeasibility::Supported { estimate } = validate(&provider, true, &query).unwrap()
    else {
        panic!("typed physical plan should be feasible")
    };
    // Borrowed F32 input32 + widened64 + two raw outputs128 + finalSingle32.
    assert_eq!(estimate.transient_bytes, Some(256));
    assert_eq!(estimate.output_bytes, Some(96));
    assert_eq!(estimate.dispatches, Some(3));
    assert!(matches!(
        validate(&provider, false, &query),
        Some(ProviderFeasibility::Rejected { .. })
    ));
}

#[test]
fn host_single_upload_is_counted_at_the_actual_native_f64_buffer_width() {
    let provider = GeometryProvider::new();
    let query = ProviderFeasibilityQuery {
        operation: runmat_accelerate_api::ProviderOperationIdentity::new("fusion.elementwise"),
        family: ProviderOperationFamily::Fusion,
        inputs: vec![value(ProviderElementType::F32, ProviderResidency::Host)],
        outputs: vec![value(ProviderElementType::F64, ProviderResidency::Device)],
        workload: Default::default(),
    };
    let ProviderFeasibility::Supported { estimate } = validate(&provider, true, &query).unwrap()
    else {
        panic!("native staging should be feasible")
    };
    assert_eq!(estimate.transient_bytes, Some(128));
    assert_eq!(estimate.output_bytes, Some(64));
    assert_eq!(estimate.dispatches, Some(1));
}

#[test]
fn broadcast_inputs_are_charged_at_source_size_but_incompatible_geometry_rejects() {
    let provider = GeometryProvider::new();
    let mut row = value(ProviderElementType::F64, ProviderResidency::Host);
    row.shape = vec![1, 8];
    let mut column = row.clone();
    column.shape = vec![4, 1];
    let mut output = value(ProviderElementType::F32, ProviderResidency::Device);
    output.shape = vec![4, 8];
    let mut query = ProviderFeasibilityQuery {
        operation: runmat_accelerate_api::ProviderOperationIdentity::new("fusion.elementwise"),
        family: ProviderOperationFamily::Fusion,
        inputs: vec![row, column],
        outputs: vec![output],
        workload: Default::default(),
    };
    let ProviderFeasibility::Supported { estimate } = validate(&provider, true, &query).unwrap()
    else {
        panic!("broadcast geometry should be feasible")
    };
    assert_eq!(estimate.transient_bytes, Some(64 + 32 + 256 + 128));
    assert_eq!(estimate.output_bytes, Some(128));
    query.inputs[0].shape = vec![2, 8];
    assert!(matches!(
        validate(&provider, true, &query),
        Some(ProviderFeasibility::Rejected { .. })
    ));
}

#[test]
fn strided_single_materialization_and_widening_have_distinct_scratch_allocations() {
    let provider = GeometryProvider::new();
    let mut input = value(ProviderElementType::F32, ProviderResidency::Device);
    input.layout = ProviderLayout::Strided;
    let query = ProviderFeasibilityQuery {
        operation: runmat_accelerate_api::ProviderOperationIdentity::new("fusion.elementwise"),
        family: ProviderOperationFamily::Fusion,
        inputs: vec![input],
        outputs: vec![value(ProviderElementType::F32, ProviderResidency::Device)],
        workload: Default::default(),
    };
    let work = physical_work(&provider, &query).unwrap();
    // Borrowed32 + materialized32 + widened64 + raw64 + final32.
    assert_eq!(work.estimate.transient_bytes, Some(224));
    assert_eq!(work.scratch_bytes, 160);
    assert_eq!(work.estimate.dispatches, Some(4));
    assert_eq!(work.queue_submissions, 7); // includes fused pipeline warmup
    let mut contiguous = query;
    contiguous.inputs[0].layout = ProviderLayout::ColumnMajorContiguous;
    let work = physical_work(&provider, &contiguous).unwrap();
    assert_eq!(work.scratch_bytes, 128);
    contiguous.inputs[0].element_type = ProviderElementType::F64;
    contiguous.inputs[0].layout = ProviderLayout::Strided;
    let work = physical_work(&provider, &contiguous).unwrap();
    assert_eq!(work.scratch_bytes, 64); // Raw output only; input stays borrowed.
    assert_eq!(work.estimate.dispatches, Some(2));
    assert_eq!(work.queue_submissions, 3); // includes fused pipeline warmup
}

#[test]
fn shared_resident_fact_preserves_lazy_transpose_layout() {
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![2, 3],
            runmat_accelerate_api::NumericElementType::F32,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .unwrap();
    let handle = allocation
        .handle
        .with_view(
            allocation
                .handle
                .view()
                .clone()
                .with_shape(vec![3, 2])
                .with_layout(
                    runmat_accelerate_api::allocation::GpuTensorLayout::Transpose(
                        runmat_accelerate_api::TransposeInfo {
                            base_rows: 2,
                            base_cols: 3,
                        },
                    ),
                ),
        )
        .unwrap();
    let fact = super::super::tensor_representation(
        &runmat_value::Value::GpuTensor(handle.clone()),
        ProviderPrecision::F64,
    )
    .unwrap();
    assert_eq!(fact.element_type, ProviderElementType::F32);
    assert_eq!(fact.shape, vec![3, 2]);
    assert_eq!(fact.layout, ProviderLayout::Strided);
    assert_eq!(allocation.handle.shape(), &[2, 3]);
    drop(handle);
    assert_eq!(allocation.probe.retirements(), 0);
    allocation
        .handle
        .allocation_lease()
        .validate_live()
        .unwrap();
    drop(allocation.handle);
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn adapter_visits_follow_selected_source_and_output_passes() {
    let provider = GeometryProvider::new();
    let mut query = ProviderFeasibilityQuery {
        operation: runmat_accelerate_api::ProviderOperationIdentity::new("fusion.elementwise"),
        family: ProviderOperationFamily::Fusion,
        inputs: vec![value(ProviderElementType::F32, ProviderResidency::Device)],
        outputs: vec![value(ProviderElementType::F32, ProviderResidency::Device)],
        workload: Default::default(),
    };
    assert_eq!(
        physical_work(&provider, &query)
            .unwrap()
            .adapter_element_visits,
        16
    ); // widen + output cast
    query.inputs[0].layout = ProviderLayout::Strided;
    assert_eq!(
        physical_work(&provider, &query)
            .unwrap()
            .adapter_element_visits,
        24
    ); // materialize + widen + cast
    query.inputs[0].element_type = ProviderElementType::F64;
    assert_eq!(
        physical_work(&provider, &query)
            .unwrap()
            .adapter_element_visits,
        8
    ); // native strided read, output cast only
    query.inputs[0].layout = ProviderLayout::ColumnMajorContiguous;
    query.inputs[0].residency = ProviderResidency::Host;
    query.inputs[0].shape = vec![1, 1];
    assert_eq!(
        physical_work(&provider, &query)
            .unwrap()
            .adapter_element_visits,
        9
    ); // scalar staging + full output cast
    query.outputs.push(query.outputs[0].clone());
    assert_eq!(
        physical_work(&provider, &query)
            .unwrap()
            .adapter_element_visits,
        17
    );
    for output in &mut query.outputs {
        output.shape = vec![0, 8];
    }
    let empty = physical_work(&provider, &query).unwrap();
    assert_eq!(empty.adapter_element_visits, 0);
    assert_eq!(empty.estimate.transient_bytes, Some(0));
    assert_eq!(empty.estimate.dispatches, Some(0));
}

#[test]
fn host_single_staging_counts_both_actual_conversion_passes_on_f32_abi() {
    struct SingleGeometry;
    impl AccelProvider for SingleGeometry {
        fn elementwise_geometry(
            &self,
        ) -> Option<runmat_accelerate_api::ProviderElementwiseGeometry> {
            Some(runmat_accelerate_api::ProviderElementwiseGeometry {
                workgroup_size: 512,
            })
        }
        fn precision(&self) -> ProviderPrecision {
            ProviderPrecision::F32
        }
        fn device_info(&self) -> String {
            "pure F32 geometry fixture".into()
        }
        fn upload(
            &self,
            _: &runmat_accelerate_api::HostTensorView,
        ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
            anyhow::bail!("geometry fixture cannot execute")
        }
        fn download<'a>(
            &'a self,
            _: &'a runmat_accelerate_api::GpuTensorHandle,
        ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
            Box::pin(async { anyhow::bail!("geometry fixture cannot execute") })
        }
    }
    let query = ProviderFeasibilityQuery {
        operation: runmat_accelerate_api::ProviderOperationIdentity::new("fusion.elementwise"),
        family: ProviderOperationFamily::Fusion,
        inputs: vec![value(ProviderElementType::F32, ProviderResidency::Host)],
        outputs: vec![value(ProviderElementType::F32, ProviderResidency::Device)],
        workload: Default::default(),
    };
    let work = physical_work(&SingleGeometry, &query).unwrap();
    assert_eq!(work.adapter_element_visits, 16); // materialize_f64 then upload's f32 conversion
    assert_eq!(work.estimate.dispatches, Some(1)); // host passes are not GPU kernels
}
use runmat_accelerate_api::ProviderWorkload;
