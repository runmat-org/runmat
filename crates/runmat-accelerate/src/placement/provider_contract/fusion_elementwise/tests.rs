use super::*;
use crate::simple_provider::InProcessProvider;

#[cfg(feature = "wgpu")]
#[test]
fn observation_accounts_for_complete_physical_chain_not_one_dispatch() {
    use runmat_accelerate_api::observation::ProviderObservationCapability;
    let provider = InProcessProvider::new(); // geometry only, not execution proof
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
fn output_cast_and_resident_widening_are_both_in_peak_and_dispatch_estimates() {
    let provider = InProcessProvider::new(); // F64 physical geometry; not an execution witness.
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
    let provider = InProcessProvider::new();
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
    let provider = InProcessProvider::new();
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
    let provider = InProcessProvider::new();
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
    assert_eq!(work.queue_submissions, 6);
    let mut contiguous = query;
    contiguous.inputs[0].layout = ProviderLayout::ColumnMajorContiguous;
    let work = physical_work(&provider, &contiguous).unwrap();
    assert_eq!(work.scratch_bytes, 128);
    contiguous.inputs[0].element_type = ProviderElementType::F64;
    contiguous.inputs[0].layout = ProviderLayout::Strided;
    let work = physical_work(&provider, &contiguous).unwrap();
    assert_eq!(work.scratch_bytes, 64); // Raw output only; input stays borrowed.
    assert_eq!(work.estimate.dispatches, Some(2));
    assert_eq!(work.queue_submissions, 2);
}

#[test]
fn shared_resident_fact_preserves_lazy_transpose_layout() {
    let handle = runmat_accelerate_api::GpuTensorHandle::new(vec![3, 2], u32::MAX - 286, 1)
        .with_numeric_descriptor(
            runmat_accelerate_api::NumericElementType::F32,
            runmat_accelerate_api::GpuTensorStorage::Real,
        );
    runmat_accelerate_api::record_handle_transpose(&handle, 2, 3);
    let fact = super::super::tensor_representation(
        &runmat_value::Value::GpuTensor(handle.clone()),
        ProviderPrecision::F64,
    )
    .unwrap();
    assert_eq!(fact.element_type, ProviderElementType::F32);
    assert_eq!(fact.shape, vec![3, 2]);
    assert_eq!(fact.layout, ProviderLayout::Strided);
    runmat_accelerate_api::clear_handle_metadata(&handle);
}

#[test]
fn adapter_visits_follow_selected_source_and_output_passes() {
    let provider = InProcessProvider::new();
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
        fn free(&self, _: &runmat_accelerate_api::GpuTensorHandle) -> anyhow::Result<()> {
            anyhow::bail!("geometry fixture cannot execute")
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
