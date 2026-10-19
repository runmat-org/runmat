use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, GpuTensorHandle, GpuTensorStorage,
    HostTensorView, NumericElementType, ProviderExtremaGeometry, ProviderExtremaInputBacking,
    ProviderExtremaWorkload, ProviderWorkload,
};

struct GeometryProvider {
    precision: ProviderPrecision,
    max_groups: u32,
    max_buffer_bytes: u64,
}

impl AccelProvider for GeometryProvider {
    fn precision(&self) -> ProviderPrecision {
        self.precision
    }

    fn extrema_geometry(&self) -> Option<ProviderExtremaGeometry> {
        Some(ProviderExtremaGeometry {
            max_buffer_bytes: self.max_buffer_bytes,
            max_binding_bytes: self.max_buffer_bytes,
            workgroup_size: 64,
            max_workgroups_per_dimension: self.max_groups,
        })
    }

    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("feasibility cannot upload")
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("feasibility cannot download") })
    }

    fn device_info(&self) -> String {
        "extrema geometry fixture".into()
    }
}

fn representation(class: ProviderElementType, shape: &[u64]) -> ProviderRepresentation {
    ProviderRepresentation {
        element_type: class,
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape: shape.to_vec(),
        residency: ProviderResidency::Device,
    }
}

fn query(
    kind: ProviderExtremaKind,
    demand: ProviderExtremaDemand,
    class: ProviderElementType,
    physical_class: NumericElementType,
) -> ProviderFeasibilityQuery {
    query_with_shape(kind, demand, class, physical_class, [2, 3])
}

fn query_with_shape(
    kind: ProviderExtremaKind,
    demand: ProviderExtremaDemand,
    class: ProviderElementType,
    physical_class: NumericElementType,
    shape: [u64; 2],
) -> ProviderFeasibilityQuery {
    let input = TestAllocationOwner::new()
        .allocate_zeroed(
            shape
                .into_iter()
                .map(|extent| usize::try_from(extent).expect("fixture extent fits usize"))
                .collect(),
            physical_class,
            GpuTensorStorage::Real,
        )
        .unwrap()
        .handle;
    let backing = ProviderExtremaInputBacking::from_full_live_handle(&input)
        .unwrap()
        .expect("full contiguous input");
    let index_class = ProviderElementType::F32;
    let mut outputs = vec![representation(class, &[1, shape[1]])];
    if demand == ProviderExtremaDemand::ValuesAndIndices {
        outputs.push(representation(index_class, &[1, shape[1]]));
    }
    ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new(identity(kind)),
        family: ProviderOperationFamily::Reduction,
        inputs: vec![representation(class, &shape)],
        outputs,
        workload: ProviderWorkload {
            extrema: Some(ProviderExtremaWorkload {
                kind,
                axis: 0,
                demand,
                input_backing: Some(backing),
            }),
            requested_outputs: Some(match demand {
                ProviderExtremaDemand::ValuesOnly => 1,
                ProviderExtremaDemand::ValuesAndIndices => 2,
            }),
            ..ProviderWorkload::default()
        },
    }
}

fn supported(result: ProviderFeasibility) -> ProviderResourceEstimate {
    let ProviderFeasibility::Supported { estimate } = result else {
        panic!("expected supported physical route, got {result:?}");
    };
    estimate
}

fn rejected(result: ProviderFeasibility) -> ProviderRejectionCode {
    let ProviderFeasibility::Rejected { rejection } = result else {
        panic!("expected rejection, got {result:?}");
    };
    rejection.code
}

#[test]
fn values_only_still_accounts_for_two_physical_float_outputs() {
    let provider = GeometryProvider {
        precision: ProviderPrecision::F32,
        max_groups: 1,
        max_buffer_bytes: 4096,
    };
    let one = query(
        ProviderExtremaKind::Minimum,
        ProviderExtremaDemand::ValuesOnly,
        ProviderElementType::F32,
        NumericElementType::F32,
    );
    let estimate = supported(validate(&provider, true, &one).unwrap());
    assert_eq!(estimate.output_bytes, Some(12));
    assert_eq!(estimate.transient_bytes, Some(64));
    assert_eq!(estimate.dispatches, Some(1));
    let mut pair = one.clone();
    pair.outputs
        .push(representation(ProviderElementType::F32, &[1, 3]));
    pair.workload.requested_outputs = Some(2);
    pair.workload.extrema.as_mut().unwrap().demand = ProviderExtremaDemand::ValuesAndIndices;
    let estimate = supported(validate(&provider, true, &pair).unwrap());
    assert_eq!(estimate.output_bytes, Some(24));
    assert_eq!(estimate.transient_bytes, Some(64));
}

#[cfg(feature = "wgpu")]
#[test]
fn optional_query_cost_excludes_index_readback_submission() {
    use runmat_accelerate_api::observation::ProviderObservationCapability;

    let provider = GeometryProvider {
        precision: ProviderPrecision::F32,
        max_groups: 1,
        max_buffer_bytes: 4096,
    };
    for (demand, readback_submissions) in [
        (ProviderExtremaDemand::ValuesOnly, 0),
        (ProviderExtremaDemand::ValuesAndIndices, 1),
    ] {
        let query = query(
            ProviderExtremaKind::Minimum,
            demand,
            ProviderElementType::F32,
            NumericElementType::F32,
        );
        let compute_encoders = supported(validate(&provider, true, &query).unwrap())
            .dispatches
            .unwrap();
        let ProviderObservationCapability::Supported(overhead) =
            observation_capability(&provider, &query)
        else {
            panic!("audited extrema route");
        };
        assert_eq!(
            overhead.submissions,
            compute_encoders + readback_submissions
        );
        let optional = overhead.wgpu_compute_query.unwrap();
        assert_eq!(optional.max_compute_encoders, compute_encoders);
        assert_eq!(optional.max_total_encoders, overhead.submissions);
        assert_eq!(optional.max_query_sets, overhead.submissions);
        assert_eq!(
            optional.max_timestamp_slots,
            u64::from(overhead.submissions) * 18
        );
    }
}

#[test]
fn native_wide_integer_value_and_float_index_have_distinct_physical_bytes() {
    let provider = GeometryProvider {
        precision: ProviderPrecision::F32,
        max_groups: 1,
        max_buffer_bytes: 4096,
    };
    let pair = query(
        ProviderExtremaKind::Maximum,
        ProviderExtremaDemand::ValuesAndIndices,
        ProviderElementType::U64,
        NumericElementType::U64,
    );
    let estimate = supported(validate(&provider, true, &pair).unwrap());
    assert_eq!(estimate.output_bytes, Some(36));
    assert_eq!(estimate.transient_bytes, Some(116));
    let mut wrong = pair.clone();
    wrong.outputs[1].element_type = ProviderElementType::F64;
    assert_eq!(
        rejected(validate(&provider, true, &wrong).unwrap()),
        ProviderRejectionCode::InvalidShape
    );
    wrong = pair.clone();
    wrong.inputs[0].shape = vec![3, 2];
    assert_eq!(
        rejected(validate(&provider, true, &wrong).unwrap()),
        ProviderRejectionCode::UnknownResourceEstimate
    );
    wrong = serde_json::from_slice(&serde_json::to_vec(&pair).unwrap()).unwrap();
    assert_eq!(
        rejected(validate(&provider, true, &wrong).unwrap()),
        ProviderRejectionCode::UnknownResourceEstimate
    );
}

#[test]
fn large_borrowed_source_is_in_complete_peak_but_not_new_allocation() {
    let provider = GeometryProvider {
        precision: ProviderPrecision::F32,
        max_groups: 1,
        max_buffer_bytes: 8 * 1024 * 1024,
    };
    let query = query_with_shape(
        ProviderExtremaKind::Minimum,
        ProviderExtremaDemand::ValuesOnly,
        ProviderElementType::F32,
        NumericElementType::F32,
        [262_145, 4],
    );
    let borrowed = query
        .workload
        .extrema
        .as_ref()
        .and_then(|route| route.input_backing)
        .and_then(ProviderExtremaInputBacking::live_requested_bytes)
        .expect("authentic borrowed source extent");
    assert_eq!(borrowed, 4_194_320);
    let estimate = supported(validate(&provider, true, &query).unwrap());
    assert_eq!(estimate.output_bytes, Some(16));
    assert_eq!(estimate.transient_bytes, Some(4_194_368));
    assert_eq!(
        estimate.transient_bytes.unwrap().checked_sub(borrowed),
        Some(48)
    );
}

#[test]
fn complete_peak_overflow_cannot_wrap_into_a_small_demand() {
    assert_eq!(complete_peak(24, 24, 16), Some(64));
    assert_eq!(complete_peak(u64::MAX, 24, 16), None);
    assert_eq!(complete_peak(24, u64::MAX, 16), None);
    assert_eq!(complete_peak(24, 24, u64::MAX), None);
}
