use super::*;
use runmat_analysis_core::*;

fn spec(id: &str, target: ObservationTarget) -> ObservationSpec {
    ObservationSpec {
        observation_id: ObservationId(id.into()),
        quantity: ObservationQuantity::Temperature,
        value_shape: ObservationValueShape::Scalar,
        target,
        source_location: FieldLocation::Node,
        output_location: FieldLocation::Node,
        recovery: RecoveryPolicy::Native,
        component: ObservationComponent::All,
        frame: CoordinateFrame::Global,
        sign_convention: SignConvention::Native,
        complex_representation: ComplexRepresentation::Real,
        sampling: SamplingPolicy::Final,
        reduction: None,
        storage: ObservationStoragePolicy {
            precision: SignalPrecision::Real64,
            compression: CompressionPolicy::Lossless,
            retention: RetentionPolicy::Canonical,
            partial_access: PartialAccessPolicy::ManifestAndChunks,
        },
        purpose: ObservationPurpose::Publication,
        cost: runmat_analysis_core::ObservationCostPolicy::UpperBound(ObservationCostEstimate {
            operation_count: 100_000,
            peak_resident_bytes: 1_000_000,
            stored_bytes: 1_000_000,
            transfer_bytes: 1_000_000,
            collective_count: 0,
        }),
    }
}

#[test]
fn original_interval_and_duplicate_history_schedules_fit_counted_heap() {
    let mut step = ProcedureStep::thermal_transient("counted-history");
    let sampling = SamplingPolicy::TimeIntervalSeconds {
        start: 0.,
        end: 1.,
        interval: 0.0001,
    };
    let mut request = spec("temperature", ObservationTarget::WholeModel);
    request.sampling = sampling;
    let mut duplicate = request.clone();
    duplicate.observation_id = ObservationId("duplicate".into());
    step.observations = vec![
        ObservationRequest::History(request),
        ObservationRequest::History(duplicate),
    ];
    let count = coordinate_count(&step).unwrap();
    assert_eq!(count, 20002);
    let bound = coordinate_resident_bytes(count).unwrap();
    let (samples, retained, peak) = crate::topology::measure_test_allocations(|| {
        required_bounded_time_history_samples_admitted(&step, 10001, bound)
    });
    let samples = samples.unwrap();
    assert_eq!(samples.len(), 10001);
    for (index, sample) in samples.iter().enumerate() {
        assert_eq!(*sample, index as f64 * 0.0001);
    }
    assert!(peak as u64 <= bound, "{peak} exceeds {bound}");
    assert_eq!(retained, 10001 * std::mem::size_of::<f64>());
    let (rejected, retained, peak) = crate::topology::measure_test_allocations(|| {
        required_bounded_time_history_samples_admitted(&step, 10001, bound - 1)
    });
    assert!(matches!(
        rejected,
        Err(ObservationError::Limit("history schedule resident bytes"))
    ));
    assert_eq!((retained, peak), (0, 0));
    assert!(required_bounded_time_history_samples_admitted(&step, 10000, bound).is_err());
}

#[test]
fn domain_and_sampling_rejections_preserve_original_history_contract() {
    let mut step = ProcedureStep::thermal_transient("invalid-history");
    let mut request = spec("temperature", ObservationTarget::WholeModel);
    for sampling in [
        SamplingPolicy::LoadFactors(vec![0., 1.]),
        SamplingPolicy::Final,
        SamplingPolicy::Adaptive {
            minimum_interval: 0.1,
            maximum_samples: 4,
            relative_change: 1e-3,
        },
    ] {
        request.sampling = sampling;
        step.observations = vec![ObservationRequest::History(request.clone())];
        assert!(required_time_history_samples(&step).is_err());
    }
}
