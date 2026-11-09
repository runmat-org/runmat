//! Actual serialized outputs must fit each typed reduction envelope.
use super::*;
use crate::planning::resource::*;

#[test]
fn typed_series_bounds_cover_statistics_extrema_and_all_ties() {
    let mut means = Vec::new();
    for ties in [
        ExtremaTiePolicy::FirstCanonical,
        ExtremaTiePolicy::AllLocations,
    ] {
        for operation in [
            SeriesReduction::Mean {
                weighting: SeriesWeighting::TrapezoidalIndependentDomain,
            },
            SeriesReduction::RootMeanSquare {
                weighting: SeriesWeighting::TrapezoidalIndependentDomain,
            },
            SeriesReduction::Variance {
                weighting: SeriesWeighting::TrapezoidalIndependentDomain,
            },
            SeriesReduction::Minimum,
            SeriesReduction::Maximum,
            SeriesReduction::AbsoluteMaximum,
            SeriesReduction::Envelope,
        ] {
            let (workload, mesh, plan, catalog, checkpoint) =
                fixture_with_ties(RetentionPolicy::Canonical, operation, ties);
            let ObservationFieldCatalogSource::Thermal { manifest } = catalog.source() else {
                unreachable!()
            };
            let context = ObservationHistorySourceContext::Thermal {
                workload: &workload,
                mesh: &mesh,
            };
            let request = ObservationHistoryFinalizationRequest::new(
                plan,
                manifest.producer,
                ObservationId("temperature_history".into()),
                &checkpoint,
                &catalog,
                context,
            );
            let request = request.unwrap();
            let bound = observation_history_series_reduction_payload_bytes_upper_bound(
                &request,
                &checkpoint,
            )
            .unwrap()
            .unwrap();
            let result = request.finalize(checkpoint, &catalog, context).unwrap();
            assert!(
                result
                    .series_reduction
                    .unwrap()
                    .canonical_bytes()
                    .unwrap()
                    .len() as u64
                    <= bound
            );
            if matches!(operation, SeriesReduction::Mean { .. }) {
                means.push(bound);
            }
        }
    }
    assert_eq!(means.len(), 2);
    assert_eq!(
        means[0], means[1],
        "mean has no extrema locations regardless of tie policy"
    );
}
