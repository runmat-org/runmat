//! Rejection preflight only: not a successful resolver, native allocation or whole-task bound.
#[path = "../support/peak_allocator/counting.rs"]
mod heap;

use runmat_analysis_core::*;
use runmat_analysis_fea::{
    results::observation::{resolve_observation_plan, ObservationError, ObservationLimits},
    topology::AnalysisProjection,
};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

fn path(count: usize) -> ObservationSpec {
    ObservationSpec {
        observation_id: ObservationId("path".into()),
        quantity: ObservationQuantity::Temperature,
        value_shape: ObservationValueShape::Scalar,
        target: ObservationTarget::Path {
            positions: vec![Position::new([0.0; 3]); count],
        },
        source_location: FieldLocation::Node,
        output_location: FieldLocation::Point,
        recovery: RecoveryPolicy::ShapeFunctionInterpolation,
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
        cost: ObservationCostPolicy::Automatic,
    }
}

// One test prevents overlapping binary-local heap measurement windows.
#[test]
fn excess_authored_counts_reject_without_allocating_topology_or_targets() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let projection =
            AnalysisProjection::derive(&canonical_tetrahedron_solver_mesh(order)).unwrap();
        for count in [16, 65_536] {
            let mut step = ProcedureStep::thermal_transient("preflight");
            step.observations = vec![ObservationRequest::Field(path(count))];
            for limits in [
                ObservationLimits {
                    maximum_selected_entities: 4,
                    ..ObservationLimits::default()
                },
                ObservationLimits {
                    maximum_observations: 0,
                    ..ObservationLimits::default()
                },
            ] {
                let baseline = heap::begin_measurement();
                let result = resolve_observation_plan(&step, &projection, &[], limits);
                let peak = heap::end_measurement().saturating_sub(baseline);
                assert!(matches!(result, Err(ObservationError::Limit(_))));
                assert_eq!(peak, 0, "authored count preflight must not allocate");
            }
        }
    }
}
