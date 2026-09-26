use super::*;
use crate::{
    excitation::{realize::realize_excitations, ExcitationSamplePoint},
    fixtures::{fixture_model, FixtureId},
};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

#[test]
fn scalar_projection_counts_unique_touched_nodes_even_after_cancellation() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let realized = realize_excitations(&model, None, ExcitationSamplePoint::Static).unwrap();
    let mut load = realized.iter().next().unwrap().clone();
    load.quantity = runmat_analysis_core::PhysicalQuantityKind::HeatGenerationDensity;
    load.kind = crate::excitation::RealizedExcitationKind::HeatGeneration(
        crate::excitation::ComplexScalar::real(0.0),
    );
    load.policy.projection.preserve_resultant = true;
    load.policy.projection.preserve_moment = false;
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let projection = AnalysisProjection::derive(&mesh).unwrap();
        let mut accumulator = ScalarProjectionAccumulator::new(
            &load,
            &projection,
            SpatialIntegrationRule::VolumeKeastFifteen,
            1,
        )
        .unwrap();
        assert_eq!(accumulator.resolved_nodes.len(), projection.nodes.len());
        accumulator.add_requested(0.0, 0.0);
        // Repeated quadrature contributions and zero weights still touch a node exactly once.
        accumulator.add_projected(0, 2.0, 1.0);
        accumulator.add_projected(0, -2.0, -1.0);
        accumulator.add_projected(projection.nodes.len() - 1, 0.0, 0.0);
        let evidence = accumulator.finish();
        assert_eq!(evidence.resolved_node_count, 2);
        assert!(evidence.accepted);
        let SpatialProjectionBalance::ScalarIntegral {
            projected_integral,
            relative_error,
            ..
        } = evidence.balance
        else {
            panic!("scalar balance")
        };
        assert_eq!(projected_integral.real, 0.0);
        assert_eq!(projected_integral.imaginary, 0.0);
        assert_eq!(relative_error, 0.0);
    }
}
