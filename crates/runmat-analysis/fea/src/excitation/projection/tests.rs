use super::*;
use crate::{
    excitation::{realize::realize_excitations, ExcitationSamplePoint},
    fixtures::{fixture_model, FixtureId},
};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

#[test]
fn repeated_collinear_forces_bound_the_rounded_resultant_and_moment() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let realized = realize_excitations(&model, None, ExcitationSamplePoint::Static).unwrap();
    let load = realized.iter().next().unwrap();
    let coordinates = [[1.0, 0.0, 0.0]];
    let mut accumulator = SpatialProjectionAccumulator::for_explicit_topology(
        load,
        runmat_execution::Digest::sha256(b"collinear-force-regression"),
        &coordinates,
        SpatialIntegrationRule::ExplicitConsistentElement,
        1,
        [0.0; 3],
    );
    for _ in 0..3 {
        let force = [0.0, 1.0 / 3.0, 1.0 / 3.0];
        accumulator.add_requested_force(coordinates[0], force);
        accumulator.add_projected_force(0, force);
    }
    let evidence = accumulator.finish();
    evidence.validate().unwrap();
    assert!(evidence.accepted);
    let SpatialProjectionBalance::VectorResultant {
        requested_resultant,
        projected_resultant,
        requested_moment,
        projected_moment,
        resultant_scale,
        moment_scale,
        resultant_relative_error,
        moment_relative_error,
    } = evidence.balance
    else {
        panic!("vector balance")
    };
    assert_eq!(requested_resultant.real, [0.0, 1.0, 1.0]);
    assert_eq!(projected_resultant, requested_resultant);
    assert_eq!(requested_moment.real, [0.0, -1.0, 1.0]);
    assert_eq!(projected_moment, requested_moment);
    assert_eq!(resultant_scale, 2.0_f64.sqrt());
    assert_eq!(moment_scale, 2.0_f64.sqrt());
    assert_eq!(resultant_relative_error, 0.0);
    assert_eq!(moment_relative_error, 0.0);
}

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

#[test]
fn complex_scalar_projection_bounds_the_rounded_integral_without_a_floor() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let realized = realize_excitations(&model, None, ExcitationSamplePoint::Static).unwrap();
    let mut load = realized.iter().next().unwrap().clone();
    load.policy.projection.preserve_moment = false;
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    for amplitude in [1.0, 1e-20] {
        let mut accumulator = ScalarProjectionAccumulator::new(
            &load,
            &projection,
            SpatialIntegrationRule::VolumeKeastFifteen,
            1,
        )
        .unwrap();
        for _ in 0..3 {
            accumulator.add_requested(amplitude / 3.0, amplitude / 3.0);
            accumulator.add_projected(0, amplitude / 3.0, amplitude / 3.0);
        }
        let evidence = accumulator.finish();
        evidence.validate().unwrap();
        assert!(evidence.accepted);
        let SpatialProjectionBalance::ScalarIntegral {
            requested_integral,
            projected_integral,
            normalization_scale,
            relative_error,
        } = evidence.balance
        else {
            panic!("scalar balance")
        };
        assert_eq!(requested_integral, projected_integral);
        assert!(normalization_scale >= requested_integral.real.hypot(requested_integral.imaginary));
        assert!(normalization_scale < 2.0 * amplitude);
        assert_eq!(relative_error, 0.0);
    }
}
