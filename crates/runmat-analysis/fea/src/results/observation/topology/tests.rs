use super::*;
use crate::results::field::VolumeFieldSamplingRule;
use runmat_meshing_core::fixtures::{canonical_tetrahedron_solver_mesh, slab_mesh};
use runmat_meshing_core::ElementOrder;
use runmat_simulation_usage::{OperationCount, UsageContext, UsageLedger};
use std::sync::Arc;

#[test]
fn projection_and_observation_quotes_conserve_their_reference_batches() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let projection_quote = AnalysisProjection::derivation_work_upper_bound(&mesh).unwrap();
        let projection_ledger =
            Arc::new(UsageLedger::new(projection_quote, projection_quote).unwrap());
        UsageContext::new(Arc::clone(&projection_ledger))
            .run_sync(|| AnalysisProjection::derive(&mesh))
            .unwrap();
        assert_eq!(
            projection_ledger.snapshot().unwrap().completed.assembly,
            projection_quote
        );

        let observation_quote =
            structural_observation_derivation_work_upper_bound(Some(&mesh)).unwrap();
        let observation_ledger =
            Arc::new(UsageLedger::new(observation_quote, observation_quote).unwrap());
        let mut model =
            crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic);
        model.structural = None;
        UsageContext::new(Arc::clone(&observation_ledger))
            .run_sync(|| ObservationTopology::derive(&model, Some(&mesh)))
            .unwrap();
        let snapshot = observation_ledger.snapshot().unwrap();
        assert_eq!(snapshot.completed.assembly, observation_quote);
        assert_eq!(snapshot.reserved, OperationCount::ZERO);
        assert_eq!(
            observation_quote.operations() - projection_quote.operations(),
            if order == ElementOrder::Tet10 {
                4 * 116
            } else {
                0
            }
        );
    }
}

// witness: FEA-SHARED-OBSERVATIONS-001/quantity_location_and_cost_contract @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Tet4/Tet10 sampled topology retains original four/fifteen-point rule identities,
//   positions, edge IDs and integration indices and rejects mismatched topology/count; no
//   recovered-value convergence or global distributed reduction.
#[test]
fn sampled_observation_topology_retains_each_original_rule_order_and_identity() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let mut previous = None;
        for rule in [
            VolumeFieldSamplingRule::TetrahedronFourPoint,
            VolumeFieldSamplingRule::TetrahedronKeastFifteen,
        ] {
            let samples = SampledVolumeFieldTopology::derive(&mesh, rule).unwrap();
            let topology = ObservationTopology::from_sampled(&samples, &mesh).unwrap();
            assert!(topology.identity.is_none());
            assert_eq!(
                topology.digest(),
                samples.identity().unwrap().discretization_digest()
            );
            assert_eq!(topology.entity_counts(), samples.entity_counts());
            assert_eq!(topology.sampling.as_ref(), Some(&samples));
            assert_eq!(
                topology
                    .solid_projection
                    .as_ref()
                    .unwrap()
                    .projection_digest,
                samples.projection_digest()
            );
            assert_eq!(
                topology
                    .integration_indices(std::iter::once(0), rule.points_per_element())
                    .unwrap(),
                (0..rule.points_per_element()).collect::<Vec<_>>()
            );
            assert!(topology
                .integration_indices(std::iter::once(0), rule.points_per_element() - 1)
                .is_err());
            assert!(topology
                .integration_indices(std::iter::once(1), usize::MAX)
                .is_err());
            for (actual, source) in topology
                .volume_integration_points()
                .unwrap()
                .iter()
                .zip(samples.integration_samples_for(&mesh).unwrap())
            {
                assert_eq!(actual.identity, source.identity);
                assert_eq!(actual.centroid_m, source.position_m);
            }
            for (actual, source) in topology
                .elements(RecoveredFieldEntityDomain::OrientedEdges)
                .unwrap()
                .iter()
                .zip(samples.oriented_edges())
            {
                assert_eq!(actual.identity, source.identity);
            }
            if let Some(previous) = previous {
                assert_ne!(topology.digest(), previous);
            }
            previous = Some(topology.digest());
            assert!(ObservationTopology::from_sampled(&samples, &slab_mesh(1)).is_err());
        }
    }
}

#[test]
fn parent_expansion_counts_actual_samples_and_preserves_structural_legacy_identity() {
    let mesh = slab_mesh(1);
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    let structural = ObservationTopology::from_projection(&projection).unwrap();
    assert_eq!(
        structural.identity,
        Some(CanonicalStructuralTopologyIdentity::Solid {
            mesh_digest: projection.mesh_digest
        })
    );
    assert_eq!(structural.digest(), projection.projection_digest);
    assert_eq!(
        structural
            .integration_indices([2, 4].into_iter(), 8)
            .unwrap(),
        vec![8, 9, 10, 11, 16, 17, 18, 19]
    );
    assert!(structural
        .integration_indices([2, 4].into_iter(), 7)
        .is_err());
    assert!(structural
        .integration_indices(std::iter::once(usize::MAX), 4)
        .is_err());
    for (parent, element) in projection.elements.iter().enumerate() {
        for (ordinal, index) in structural
            .integration_indices(std::iter::once(parent), 4)
            .unwrap()
            .into_iter()
            .enumerate()
        {
            assert_eq!(
                structural.volume_integration_points().unwrap()[index].identity,
                integration_point_identity(element.stable_identity, ordinal)
            );
        }
    }
    let samples =
        SampledVolumeFieldTopology::derive(&mesh, VolumeFieldSamplingRule::TetrahedronKeastFifteen)
            .unwrap();
    let sampled = ObservationTopology::from_sampled(&samples, &mesh).unwrap();
    assert_eq!(
        sampled.integration_indices([2, 4].into_iter(), 30).unwrap(),
        (30..45).chain(60..75).collect::<Vec<_>>()
    );
    assert!(sampled.integration_indices([2, 4].into_iter(), 29).is_err());
}
