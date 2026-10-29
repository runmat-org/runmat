use super::*;
use crate::callable::executable::tests as fixture;
use runmat_analysis_core::{AuthoredSignalFieldLocation, AuthoredSignalTopologySelector};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

fn frozen() -> FrozenDeferredSignal {
    let authored = fixture::signal().with_shape(SignalShape::DeferredTopologyField {
        selector: AuthoredSignalTopologySelector::StudySolverMesh {
            location: AuthoredSignalFieldLocation::Node,
        },
        component_count: 1,
    });
    FrozenDeferredSignal::prepare(
        authored,
        fixture::revision(),
        fixture::callable(),
        fixture::envelope(&fixture::analysis()),
        BTreeSet::new(),
        8192,
    )
    .unwrap()
}

#[test]
fn frozen_intent_has_no_chunks_and_resolves_new_mesh_specific_materialization_identity() {
    let expected = frozen();
    let identity = expected.identity().unwrap();
    let reopened: FrozenDeferredSignal =
        serde_json::from_slice(&serde_json::to_vec(&expected).unwrap()).unwrap();
    assert_eq!(reopened.identity().unwrap(), identity);
    reopened.validate_for(&expected).unwrap();
    assert_eq!(
        reopened
            .authored()
            .shape()
            .realized_component_count(Some(4)),
        None
    );
    let mut materializations = Vec::new();
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let resolved = reopened.resolve(&mesh).unwrap();
        resolved
            .resolution()
            .validate_for(expected.authored(), &mesh)
            .unwrap();
        let prepared = resolved.prepared();
        prepared.validate_for_mesh(&mesh).unwrap();
        let expectation = prepared.expectation().unwrap();
        expected
            .validate_resolved_expectation(&expectation, &mesh)
            .unwrap();
        assert_ne!(
            prepared.admitted.materialization_identity,
            expected.admitted().materialization_identity
        );
        assert_eq!(
            prepared.admitted.generator_identity,
            expected.admitted().generator_identity
        );
        assert_eq!(prepared.admitted.provenance, expected.admitted().provenance);
        materializations.push(prepared.admitted.materialization_identity);
    }
    assert_ne!(materializations[0], materializations[1]);
    assert_eq!(expected.identity().unwrap(), identity);
}

#[test]
fn changed_frozen_admission_source_limits_and_foreign_resolved_expectations_reject() {
    let expected = frozen();
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let resolved = expected.resolve(&mesh).unwrap();
    let mut changed = expected.clone();
    let descriptor = changed.admitted.sample_points.descriptor();
    changed
        .admitted
        .sample_points
        .try_replace(
            1,
            runmat_types::physical::Quantity::try_from_descriptor(descriptor, 2.0).unwrap(),
        )
        .unwrap();
    assert!(changed.validate().is_err());
    for bound in [0, 4096, usize::MAX] {
        let mut changed = expected.clone();
        changed.maximum_inline_result_bytes = bound;
        assert!(changed.validate().is_err());
    }
    let mut changed = expected.clone();
    changed.maximum_inline_result_bytes = 8193;
    changed.validate().unwrap();
    assert_ne!(changed.identity().unwrap(), expected.identity().unwrap());
    assert!(changed.validate_for(&expected).is_err());
    let mut changed = expected.clone();
    changed.authored = changed.authored.with_shape(SignalShape::Scalar);
    assert!(changed.validate().is_err());
    let foreign = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
    let expectation = resolved.prepared().expectation().unwrap();
    assert!(expected
        .validate_resolved_expectation(&expectation, &foreign)
        .is_err());
    let mut substituted = expectation.clone();
    substituted
        .admitted
        .sample_points
        .try_replace(
            1,
            runmat_types::physical::Quantity::try_from_descriptor(
                substituted.admitted.sample_points.descriptor(),
                0.125,
            )
            .unwrap(),
        )
        .unwrap();
    assert!(expected
        .validate_resolved_expectation(&substituted, &mesh)
        .is_err());
    let mut substituted = expectation;
    substituted.admitted.materialization_identity = expected.admitted().materialization_identity;
    assert!(expected
        .validate_resolved_expectation(&substituted, &mesh)
        .is_err());
}
