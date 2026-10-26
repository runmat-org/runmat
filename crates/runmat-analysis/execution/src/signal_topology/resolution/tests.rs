use super::*;
use runmat_analysis_core::{
    AuthoredSignalFieldLocation, BodyForceDensity, CoordinateFrame, GeneratorCallable,
    GeneratorDomain, GeneratorEffectPolicy, GeneratorInvocation, Signal,
    SignalApproximationEvidence, SignalExtrapolation, SignalGenerator, SignalId,
    SignalInterpolation, SignalPrecision, SignalProvenance, SignalRepresentation,
    SignalSamplingPolicy, Temperature,
};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

fn authored(thermal: bool, location: AuthoredSignalFieldLocation) -> PhysicalSignal {
    let source = Signal::<BodyForceDensity> {
        signal_id: SignalId("field".into()),
        shape: SignalShape::DeferredTopologyField {
            selector: AuthoredSignalTopologySelector::StudySolverMesh { location },
            component_count: if thermal { 1 } else { 3 },
        },
        frame: if thermal {
            None
        } else {
            Some(CoordinateFrame::Global)
        },
        representation: SignalRepresentation::Generator(SignalGenerator {
            invocation: GeneratorInvocation {
                callable: GeneratorCallable::ProjectFunction {
                    qualified_name: "fields.field".into(),
                },
                arguments: vec![],
                effects: GeneratorEffectPolicy::Pure,
            },
            domain: GeneratorDomain::TimeSeconds,
            sampling: SignalSamplingPolicy::CanonicalPoints(vec![0.0, 1.0]),
            interpolation: SignalInterpolation::Linear,
            extrapolation: SignalExtrapolation::Reject,
        }),
        precision: SignalPrecision::Real64,
        evidence: SignalApproximationEvidence {
            max_absolute_error: Some(0.0),
            max_relative_error: None,
            derivative_max_absolute_error: None,
            sample_count: 0,
        },
        provenance: SignalProvenance::Authored,
    };
    if thermal {
        PhysicalSignal::Temperature(Signal::<Temperature> {
            signal_id: source.signal_id,
            shape: source.shape,
            frame: source.frame,
            representation: match source.representation {
                SignalRepresentation::Generator(generator) => {
                    SignalRepresentation::Generator(generator)
                }
                _ => unreachable!(),
            },
            precision: source.precision,
            evidence: source.evidence,
            provenance: source.provenance,
        })
    } else {
        PhysicalSignal::BodyForceDensity(source)
    }
}

#[test]
fn deferred_fields_resolve_without_mutation_and_transport_exact_authored_mesh_lineage() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        for thermal in [false, true] {
            for location in [
                AuthoredSignalFieldLocation::Node,
                AuthoredSignalFieldLocation::Cell,
                AuthoredSignalFieldLocation::Element,
                AuthoredSignalFieldLocation::Face,
                AuthoredSignalFieldLocation::Edge,
            ] {
                let source = authored(thermal, location);
                let original = source.clone();
                assert_eq!(source.shape().realized_component_count(Some(100)), None);
                assert!(SignalTopologyBinding::derive(&source, &mesh).is_err());
                let resolved = SignalTopologyResolution::derive(&source, &mesh).unwrap();
                assert_eq!(source, original);
                resolved.validate_for(&source, &mesh).unwrap();
                resolved
                    .binding()
                    .validate_for(resolved.resolved(), &mesh)
                    .unwrap();
                assert!(resolved.binding().entity_count() > 0);
                assert_eq!(
                    resolved.binding().component_count(),
                    if thermal { 1 } else { 3 }
                );
                let bytes = serde_json::to_vec(&resolved).unwrap();
                let reopened: SignalTopologyResolution = serde_json::from_slice(&bytes).unwrap();
                reopened.validate_for(&source, &mesh).unwrap();
                assert_eq!(reopened.identity().unwrap(), resolved.identity().unwrap());
                assert!(SignalTopologyResolution::derive(resolved.resolved(), &mesh).is_err());
            }
        }
    }
}

#[test]
fn foreign_mesh_changed_selector_and_transported_substitution_reject() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let source = authored(false, AuthoredSignalFieldLocation::Node);
    let resolved = SignalTopologyResolution::derive(&source, &mesh).unwrap();
    let foreign = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
    assert!(resolved.validate_for(&source, &foreign).is_err());
    let changed = authored(false, AuthoredSignalFieldLocation::Face);
    assert!(resolved.validate_for(&changed, &mesh).is_err());
    let other = SignalTopologyResolution::derive(&changed, &mesh).unwrap();
    assert_ne!(resolved.identity().unwrap(), other.identity().unwrap());
    assert_ne!(
        resolved.authored_contract_digest(),
        other.authored_contract_digest()
    );
    let mut forged = resolved.clone();
    forged.authored_contract_digest = Digest::sha256("foreign authored input");
    assert!(forged.validate_for(&source, &mesh).is_err());
    forged = resolved.clone();
    forged.resolved = other.resolved.clone();
    assert!(forged.validate_for(&source, &mesh).is_err());
    let mut broken = mesh.clone();
    {
        let quantity_change = 1.0;
        let quantity_index = 0;
        let quantity_slot = &mut broken.topology.nodes[0].coordinates_m;
        let mut quantity_coordinates = quantity_slot.canonical_value();
        quantity_coordinates[quantity_index] += quantity_change;
        *quantity_slot = runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Position,
            _,
        >::new(quantity_coordinates);
    };
    assert!(SignalTopologyResolution::derive(&source, &broken).is_err());
}
