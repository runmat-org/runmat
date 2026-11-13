//! Actual generator placement/layout evidence on already authenticated canonical meshes.
//! Not authored mesh selection, whole-study resource admission or physics/restart qualification.
use super::*;
use runmat_analysis_core::{
    AuthoredSignalFieldLocation, AuthoredSignalTopologySelector, BodyForceDensity, FieldLocation,
    Temperature,
};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

pub(super) async fn run() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        for thermal in [false, true] {
            let width = if thermal { 1 } else { 3 };
            let nodes = mesh.topology.nodes.len();
            let shape = SignalShape::TopologyField {
                topology_id: Digest::from_bytes(*mesh.canonical_digest.bytes()).to_string(),
                location: FieldLocation::Node,
                component_count: width,
            };
            let (authored, explicit) = if thermal {
                field_signals(
                    shape,
                    nodes,
                    width,
                    Temperature::new,
                    PhysicalSignal::Temperature,
                )
            } else {
                field_signals(
                    shape,
                    nodes,
                    width,
                    BodyForceDensity::new,
                    PhysicalSignal::BodyForceDensity,
                )
            };
            let deferred = authored.with_shape(SignalShape::DeferredTopologyField {
                selector: AuthoredSignalTopologySelector::StudySolverMesh {
                    location: AuthoredSignalFieldLocation::Node,
                },
                component_count: width,
            });
            assert_eq!(deferred.shape().realized_component_count(Some(nodes)), None);
            runmat_analysis_core::validate_physical_signals(std::slice::from_ref(&authored))
                .unwrap();
            runmat_analysis_core::validate_physical_signals(std::slice::from_ref(&explicit))
                .unwrap();
            // Independently spell MATLAB's tail storage: component, then canonical node.
            let mut scale = Vec::new();
            for component in 0..width {
                for node in 0..nodes {
                    scale.push((10 * node + component + 1) as f64);
                }
            }
            let fixture =
                SignalFixture::prepare_named_field(deferred, explicit, scale, &mesh).await;
            assert_eq!(fixture.prepared.program.chunks.len(), 3);
            fixture
                .prepared
                .expectation()
                .unwrap()
                .validate_for_mesh(&mesh)
                .unwrap();
            let expected = canonical_tabulated_signal_content_digest(&fixture.explicit).unwrap();
            for (placement, realized) in [
                ("inline field", fixture.inline().await),
                ("isolated field process", fixture.isolated().await),
                ("durable field process", fixture.durable().await),
                (
                    "encrypted remote field task",
                    fixture.encrypted_remote().await,
                ),
            ] {
                assert_realization(&fixture, &realized, expected, placement);
            }
            eprintln!("field layout PASS order={order:?}, width={width}, nodes={nodes}, 3 chunks, 4 placements");
        }
    }
}

fn field_signals<Q: Clone>(
    shape: SignalShape,
    nodes: usize,
    width: usize,
    quantity: fn(f64) -> Q,
    wrap: fn(Signal<Q>) -> PhysicalSignal,
) -> (PhysicalSignal, PhysicalSignal) {
    let PhysicalSignal::Force(template) = generated_signal() else {
        unreachable!()
    };
    let SignalRepresentation::Generator(generator) = template.representation else {
        unreachable!()
    };
    let authored = Signal {
        signal_id: template.signal_id,
        shape,
        frame: if width == 3 {
            Some(CoordinateFrame::Global)
        } else {
            None
        },
        representation: SignalRepresentation::Generator(generator),
        precision: template.precision,
        evidence: template.evidence,
        provenance: template.provenance,
    };
    let mut explicit = authored.clone();
    explicit.evidence.sample_count = 5;
    explicit.representation = SignalRepresentation::Tabulated(TabulatedSignal {
        domain: runmat_analysis_core::TabulatedSignalDomain::TimeSeconds,
        knots: [0.0, 0.25, 0.5, 0.75, 1.0]
            .into_iter()
            .map(|time| {
                let mut values = Vec::new();
                for node in 0..nodes {
                    for component in 0..width {
                        values.push(quantity(time * (10 * node + component + 1) as f64));
                    }
                }
                runmat_analysis_core::SignalKnot::continuous(
                    time,
                    SignalOrdinate::Real(SignalValue::Tensor {
                        shape: vec![nodes, width],
                        values,
                    }),
                )
            })
            .collect(),
        interpolation: SignalInterpolation::Linear,
        extrapolation: SignalExtrapolation::Reject,
    });
    (wrap(authored), wrap(explicit))
}
