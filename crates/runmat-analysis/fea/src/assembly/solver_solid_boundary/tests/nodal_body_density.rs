use super::*;
use crate::excitation::{field::NodalVectorField, ExcitationSamplePoint};
use runmat_analysis_core::{
    BodyForceDensity, FieldLocation, Signal, SignalApproximationEvidence, SignalPrecision,
    SignalProvenance, SignalRepresentation, SignalShape, SignalValue,
};

fn signal(mesh: &SolverMeshArtifact) -> Signal<BodyForceDensity> {
    Signal {
        signal_id: SignalId("spatial-density".into()),
        shape: SignalShape::TopologyField {
            topology_id: runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes())
                .to_string(),
            location: FieldLocation::Node,
            component_count: 3,
        },
        frame: Some(CoordinateFrame::Global),
        representation: SignalRepresentation::Constant(SignalValue::Tensor {
            shape: vec![mesh.topology.nodes.len(), 3],
            values: mesh
                .topology
                .nodes
                .iter()
                .flat_map(|node| {
                    let [x, y, z] = node.coordinates_m.canonical_value();
                    [x, 2.0 * y, 3.0 * z].map(BodyForceDensity::new)
                })
                .collect(),
        }),
        precision: SignalPrecision::Real64,
        evidence: SignalApproximationEvidence {
            max_absolute_error: Some(0.0),
            max_relative_error: Some(0.0),
            derivative_max_absolute_error: None,
            sample_count: 1,
        },
        provenance: SignalProvenance::Authored,
    }
}

fn load(signal: &Signal<BodyForceDensity>) -> RealizedExcitation {
    let mut load = force();
    load.quantity = PhysicalQuantityKind::BodyForceDensity;
    load.target = RealizedExcitationTarget::WholeModel;
    load.policy.distribution = ExcitationDistribution::TopologyField;
    load.kind = RealizedExcitationKind::BodyForceField(
        NodalVectorField::sample(signal, ExcitationSamplePoint::Static, |value| {
            value.canonical_value()
        })
        .unwrap(),
    );
    load
}

#[test]
fn tet4_and_tet10_affine_density_preserve_continuum_force_moment_and_virtual_work() {
    for quadratic in [false, true] {
        for length in [1.0_f64, 2.0] {
            let mesh = mesh(quadratic, length);
            let layout = layout(&mesh);
            let mut rhs = vec![0.0; layout.total_dof_count()];
            let projected = apply_solver_mesh_structural_loads(
                &RealizedExcitations::new(vec![load(&signal(&mesh))]),
                &mesh,
                &constraints(&mesh),
                &layout,
                &mut rhs,
                true,
            )
            .unwrap();
            let mut resultant = [0.0; 3];
            let mut moment = [0.0; 3];
            let mut work = [0.0; 3];
            for (i, node) in mesh.topology.nodes.iter().enumerate() {
                let f = [
                    StructuralDofKind::Ux,
                    StructuralDofKind::Uy,
                    StructuralDofKind::Uz,
                ]
                .map(|kind| rhs[layout.index(i, kind).unwrap()]);
                let m = cross(node.coordinates_m.canonical_value(), f);
                for axis in 0..3 {
                    resultant[axis] += f[axis];
                    moment[axis] += m[axis];
                    work[axis] += node.coordinates_m.canonical_value()[axis] * f[axis];
                }
            }
            // Independent cube integrals, not production quadrature or assembled matrices:
            // b=(x,2y,3z); integral x=L^4/2, xy=L^5/4, x²=L^5/3.
            for (actual, expected) in resultant
                .into_iter()
                .zip([0.5, 1.0, 1.5].map(|v| v * length.powi(4)))
                .chain(
                    moment
                        .into_iter()
                        .zip([0.25, -0.5, 0.25].map(|v| v * length.powi(5))),
                )
                .chain(
                    work.into_iter()
                        .zip([1.0 / 3.0, 2.0 / 3.0, 1.0].map(|v| v * length.powi(5))),
                )
            {
                assert!(
                    (actual - expected).abs() < 1e-11 * expected.abs().max(1.0),
                    "{actual} != {expected}"
                );
            }
            assert!(projected.evidence[0].accepted);
        }
    }
}

#[test]
fn spatial_density_rejects_foreign_mesh_and_incomplete_rows_before_projection() {
    let mesh = mesh(false, 1.0);
    let mut signal = signal(&mesh);
    let field = load(&signal);
    let foreign = super::mesh(false, 2.0);
    let layout = layout(&foreign);
    assert!(apply_solver_mesh_structural_loads(
        &RealizedExcitations::new(vec![field]),
        &foreign,
        &constraints(&foreign),
        &layout,
        &mut vec![0.0; layout.total_dof_count()],
        true
    )
    .is_err());
    if let SignalRepresentation::Constant(SignalValue::Tensor { shape, values }) =
        &mut signal.representation
    {
        shape[0] -= 1;
        values.truncate(shape[0] * 3);
    }
    let layout = super::layout(&mesh);
    assert!(apply_solver_mesh_structural_loads(
        &RealizedExcitations::new(vec![load(&signal)]),
        &mesh,
        &constraints(&mesh),
        &layout,
        &mut vec![0.0; layout.total_dof_count()],
        true
    )
    .is_err());
}
