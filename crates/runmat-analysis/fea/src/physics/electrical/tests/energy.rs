//! Check the actual original operators under the original factor-work/storage envelope.
//! Use the source captured during the original quadrature pass; never re-integrate it.
use super::*;
use crate::algebra::{
    CoercivityCertificate, CoercivityWorkEnvelope, MatrixProperties, OperatorRole, RealSparseMatrix,
};

#[test]
fn original_electrical_operators_have_verified_anchored_local_energy_bounds() {
    for equation in [
        StaticElectricalEquation::Electrostatics,
        StaticElectricalEquation::SteadyConduction,
    ] {
        for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
            for cells in [2, 4, 8] {
                let mesh = mesh(order, cells);
                let model = model(equation);
                let projection = AnalysisProjection::derive(&mesh).unwrap();
                let properties =
                    crate::properties::compile_solid_property_map(&model, &mesh).unwrap();
                let mut assembly = super::super::assembly::ElectricalAssembly::assemble(
                    projection,
                    &properties,
                    equation,
                    None,
                )
                .unwrap();
                let scale = assembly
                    .coefficients
                    .canonical_values()
                    .iter()
                    .copied()
                    .reduce(f64::max)
                    .unwrap();
                let normalized = RealSparseMatrix::scaled_sum(
                    &[(scale.recip(), &assembly.stiffness)],
                    OperatorRole::Stiffness,
                    MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
                )
                .unwrap();
                let mut prescribed = vec![false; assembly.projection.nodes.len()];
                for face in &assembly.projection.boundary_faces {
                    for &node in &face.node_indices {
                        prescribed[node] = true;
                    }
                }
                let prepared = normalized.with_identity_constraints(&prescribed).unwrap();
                let envelope = CoercivityWorkEnvelope::derive(prepared.pattern()).unwrap();
                let certificate = CoercivityCertificate::prove_local_energy(
                    &prepared,
                    &prescribed,
                    assembly.local_energy.take().unwrap(),
                    scale.recip(),
                )
                .unwrap();
                certificate.validate_for(&prepared).unwrap();
                let (_, _, operations, resident) = certificate.work_counts();
                assert!(operations <= envelope.update_limit());
                let admitted = crate::algebra::anchored_energy::SourceEnergyDemand::derive(
                    envelope,
                    prescribed.len(),
                    prescribed.len() as u64,
                    assembly
                        .projection
                        .elements
                        .iter()
                        .map(|element| element.node_indices.len()),
                )
                .unwrap();
                assert!(resident <= admitted.resident_bytes());
                eprintln!(
                    "energy {equation:?}/{order:?}/cells{cells}: ops={operations},bytes={resident}"
                );
            }
        }
    }
}
