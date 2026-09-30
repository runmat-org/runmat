//! Analytic curved extrusion, not a CAD meshing or general curved-solid claim.
//! F(x,y,z)=(x,y+eta*z*(1-z),z) is exactly quadratic with det(DF)=1.
//! Cross-sectional area is one and lateral n_x=0, so the nu=0 axial PDE
//! and the independent straight-rod references remain exact.
use super::*;
use runmat_meshing_core::{CanonicalMeshingContract, SolverNodeExactParameter, StableDigest};

const ETA: f64 = 0.4;

mod geometry;
mod undamped;

fn curved_mesh(cells: usize) -> SolverMeshArtifact {
    curved_mesh_with_eta(cells, ETA)
}

fn curved_mesh_with_eta(cells: usize, eta: f64) -> SolverMeshArtifact {
    let mut mesh = quadratic_slab_mesh(cells);
    for node in &mut mesh.topology.nodes {
        // Existing UV coordinates are the preimage chart coordinates. Bind each
        // chart to the analytic transformed evaluator, not the former planar one.
        for parameter in &mut node.exact_parameters {
            if let SolverNodeExactParameter::Surface { chart_id, .. } = parameter {
                let input = ("analytic-rod-quadratic-shear-chart", *chart_id, eta);
                let encoded = runmat_canonical_codec::encode_contract(
                    b"analytic-rod-chart/v1\0",
                    "analytic-rod-chart",
                    &input,
                    runmat_canonical_codec::CanonicalLimits::new(1024, 16, 256, 8),
                )
                .unwrap();
                *chart_id =
                    StableDigest::from_bytes(*runmat_execution::Digest::sha256(encoded).bytes());
            }
        }
        runmat_meshing_core::sort_solver_node_exact_parameters(&mut node.exact_parameters);
        let z = node.coordinates_m[2];
        node.coordinates_m[1] += eta * z * (1.0 - z);
    }
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();
    mesh.validate_canonical().unwrap();
    mesh
}

#[test]
fn curved_tet10_damped_continuum_preserves_phase_reaction_and_power() {
    // Same frequencies, refinement, residual, field/reaction/power tolerances
    // as the existing straight Tet10 qualification; no ordering/cap changes.
    super::damped::qualify_meshes(true, curved_mesh);
}

#[test]
fn curved_tet10_cold_execution_and_recovery_preserve_continuum_phase() {
    for damped in [false, true] {
        super::execution::cold_execution_on_mesh(
            curved_mesh(4),
            true,
            damped,
            Some(curved_mesh_with_eta(4, 0.41)),
        );
    }
}
