//! Canonical and reduced acoustic formulations.
//!
//! The canonical library entry uses physical Tet4/Tet10 operators and exact boundaries.
//! The graph-domain proxy remains separately named for its existing callers. Neither has
//! production qualification; runtime must not own or reproduce their numerical behavior.

mod canonical;
mod domain;
mod linear;
mod medium;
mod recovery;
mod reduced;
pub(crate) use canonical::validate_input as validate_canonical_input;

pub use canonical::{
    solve_canonical_harmonic, solve_canonical_modal, solve_canonical_transient,
    AcousticHarmonicError, AcousticHarmonicPolicy, AcousticModalPolicy, AcousticSolveCertificate,
    AcousticTransientPolicy, CanonicalAcousticModes, CanonicalAcousticResponse,
    CanonicalAcousticTransient, TransientRetention,
};

pub(crate) use reduced::nodal_sound_pressure_levels_db;
pub use reduced::solve_reduced_harmonic;

/// Returns whether the authored model contains a materialized nonzero pressure source that the
/// reduced acoustic formulation can consume.
pub fn has_nonzero_pressure_source(
    model: &runmat_analysis_core::AnalysisModel,
    mode_count: usize,
) -> Result<bool, String> {
    let node_count = domain::node_count(model);
    let frequency_hz = domain::drive_frequency_hz(mode_count, node_count);
    Ok(domain::pressure_sources(model, frequency_hz)?
        .iter()
        .any(|(_, real, imaginary)| {
            real.is_finite() && imaginary.is_finite() && real.hypot(*imaginary) > 0.0
        }))
}

#[cfg(test)]
mod tests {
    use super::{domain, linear};

    #[test]
    fn complex_graph_solve_matches_closed_form_real_system() {
        let system = domain::AcousticDomainSystem {
            diag_real: vec![4.0, 3.0],
            diag_imag: vec![0.0, 0.0],
            edges: vec![domain::AcousticGraphEdge {
                left: 0,
                right: 1,
                stiffness: 1.0,
            }],
        };
        let source_real = vec![2.0, 5.0];
        let source_imag = vec![0.0, 0.0];

        let (solution_real, solution_imag) =
            linear::solve_complex_graph_operator(&system, &source_real, &source_imag);

        assert!((solution_real[0] - 1.0).abs() <= 1.0e-12);
        assert!((solution_real[1] - 2.0).abs() <= 1.0e-12);
        assert!(solution_imag.iter().all(|value| value.abs() <= 1.0e-12));
        assert!(
            domain::residual_norm(
                &system,
                &solution_real,
                &solution_imag,
                &source_real,
                &source_imag,
            ) <= 1.0e-12
        );
    }

    #[test]
    fn graph_topology_has_deterministic_bounded_connectivity() {
        let topology = domain::topology(24);

        assert_eq!(topology.node_count, 24);
        assert_eq!(topology.active_dimension_count, 3);
        assert!(topology.edge_count > topology.node_count);
        assert!(topology.boundary_node_count <= topology.node_count);
        assert_eq!(topology.neighbor(0, 0, -1), None);
        assert_eq!(topology.neighbor(0, 0, 1), Some(1));
    }
}
