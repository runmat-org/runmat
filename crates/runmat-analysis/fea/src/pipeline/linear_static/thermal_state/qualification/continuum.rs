//! Solved-temperature continuum refinement; no discrete operator is used as its reference.
//!
//! On the unit cube: -k T''=q, T(0)=T(1)=Tref, insulated lateral faces.
//! With nu=0 and uy=uz=0, ux'=alpha*(T-Tref) is traction-free axial equilibrium.
//! Thus T-Tref=q*x*(1-x)/(2k), ux=alpha*q*(x²/2-x³/3)/(2k),
//! sigma_xx=0, sigma_yy=sigma_zz=-E*alpha*(T-Tref).
//! Lateral constraints are part of the advertised reference, not a free-expansion claim.

use super::*;
use crate::topology::{
    test_meshes::{quadratic_slab_mesh, slab_mesh},
    AnalysisProjection,
};

mod evidence;
mod execution;
mod fixture;

const CONDUCTIVITY: f64 = 1.0;
const GENERATION: f64 = 80.0;
const YOUNGS: f64 = 100.0;
const REFINEMENTS: [usize; 3] = [1, 2, 4];

fn excess_temperature(x: f64) -> f64 {
    GENERATION * x * (1.0 - x) / (2.0 * CONDUCTIVITY)
}

fn displacement(x: f64) -> f64 {
    ALPHA * GENERATION * (x * x / 2.0 - x * x * x / 3.0) / (2.0 * CONDUCTIVITY)
}

#[derive(Debug)]
struct Errors {
    temperature: f64,
    flux: f64,
    displacement: f64,
    stress: f64,
    energy: f64,
    reaction_moment: f64,
}

#[test]
fn solved_nonuniform_temperature_to_thermoelasticity_converges_to_independent_continuum() {
    for quadratic in [false, true] {
        let mut errors = Vec::new();
        for cells in REFINEMENTS {
            let mesh = if quadratic {
                quadratic_slab_mesh(cells)
            } else {
                slab_mesh(cells)
            };
            let model = fixture::model(&mesh);
            let heat = crate::run_thermal_with_options(
                &model,
                ComputeBackend::Cpu,
                ThermalSolveOptions {
                    procedure: ThermalProcedureControl::Steady,
                    residual_target: 1.0e-12,
                    solver_mesh: Some(mesh.clone()),
                    ..Default::default()
                },
            )
            .unwrap();
            let original =
                CanonicalSemanticState::from_thermal_result(&model, &mesh, &heat).unwrap();
            let bytes = original.canonical_bytes().unwrap();
            let cold = CanonicalSemanticState::from_canonical_bytes(&bytes).unwrap();
            assert_eq!(cold, original);
            let result = execution::solve(&model, &mesh, &cold);
            assert_eq!(cold.canonical_bytes().unwrap(), bytes);
            assert_eq!(result.thermal_transition.source_state, cold.identity);
            assert_eq!(
                result.linearization.thermal_source().unwrap().transition(),
                &result.thermal_transition
            );
            let error = evidence::measure(&mesh, &heat, &cold, &result);
            println!("thermoelastic parabolic heat: quadratic={quadratic} cells={cells} {error:?}");
            errors.push(error);
        }
        // The quadratic temperature and its affine heat flux are exact in Tet10. Its
        // cubic mechanical response is deliberately not representable on a coarse mesh.
        if quadratic {
            assert!(errors
                .iter()
                .all(|e| e.temperature < 1.0e-8 && e.flux < 1.0e-8));
            refinement(
                "Tet10 displacement",
                errors.iter().map(|e| e.displacement),
                0.17,
            );
            refinement("Tet10 mean stress", errors.iter().map(|e| e.stress), 0.45);
            refinement(
                "Tet10 elastic energy",
                errors.iter().map(|e| e.energy),
                0.10,
            );
            assert!(errors.iter().all(|e| e.reaction_moment < 1.0e-10));
            let finest = errors.last().unwrap();
            assert!(finest.displacement < 0.002);
            assert!(finest.stress < 0.005);
            assert!(finest.energy < 0.0005);
        } else {
            refinement(
                "Tet4 temperature",
                errors.iter().map(|e| e.temperature),
                0.26,
            );
            refinement("Tet4 mean heat flux", errors.iter().map(|e| e.flux), 0.51);
            refinement(
                "Tet4 displacement",
                errors.iter().map(|e| e.displacement),
                0.34,
            );
            refinement("Tet4 mean stress", errors.iter().map(|e| e.stress), 0.50);
            refinement("Tet4 elastic energy", errors.iter().map(|e| e.energy), 0.35);
            refinement(
                "Tet4 reaction moments",
                errors.iter().map(|e| e.reaction_moment),
                0.26,
            );
            let finest = errors.last().unwrap();
            assert!(finest.temperature < 0.05);
            assert!(finest.flux < 0.11);
            assert!(finest.displacement < 0.05);
            assert!(finest.stress < 0.13);
            assert!(finest.energy < 0.09);
            assert!(finest.reaction_moment < 0.07);
        }
    }
}

fn refinement(name: &str, values: impl Iterator<Item = f64>, maximum_ratio: f64) {
    let values = values.collect::<Vec<_>>();
    assert!(
        values.iter().all(|value| value.is_finite()),
        "{name}: {values:?}"
    );
    assert!(
        values
            .windows(2)
            .all(|pair| pair[1] < maximum_ratio * pair[0]),
        "{name} did not converge: {values:?}"
    );
}
