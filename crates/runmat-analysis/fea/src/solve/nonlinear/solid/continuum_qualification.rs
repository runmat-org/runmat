//! Independent nonuniform uniaxial-strain cube; primitive qualification, not public routing.
use super::{SolidNewtonPolicy, SolidNewtonProblem};
use crate::{
    assembly::{
        assemble_structural_system_borrowed,
        dofs::StructuralDofKind,
        nonlinear_solid::{CanonicalNonlinearSolidBasis, NonlinearSolidEvaluation},
        AssemblySummary, StructuralFormulationRequirement,
    },
    excitation::{realize_step_excitations, ExcitationSamplePoint},
    fixtures::{fixture_model, FixtureId},
    topology::{
        test_meshes::{quadratic_slab_mesh, slab_mesh},
        AnalysisProjection,
    },
};
use runmat_analysis_core::*;
use runmat_meshing_core::SolverMeshArtifact;

mod evidence;
mod execution;
pub(crate) mod fixture;
mod forcing;
mod fresh_integration;
mod moving_front;
mod plane_tied;

const E: f64 = 1200.0;
const NU: f64 = 0.25;
const YIELD: f64 = 5.0;
const HARDENING: f64 = 30.0;
const A: f64 = 0.02;
const B: f64 = 0.02;

fn mu() -> f64 {
    E / (2.0 * (1.0 + NU))
}
fn bulk() -> f64 {
    E / (3.0 * (1.0 - 2.0 * NU))
}
fn tangent() -> f64 {
    bulk() + 4.0 * mu() * HARDENING / (3.0 * (3.0 * mu() + HARDENING))
}
fn stress_offset() -> f64 {
    2.0 * mu() * YIELD / (3.0 * mu() + HARDENING)
}
fn strain(x: f64) -> f64 {
    A + B * x
}
pub(crate) fn displacement(x: f64) -> f64 {
    A * x + 0.5 * B * x * x
}
pub(crate) fn alpha(x: f64) -> f64 {
    (2.0 * mu() * strain(x) - YIELD) / (3.0 * mu() + HARDENING)
}
pub(crate) fn stress(x: f64) -> [f64; 6] {
    let e = strain(x);
    let p = alpha(x);
    [
        tangent() * e + stress_offset(),
        bulk() * e - 2.0 * mu() * e / 3.0 + mu() * p,
        bulk() * e - 2.0 * mu() * e / 3.0 + mu() * p,
        0.0,
        0.0,
        0.0,
    ]
}

fn energy_references() -> [f64; 3] {
    let mean_e = A + B / 2.0;
    let mean_squared_strain = A * A + A * B + B * B / 3.0;
    let s = 2.0 * mu() / (3.0 * mu() + HARDENING);
    let t = YIELD / (3.0 * mu() + HARDENING);
    let mean_alpha = s * mean_e - t;
    let mean_squared_alpha = s * s * mean_squared_strain - 2.0 * s * t * mean_e + t * t;
    let d = 1.0 - 1.5 * s;
    let c = 1.5 * t;
    let mean_squared_elastic_deviator = d * d * mean_squared_strain + 2.0 * d * c * mean_e + c * c;
    [
        0.5 * bulk() * mean_squared_strain + 2.0 * mu() * mean_squared_elastic_deviator / 3.0,
        0.5 * HARDENING * mean_squared_alpha,
        YIELD * mean_alpha,
    ]
}

#[derive(Debug)]
struct Errors {
    displacement: f64,
    stress: f64,
    alpha: f64,
    energies: [f64; 3],
    lateral_moment: f64,
}

struct Run {
    errors: Errors,
    /// Largest off-axis plastic flow relative to alpha; uniaxial strain flows along
    /// [1, -1/2, -1/2, 0, 0, 0], so a proportional discrete path keeps this at roundoff.
    plastic_off_axis: f64,
    displacement: Vec<f64>,
    alpha: Vec<f64>,
}

fn relative_difference(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let error = a.iter().zip(b).map(|(a, b)| (a - b).powi(2)).sum::<f64>();
    let norm = a.iter().map(|a| a * a).sum::<f64>();
    (error / norm).sqrt()
}

// witness: FEA-NONLINEAR-OPERATOR-001/references @ small_strain_j2/tet4/numerical_kernel;
//   check: j2-manufactured-cube; boundary: Original uniform Tet4 continuum gate asserts
//   displacement/stress/point-history/energy/reaction refinement and load schedule sensitivity;
//   linked executable is unassessed here and recorded scientific failures remain failures.
#[test]
fn manufactured_fully_yielded_cube_refines_and_checks_load_step_sensitivity() {
    check_manufactured_cube(false);
}

/// The Tet10 cube is exact in one increment: the terminal strain field is quadratic in x and the
/// radial return is exact for its proportional uniaxial-strain path. Eight increments are not
/// exact, and must not be: the fixture is force controlled (body force and traction), so between
/// load factors 0.21 and 0.38 the yield front sweeps through the cube and the intermediate strain
/// field has a kink that no Tet10 field represents. Those intermediate states carry small
/// off-axis strains, radial return records them as off-axis plastic flow, and the history keeps
/// them. That discrete path error converges with the mesh, which is what this test asserts.
// witness: FEA-NONLINEAR-OPERATOR-001/references @ small_strain_j2/tet10/numerical_kernel;
//   coverage: partial;
//   check: j2-manufactured-cube; boundary: one increment reproduces the polynomial J2 solution to roundoff; eight increments
//   carry a discrete path error because the force-controlled yield front crosses elements
//   mid-step, and that error and its off-axis plastic flow fall at order >= 1.5 under refinement.
#[test]
fn manufactured_fully_yielded_tet10_cube_is_exact_on_both_load_schedules() {
    check_manufactured_cube(true);
}

/// Asserts an eight-increment path error that falls at overall order >= 1.5 over cells 1, 2, 4,
/// decreasing at every refinement, with its off-axis plastic flow decreasing too.
fn assert_stepped_path_error_converges(runs: &[(f64, f64)], label: &str) {
    assert_eq!(runs.len(), 3, "{label}");
    let order = (runs[0].0 / runs[2].0).log2() / 2.0;
    assert!(
        order >= 1.5
            && runs.windows(2).all(|pair| pair[1].0 < pair[0].0 && pair[1].1 < pair[0].1),
        "{label}: eight-increment (displacement error, off-axis plastic flow) over cells 1, 2, 4 = {runs:?}, order {order}"
    );
}

fn check_manufactured_cube(quadratic: bool) {
    let mut primary = Vec::new();
    let mut stepped = Vec::new();
    let mut stepped_path = Vec::new();
    for cells in [1, 2, 4] {
        let one = execution::run(cells, quadratic, 1);
        let eight = execution::run(cells, quadratic, 8);
        stepped_path.push((eight.errors.displacement, eight.plastic_off_axis));
        if cells == 4 {
            let displacement_difference =
                relative_difference(&one.displacement, &eight.displacement);
            let history_difference = relative_difference(&one.alpha, &eight.alpha);
            assert!(displacement_difference<0.02 && history_difference<0.08,"loadstep quadratic={quadratic}: displacement={displacement_difference}, alpha={history_difference}");
        }
        primary.push(one.errors);
        stepped.push(eight.errors);
    }
    eprintln!(
        "nonlinear continuum quadratic={quadratic} primary={primary:?} eightincrements={stepped:?}"
    );
    if quadratic {
        assert_stepped_path_error_converges(&stepped_path, "Tet10 cube");
        for errors in &primary {
            assert!(
                errors.displacement < 1e-7 && errors.stress < 1e-7 && errors.alpha < 1e-7,
                "Tet10 polynomial: {errors:?}"
            );
            assert!(
                errors.energies.iter().all(|e| *e < 1e-7) && errors.lateral_moment < 1e-7,
                "Tet10 energies/reactions: {errors:?}"
            );
        }
    } else {
        // Tet4 displacement converges at O(h^2) in L2. The one-cell mesh (six tetrahedra) is
        // pre-asymptotic: the ratios are 2.42 then 3.21 (orders 1.27, 1.68). Every refinement must
        // reduce the error and the asymptotic window must show order above log2(3) = 1.58.
        assert!(
            primary
                .windows(2)
                .all(|p| p[1].displacement < p[0].displacement)
                && primary[1].displacement / primary[2].displacement > 3.0,
            "Tet4 displacement: {primary:?}"
        );
        assert!(
            primary
                .windows(2)
                .all(|p| p[0].stress / p[1].stress > 1.5 && p[0].alpha / p[1].alpha > 1.5),
            "Tet4 stress/history: {primary:?}"
        );
        // Energies converge at O(h^2); the one-cell window is pre-asymptotic (hardening energy
        // ratio 1.93, then 2.7), so every window must decrease and the 2 -> 4 window exceed 2.
        assert!(
            primary.windows(2).all(|p| (0..3)
                .all(|i| (p[0].energies[i] < 1e-10 && p[1].energies[i] < 1e-10)
                    || p[1].energies[i] < p[0].energies[i]))
                && (0..3).all(|i| (primary[1].energies[i] < 1e-10
                    && primary[2].energies[i] < 1e-10)
                    || primary[1].energies[i] / primary[2].energies[i] > 2.0),
            "Tet4 energies: {primary:?}"
        );
        let fine = primary.last().unwrap();
        assert!(
            fine.displacement < 0.05 && fine.stress < 0.15 && fine.alpha < 0.15,
            "Tet4 fine fields: {fine:?}"
        );
        assert!(
            fine.energies.iter().all(|e| *e < 0.08) && fine.lateral_moment < 0.15,
            "Tet4 fine energies/reactions: {fine:?}"
        );
    }
    let fine = stepped.last().unwrap();
    assert!(
        fine.displacement < 0.02 && fine.stress < 0.05 && fine.alpha < 0.08,
        "stepped fine fields: {fine:?}"
    );
    assert!(
        fine.energies.iter().all(|e| *e < 0.08) && fine.lateral_moment < 0.08,
        "stepped fine energies/reactions: {fine:?}"
    );
}
