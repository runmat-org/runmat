//! Continuum Rayleigh reference, derived from the rod PDE, not assembled K/M/C.
//! For exp(+i omega t), E*(1+i omega beta) U'' + rho*(omega²-i omega alpha) U=0.
//! U(0)=0 and E*(1+i omega beta) U'(1)=T. Sine/cosine entire power series
//! evaluate the exact complex solution without selecting a square-root branch.
use super::*;
use runmat_analysis_core::{
    MechanicalDampingModel, RayleighMassCoefficient, RayleighStiffnessCoefficient,
};

pub(super) const ALPHA: f64 = 0.8;
pub(super) const BETA: f64 = 0.002;

pub(super) fn add_damping(model: &mut AnalysisModel) {
    model.materials[0].damping = Some(MechanicalDampingModel::Rayleigh {
        mass_coefficient: RayleighMassCoefficient::new(ALPHA),
        stiffness_coefficient: RayleighStiffnessCoefficient::new(BETA),
    });
}

fn multiply(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
}

fn divide(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    let norm = b[0] * b[0] + b[1] * b[1];
    [
        (a[0] * b[0] + a[1] * b[1]) / norm,
        (a[1] * b[0] - a[0] * b[1]) / norm,
    ]
}

fn wave_series(x: f64, omega: f64) -> ([f64; 2], [f64; 2]) {
    let q_squared = divide(
        [omega * omega, -omega * ALPHA],
        [YOUNGS, YOUNGS * omega * BETA],
    );
    let negative_qx_squared = q_squared.map(|value| -value * x * x);
    let mut sine_over_q = [x, 0.0];
    let mut cosine = [1.0, 0.0];
    let mut sine_term = sine_over_q;
    let mut cosine_term = cosine;
    for n in 1..64 {
        let twice = 2.0 * n as f64;
        sine_term = multiply(sine_term, negative_qx_squared).map(|v| v / (twice * (twice + 1.0)));
        cosine_term =
            multiply(cosine_term, negative_qx_squared).map(|v| v / ((twice - 1.0) * twice));
        for axis in 0..2 {
            sine_over_q[axis] += sine_term[axis];
            cosine[axis] += cosine_term[axis];
        }
    }
    (sine_over_q, cosine)
}

pub(super) fn reference(x: f64, k: f64) -> [f64; 2] {
    let omega = k * WAVE_SPEED;
    let modulus = [YOUNGS, YOUNGS * omega * BETA];
    divide(
        multiply(PHASOR, wave_series(x, omega).0),
        multiply(modulus, wave_series(1.0, omega).1),
    )
}

fn derivative(x: f64, k: f64) -> [f64; 2] {
    let omega = k * WAVE_SPEED;
    divide(
        multiply(PHASOR, wave_series(x, omega).1),
        multiply([YOUNGS, YOUNGS * omega * BETA], wave_series(1.0, omega).1),
    )
}

pub(super) fn field_error(
    mesh: &SolverMeshArtifact,
    k: f64,
    real: &[f64],
    imaginary: &[f64],
) -> f64 {
    super::field_error_with(mesh, real, imaginary, |x| reference(x, k))
}

#[test]
fn damped_solid_rod_continuum_phase_reaction_and_power_converge_for_both_orders() {
    for quadratic in [true, false] {
        qualify_meshes(quadratic, |cells| rod_mesh(cells, quadratic));
    }
}

pub(super) fn qualify_meshes(quadratic: bool, mesh_at: impl Fn(usize) -> SolverMeshArtifact) {
    let cases = [0.5, std::f64::consts::FRAC_PI_2, 2.0];
    let refinements: &[usize] = if quadratic {
        &[1, 2, 4]
    } else {
        &[2, 4, 8, 16]
    };
    let mut previous = [f64::INFINITY; 3];
    for &cells in refinements {
        let mesh = mesh_at(cells);
        let mut model = rod_model(&mesh, &cases);
        add_damping(&mut model);
        for (case, k) in cases.into_iter().enumerate() {
            let assembly = assemble(&model, &mesh, k);
            let omega = k * WAVE_SPEED;
            let solved = solve_direct_harmonic(
                &assembly.real,
                &assembly.imaginary_load,
                omega,
                DirectHarmonicPolicy {
                    relative_tolerance: policy().relative_tolerance,
                    restart: 256,
                    maximum_iterations: 8192,
                },
            )
            .unwrap();
            let error = field_error(
                &mesh,
                k,
                &solved.displacement_real,
                &solved.displacement_imaginary,
            );
            eprintln!("damped rod quadratic={quadratic}, cells={cells}, k={k}, error={error}");
            assert!(error < previous[case] / if quadratic { 5.0 } else { 2.5 });
            previous[case] = error;
            if cells == *refinements.last().unwrap() {
                assert!(error < if quadratic { 0.003 } else { 0.03 });
            }
            let physical = evaluate_harmonic_physical_state(
                &assembly.real,
                omega,
                &solved.displacement_real,
                &solved.displacement_imaginary,
                assembly.real.operator.load(),
                &assembly.imaginary_load,
            )
            .unwrap();
            assert!(
                physical
                    .balance
                    .closure_metrics(&physical.reaction)
                    .unwrap()
                    .1
                    < 1e-8
            );
            let expected_reaction = divide(PHASOR.map(|v| -v), wave_series(1.0, omega).1);
            let actual_reaction = [physical.reaction.real(), physical.reaction.imaginary()]
                .map(|values| values.chunks_exact(3).map(|v| v[0]).sum::<f64>());
            let reaction_error = (actual_reaction[0] - expected_reaction[0])
                .hypot(actual_reaction[1] - expected_reaction[1])
                / expected_reaction[0].hypot(expected_reaction[1]);
            if cells == *refinements.last().unwrap() {
                assert!(reaction_error < if quadratic { 0.003 } else { 0.03 });
            }
            let input = 0.5
                * omega
                * assembly
                    .real
                    .operator
                    .load()
                    .iter()
                    .zip(&assembly.imaginary_load)
                    .zip(
                        solved
                            .displacement_real
                            .iter()
                            .zip(&solved.displacement_imaginary),
                    )
                    .map(|((fr, fi), (ur, ui))| fi * ur - fr * ui)
                    .sum::<f64>();
            let damping = assembly.real.operator.operators().damping();
            let loss = 0.5
                * omega
                * omega
                * [&solved.displacement_real, &solved.displacement_imaginary]
                    .into_iter()
                    .map(|u| {
                        u.iter()
                            .zip(crate::algebra::spmv(damping, u).unwrap())
                            .map(|(u, cu)| u * cu)
                            .sum::<f64>()
                    })
                    .sum::<f64>();
            assert!(input > 0.0 && loss > 0.0);
            assert!((input / loss - 1.0).abs() < 1e-8);
            let tip = reference(1.0, k);
            let exact_power = 0.5 * omega * (PHASOR[1] * tip[0] - PHASOR[0] * tip[1]);
            // Independent continuum volume-loss quadrature, not FE matrix entries.
            let count = 1024;
            let integrated = (0..=count)
                .map(|n| {
                    let x = n as f64 / count as f64;
                    let u = reference(x, k);
                    let strain = derivative(x, k);
                    let density = 0.5
                        * omega
                        * omega
                        * (ALPHA * (u[0] * u[0] + u[1] * u[1])
                            + BETA * YOUNGS * (strain[0] * strain[0] + strain[1] * strain[1]));
                    density
                        * if n == 0 || n == count {
                            1.0
                        } else if n % 2 == 0 {
                            2.0
                        } else {
                            4.0
                        }
                })
                .sum::<f64>()
                / (3.0 * count as f64);
            assert!((integrated / exact_power - 1.0).abs() < 1e-9);
            if cells == *refinements.last().unwrap() {
                assert!((input / exact_power - 1.0).abs() < if quadratic { 0.005 } else { 0.05 });
            }
            assert!(solved.relative_residual <= policy().relative_tolerance);
        }
    }
}
