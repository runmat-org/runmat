//! Independent fixed/free rod modal response, not projected RunMat matrices as a reference.
//! For phi_n=sin(lambda_n x), lambda_n=(n+1/2)pi and unit rho/A/L:
//! modal mass=1/2, stiffness=E lambda_n²/2, force=T(-1)^n.
//! exp(+i omega t) and Rayleigh damping give the complex denominator below.
use super::*;

mod coercivity;
mod execution;

pub(super) fn first_mode_reference(x: f64, k: f64) -> [f64; 2] {
    let lambda = std::f64::consts::FRAC_PI_2;
    let omega = k * WAVE_SPEED;
    let denominator = [
        YOUNGS * lambda * lambda - omega * omega,
        omega * (damped::ALPHA + damped::BETA * YOUNGS * lambda * lambda),
    ];
    let norm = denominator[0].powi(2) + denominator[1].powi(2);
    let scale = 2.0 * (lambda * x).sin() / norm;
    [
        scale * (PHASOR[0] * denominator[0] + PHASOR[1] * denominator[1]),
        scale * (PHASOR[1] * denominator[0] - PHASOR[0] * denominator[1]),
    ]
}

fn projected_response(
    assembly: &ComplexStructuralAssemblySummary,
    basis: &crate::pipeline::modal_basis::ModalBasisArtifact,
    k: f64,
) -> (Vec<f64>, Vec<f64>) {
    let reduced = ModalReducedSystem::project(&assembly.real, basis).unwrap();
    let solved = solve_modal_harmonic(
        &reduced,
        k * WAVE_SPEED,
        &ModalHarmonicLoad {
            real: (reduced)
                .load_field(assembly.real.operator.load())
                .and_then(|field| (reduced).project_load(&field))
                .unwrap(),
            imaginary: (reduced)
                .load_field(&assembly.imaginary_load)
                .and_then(|field| (reduced).project_load(&field))
                .unwrap(),
        },
    )
    .unwrap();
    (
        (reduced)
            .expand_state(&solved.displacement_real)
            .map(|field| field.into_canonical_storage())
            .unwrap(),
        (reduced)
            .expand_state(&solved.displacement_imaginary)
            .map(|field| field.into_canonical_storage())
            .unwrap(),
    )
}

fn modal_basis(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
    requested: usize,
) -> crate::pipeline::modal_basis::ModalBasisArtifact {
    super::super::run_modal_with_options(
        &modal_model(model),
        ComputeBackend::Cpu,
        ModalSolveOptions {
            solver_mesh: Some(mesh.clone()),
            truncation: ModalTruncationPolicy::requested(requested),
            ..ModalSolveOptions::default()
        },
    )
    .unwrap()
    .basis
}

fn power(
    assembly: &ComplexStructuralAssemblySummary,
    k: f64,
    real: &[f64],
    imaginary: &[f64],
) -> f64 {
    let omega = k * WAVE_SPEED;
    let input = 0.5
        * omega
        * assembly
            .real
            .operator
            .load()
            .iter()
            .zip(&assembly.imaginary_load)
            .zip(real.iter().zip(imaginary))
            .map(|((fr, fi), (ur, ui))| fi * ur - fr * ui)
            .sum::<f64>();
    let loss = 0.5
        * omega.powi(2)
        * [real, imaginary]
            .into_iter()
            .map(|u| {
                u.iter()
                    .zip(
                        crate::algebra::spmv(assembly.real.operator.operators().damping(), u)
                            .unwrap(),
                    )
                    .map(|(u, cu)| u * cu)
                    .sum::<f64>()
            })
            .sum::<f64>();
    assert!(input > 0.0 && loss > 0.0);
    assert!((input / loss - 1.0).abs() < 1e-8);
    input
}

#[test]
fn damped_first_modal_tet4_field_phase_and_power_converge_to_analytic_mode() {
    first_mode_continuum(false);
}

#[test]
fn damped_first_modal_tet10_field_phase_and_power_converge_to_analytic_mode() {
    first_mode_continuum(true);
}

fn first_mode_continuum(quadratic: bool) {
    let cases = [0.5, std::f64::consts::FRAC_PI_2, 2.0];
    let refinements: &[usize] = if quadratic {
        &[1, 2, 4]
    } else {
        &[2, 4, 8, 16]
    };
    let mut previous = [f64::INFINITY; 3];
    for &cells in refinements {
        let mesh = rod_mesh(cells, quadratic);
        let mut model = rod_model(&mesh, &cases);
        damped::add_damping(&mut model);
        let basis = modal_basis(&model, &mesh, 1);
        assert_eq!(basis.modes().len(), 1);
        for (case, k) in cases.into_iter().enumerate() {
            let assembly = assemble(&model, &mesh, k);
            let (real, imaginary) = projected_response(&assembly, &basis, k);
            let error = field_error_with(&mesh, &real, &imaginary, |x| first_mode_reference(x, k));
            eprintln!("damped first mode quadratic={quadratic}, cells={cells}, k={k}, field error={error}");
            assert!(error < previous[case] / if quadratic { 5.0 } else { 2.5 });
            previous[case] = error;
            let input = power(&assembly, k, &real, &imaginary);
            let tip = first_mode_reference(1.0, k);
            let expected_power = 0.5 * k * WAVE_SPEED * (PHASOR[1] * tip[0] - PHASOR[0] * tip[1]);
            if cells == *refinements.last().unwrap() {
                assert!(error < if quadratic { 0.005 } else { 0.02 });
                assert!(
                    (input / expected_power - 1.0).abs() < if quadratic { 0.005 } else { 0.02 }
                );
            }
        }
    }
}

#[test]
fn damped_modal_truncation_agrees_with_direct_and_independent_continuum() {
    let cases = [0.5, std::f64::consts::FRAC_PI_2, 2.0];
    for quadratic in [false, true] {
        // Both physical meshes have100 free axial DOFs. Full retention isolates
        // modal reduction from spatial error; the independent PDE still bounds that error.
        let mesh = rod_mesh(if quadratic { 2 } else { 4 }, quadratic);
        let mut model = rod_model(&mesh, &cases);
        damped::add_damping(&mut model);
        let mut prior = [f64::INFINITY; 3];
        for requested in [1, 8, 100] {
            let basis = modal_basis(&model, &mesh, requested);
            for (case, k) in cases.into_iter().enumerate() {
                let assembly = assemble(&model, &mesh, k);
                let direct = solve_direct_harmonic(
                    &assembly.real,
                    &assembly.imaginary_load,
                    k * WAVE_SPEED,
                    policy(),
                )
                .unwrap();
                let (real, imaginary) = projected_response(&assembly, &basis, k);
                let difference = real
                    .iter()
                    .zip(&imaginary)
                    .zip(
                        direct
                            .displacement_real
                            .iter()
                            .zip(&direct.displacement_imaginary),
                    )
                    .map(|((r, i), (dr, di))| (r - dr).powi(2) + (i - di).powi(2))
                    .sum::<f64>()
                    .sqrt()
                    / direct
                        .displacement_real
                        .iter()
                        .zip(&direct.displacement_imaginary)
                        .map(|(r, i)| r * r + i * i)
                        .sum::<f64>()
                        .sqrt();
                assert!(difference < prior[case]);
                prior[case] = difference;
                let input = power(&assembly, k, &real, &imaginary);
                if requested == 100 {
                    assert!(difference < 1e-7);
                    let error = damped::field_error(&mesh, k, &real, &imaginary);
                    eprintln!(
                        "damped full modal quadratic={quadratic}, k={k}, continuum error={error}"
                    );
                    assert!(error < if quadratic { 0.10 } else { 0.25 });
                    let exact = damped::reference(1.0, k);
                    let expected =
                        0.5 * k * WAVE_SPEED * (PHASOR[1] * exact[0] - PHASOR[0] * exact[1]);
                    // Coarse-mesh power has its own independent, explicitly coarse envelope.
                    assert!((input / expected - 1.0).abs() < if quadratic { 0.15 } else { 0.40 });
                }
            }
        }
    }
}
