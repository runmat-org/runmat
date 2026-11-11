//! Continuum and typed-execution qualification, independent of discrete K/M entries.
//! Reference wave equation and fixed/free spectrum: UT Austin, equations 5.15–5.22:
//! https://farside.ph.utexas.edu/teaching/315/Waves/node34.html
//! Reflecting its coordinate gives u(0)=0. Applying E u'(1)=T gives
//! u(x)=T sin(k x)/(E k cos(k)), k=omega sqrt(rho/E).

use runmat_analysis_core::{
    AnalysisConstraint, AnalysisModel, CoordinateFrame, Displacement, IndependentDomain,
    MaterialElasticityModel, PhysicalSignal, PrescribedStructuralDofs, PrescribedValue,
    ProcedureStep, PropertyRegionTarget, RegionTarget, SignalExtrapolation, SignalInterpolation,
    SignalKnot, SignalOrdinate, SignalPrecision, SignalRepresentation, SignalValue,
    StructuralConstraint, StructuralConstraintKind, StructuralTranslationDof, TabulatedSignal,
    TabulatedSignalDomain, Traction, TranslationDofValue,
};
use runmat_meshing_core::SolverMeshArtifact;

use crate::{
    assembly::{
        complex_structural::{
            assemble_complex_structural_system, ComplexStructuralAssemblySummary,
        },
        StructuralFormulationRequirement,
    },
    fixtures::{
        fixture_model, replace_fixture_excitations, FixtureExcitation, FixtureExcitationKind,
        FixtureId,
    },
    pipeline::modal_governance::ModalTruncationPolicy,
    solve::dynamics::{
        evaluate_harmonic_physical_state, solve_direct_harmonic, solve_modal_harmonic,
        DirectHarmonicPolicy, ModalHarmonicLoad, ModalReducedSystem,
    },
    topology::test_meshes::{quadratic_slab_mesh, slab_mesh},
    ComputeBackend, ModalSolveOptions,
};

const YOUNGS: f64 = 100.0;
pub(crate) const WAVE_SPEED: f64 = 10.0;
const PHASOR: [f64; 2] = [1.0, 0.4];

mod curved;
mod damped;
mod damped_modal;
mod execution;

pub(crate) fn rod_model(mesh: &SolverMeshArtifact, wave_numbers: &[f64]) -> AnalysisModel {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.material_assignments[0].target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("slab"));
    let MaterialElasticityModel::IsotropicLinear(elasticity) =
        model.materials[0].elasticity.as_mut().unwrap()
    else {
        panic!("expected isotropic material");
    };
    elasticity.youngs_modulus_pa = runmat_types::physical::YoungsModulus::new(YOUNGS);
    // With nu=0, u=(u(x),0,0) is an exact three-dimensional solid solution.
    // Holding lateral translations excludes bending/shear modes, not axial strain.
    elasticity.poisson_ratio = runmat_types::physical::PoissonRatio::new(0.0);
    model.materials[0].mass_density_kg_per_m3 =
        (Some(1.0)).map(runmat_types::physical::MassDensity::new);
    model.materials[0].damping = None;
    model.constraints = mesh
        .topology
        .nodes
        .iter()
        .map(|node| {
            let translations = [
                StructuralTranslationDof::Ux,
                StructuralTranslationDof::Uy,
                StructuralTranslationDof::Uz,
            ]
            .into_iter()
            .filter(|dof| {
                node.coordinates_m.canonical_value()[0] == 0.0
                    || *dof != StructuralTranslationDof::Ux
            })
            .map(|dof| TranslationDofValue {
                dof,
                value: PrescribedValue::Constant(Displacement::new(0.0)),
            })
            .collect();
            AnalysisConstraint::Structural(StructuralConstraint {
                constraint_id: format!("rod_support_{}", node.node_id),
                frame: CoordinateFrame::Global,
                kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
                    target: RegionTarget::point(format!("node:{}", node.node_id)),
                    translations,
                    rotations: Vec::new(),
                }),
            })
        })
        .collect();
    let frequencies = wave_numbers
        .iter()
        .map(|k| k * WAVE_SPEED / std::f64::consts::TAU)
        .collect::<Vec<_>>();
    let mut step = ProcedureStep::structural_harmonic("rod_response");
    step.independent_domain = IndependentDomain::FrequencyHertz(frequencies.clone());
    model.sequence.steps = vec![step];
    replace_fixture_excitations(
        &mut model,
        vec![FixtureExcitation {
            id: "end_traction".into(),
            region_id: "slab_right".into(),
            kind: FixtureExcitationKind::Traction([PHASOR[0], 0.0, 0.0]),
        }],
    );
    let PhysicalSignal::Traction(signal) = &mut model.signals[0] else {
        unreachable!()
    };
    signal.precision = SignalPrecision::Complex128;
    signal.evidence.sample_count = frequencies.len();
    signal.representation = SignalRepresentation::Tabulated(TabulatedSignal {
        domain: TabulatedSignalDomain::FrequencyHertz,
        knots: frequencies
            .into_iter()
            .map(|f| {
                SignalKnot::continuous(
                    f,
                    SignalOrdinate::Complex {
                        real: SignalValue::Vector3([PHASOR[0], 0.0, 0.0].map(Traction::new)),
                        imaginary: SignalValue::Vector3([PHASOR[1], 0.0, 0.0].map(Traction::new)),
                    },
                )
            })
            .collect(),
        interpolation: SignalInterpolation::Linear,
        extrapolation: SignalExtrapolation::Reject,
    });
    model
}

pub(crate) fn assemble(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
    k: f64,
) -> ComplexStructuralAssemblySummary {
    assemble_complex_structural_system(
        model,
        &model.sequence.steps[0],
        k * WAVE_SPEED / std::f64::consts::TAU,
        Some(mesh.clone()),
        StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .unwrap()
}

fn policy() -> DirectHarmonicPolicy {
    DirectHarmonicPolicy {
        relative_tolerance: 1.0e-10,
        restart: 64,
        maximum_iterations: 4096,
    }
}

fn reference(x: f64, k: f64) -> [f64; 2] {
    let compliance = (k * x).sin() / (YOUNGS * k * k.cos());
    PHASOR.map(|t| t * compliance)
}

fn field_error(mesh: &SolverMeshArtifact, k: f64, real: &[f64], imaginary: &[f64]) -> f64 {
    field_error_with(mesh, real, imaginary, |x| reference(x, k))
}

fn field_error_with(
    mesh: &SolverMeshArtifact,
    real: &[f64],
    imaginary: &[f64],
    reference: impl Fn(f64) -> [f64; 2],
) -> f64 {
    let mut error = 0.0;
    let mut norm = 0.0;
    for (index, node) in mesh.topology.nodes.iter().enumerate() {
        let expected = reference(node.coordinates_m.canonical_value()[0]);
        error +=
            (real[3 * index] - expected[0]).powi(2) + (imaginary[3 * index] - expected[1]).powi(2);
        norm += expected[0].powi(2) + expected[1].powi(2);
        for component in [1, 2] {
            assert_eq!(real[3 * index + component], 0.0);
            assert_eq!(imaginary[3 * index + component], 0.0);
        }
    }
    (error / norm).sqrt()
}

// witness: FEA-DYNAMICS-RESPONSE-001/direct_modal_procedures @
//   linear_structural_dynamics/tet4/numerical_kernel;
//   check: axial-rod-harmonic; boundary: Original axial Tet4 rod harmonic traction matrix at three frequencies on both sides
//   of resonance; independent complex displacement/reaction refinement and phase resultant checks.
//   Not transient, spectrum or random-response evidence.
#[test]
fn solid_rod_harmonic_face_traction_converges_to_independent_continuum() {
    harmonic_continuum(false);
}

#[test]
fn solid_rod_tet10_harmonic_face_traction_converges_to_independent_continuum() {
    harmonic_continuum(true);
}

fn harmonic_continuum(quadratic: bool) {
    let wave_numbers = [0.5, 1.0, 2.0]; // Both sides of first resonance pi/2, never exactly at it.
    let mut previous = [f64::INFINITY; 3];
    let mut previous_reaction = [f64::INFINITY; 3];
    let refinements = if quadratic { [1, 2, 4] } else { [2, 4, 8] };
    let reduction = if quadratic { 6.0 } else { 2.5 };
    let finest_bound = if quadratic { 0.002 } else { 0.02 };
    for cells in refinements {
        let mesh = rod_mesh(cells, quadratic);
        let model = rod_model(&mesh, &wave_numbers);
        for (case, k) in wave_numbers.into_iter().enumerate() {
            let assembly = assemble(&model, &mesh, k);
            let real_load = assembly.real.operator.load();
            // Unit end area: exact resultant and moment about the origin, in each phase.
            for (load, phase) in [
                (real_load, PHASOR[0]),
                (&assembly.imaginary_load[..], PHASOR[1]),
            ] {
                let resultant: f64 = load.chunks_exact(3).map(|v| v[0]).sum();
                let moments = mesh.topology.nodes.iter().zip(load.chunks_exact(3)).fold(
                    [0.0; 2],
                    |mut m, (node, v)| {
                        m[0] += node.coordinates_m.canonical_value()[2] * v[0];
                        m[1] -= node.coordinates_m.canonical_value()[1] * v[0];
                        m
                    },
                );
                assert!((resultant - phase).abs() < 1.0e-12);
                assert!((moments[0] - phase * 0.5).abs() < 1.0e-12);
                assert!((moments[1] + phase * 0.5).abs() < 1.0e-12);
            }
            let omega = k * WAVE_SPEED;
            let response =
                solve_direct_harmonic(&assembly.real, &assembly.imaginary_load, omega, policy())
                    .unwrap();
            let error = field_error(
                &mesh,
                k,
                &response.displacement_real,
                &response.displacement_imaginary,
            );
            eprintln!(
                "rod quadratic={quadratic}, cells={cells}, k={k}, complex field error={error}"
            );
            assert!(
                error < previous[case] / reduction,
                "cells={cells}, k={k}, error={error}, prior={}",
                previous[case]
            );
            previous[case] = error;
            if cells == refinements[2] {
                assert!(error < finest_bound);
            }
            let physical = evaluate_harmonic_physical_state(
                &assembly.real,
                omega,
                &response.displacement_real,
                &response.displacement_imaginary,
                real_load,
                &assembly.imaginary_load,
            )
            .unwrap();
            let (_, closure) = physical
                .balance
                .closure_metrics(&physical.reaction)
                .unwrap();
            assert!(closure < 1.0e-8);
            // Independent support reaction R=-T/cos(k), not merely a residual check.
            let mut reaction_error = 0.0_f64;
            for (reaction, phase) in [
                (physical.reaction.real(), PHASOR[0]),
                (physical.reaction.imaginary(), PHASOR[1]),
            ] {
                let actual: f64 = reaction.chunks_exact(3).map(|v| v[0]).sum();
                let expected = -phase / k.cos();
                reaction_error = reaction_error.max((actual / expected - 1.0).abs());
            }
            eprintln!("rod quadratic={quadratic}, cells={cells}, k={k}, support reaction error={reaction_error}");
            assert!(reaction_error < previous_reaction[case] / reduction);
            previous_reaction[case] = reaction_error;
            if cells == refinements[2] {
                assert!(reaction_error < finest_bound);
            }
            let input_power = 0.5
                * omega
                * real_load
                    .iter()
                    .zip(&assembly.imaginary_load)
                    .zip(
                        response
                            .displacement_real
                            .iter()
                            .zip(&response.displacement_imaginary),
                    )
                    .map(|((fr, fi), (ur, ui))| fi * ur - fr * ui)
                    .sum::<f64>();
            assert!(input_power.abs() < 1.0e-9); // Undamped steady response has zero mean input/loss.
            assert!(response.relative_residual <= policy().relative_tolerance);
        }
    }
}

pub(crate) fn modal_model(model: &AnalysisModel) -> AnalysisModel {
    let mut modal = model.clone();
    modal.sequence.steps = vec![ProcedureStep::structural_modal("rod_basis")];
    replace_fixture_excitations(&mut modal, Vec::new());
    modal
}

// witness: FEA-DYNAMICS-MODAL-001/physical_mass_base_state @ modal_structural/tet4/public_study;
//   check: axial-rod-modal; boundary: Axial affine Tet4 slab rod frequency refinement and modal truncation against direct
//   harmonic response. Does not cover beam/shell modes, repeated eigenspaces, every participation
//   or remote envelope.
#[test]
fn solid_rod_modal_frequency_and_retained_response_have_independent_evidence() {
    modal_continuum(false);
}

// witness: FEA-DYNAMICS-MODAL-001/physical_mass_base_state @ modal_structural/tet10/public_study;
//   coverage: partial;
//   boundary: Axial affine Tet10 slab rod frequency refinement and modal truncation against direct
//   harmonic response; beam and shell modes, repeated eigenspaces and remote envelopes are not
//   covered.
// witness: FEA-DYNAMICS-MODAL-001/references @ modal_structural/tet10/public_study; coverage:
//   partial;
//   check: axial-rod-modal; boundary: Axial rod modal frequency against the continuum reference with retained-response
//   truncation checked against direct harmonic response; no beam modes, repeated modes or
//   mass-orthogonality series.
#[test]
fn solid_rod_tet10_modal_frequency_and_retained_response_have_independent_evidence() {
    modal_continuum(true);
}

pub(crate) fn rod_mesh(cells: usize, quadratic: bool) -> SolverMeshArtifact {
    if quadratic {
        quadratic_slab_mesh(cells)
    } else {
        slab_mesh(cells)
    }
}

fn modal_continuum(quadratic: bool) {
    let mut previous_frequency_error = f64::INFINITY;
    let refinements = if quadratic { [1, 2, 4] } else { [2, 4, 8] };
    for cells in refinements {
        let mesh = rod_mesh(cells, quadratic);
        let model = rod_model(&mesh, &[1.0]);
        let modal = super::run_modal_with_options(
            &modal_model(&model),
            ComputeBackend::Cpu,
            ModalSolveOptions {
                solver_mesh: Some(mesh.clone()),
                truncation: ModalTruncationPolicy::requested(1),
                ..ModalSolveOptions::default()
            },
        )
        .unwrap();
        let expected_hz = WAVE_SPEED / 4.0;
        let error = (modal.frequencies_hz()[0] / expected_hz - 1.0).abs();
        eprintln!("rod quadratic={quadratic}, cells={cells}, fundamental frequency error={error}");
        assert!(error < previous_frequency_error / if quadratic { 6.0 } else { 3.0 });
        previous_frequency_error = error;
        if cells == refinements[2] {
            assert!(error < if quadratic { 0.0002 } else { 0.002 });
        }
    }

    // On one fixed physical mesh, isolate truncation from spatial discretization.
    let mesh = rod_mesh(if quadratic { 1 } else { 2 }, quadratic);
    let model = rod_model(&mesh, &[1.0]);
    let assembly = assemble(&model, &mesh, 1.0);
    let direct = solve_direct_harmonic(
        &assembly.real,
        &assembly.imaginary_load,
        WAVE_SPEED,
        policy(),
    )
    .unwrap();
    let mut prior = f64::INFINITY;
    for requested in [1, 8, 18] {
        // Both coarse meshes have 27 nodes and 18 axial free DOFs.
        let modal = super::run_modal_with_options(
            &modal_model(&model),
            ComputeBackend::Cpu,
            ModalSolveOptions {
                solver_mesh: Some(mesh.clone()),
                truncation: ModalTruncationPolicy::requested(requested),
                ..ModalSolveOptions::default()
            },
        )
        .unwrap();
        let reduced = ModalReducedSystem::project(&assembly.real, &modal.basis).unwrap();
        let response = solve_modal_harmonic(
            &reduced,
            WAVE_SPEED,
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
        let real = (reduced)
            .expand_state(&response.displacement_real)
            .map(|field| field.into_canonical_storage())
            .unwrap();
        let imaginary = (reduced)
            .expand_state(&response.displacement_imaginary)
            .map(|field| field.into_canonical_storage())
            .unwrap();
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
        eprintln!(
            "rod quadratic={quadratic}, retained={}, direct/modal difference={difference}",
            reduced.mode_count()
        );
        assert!(difference < prior);
        prior = difference;
        // Each projected response is also checked against the independent continuum,
        // with an explicit bounded coarse-mesh envelope rather than direct=reference.
        assert!(field_error(&mesh, 1.0, &real, &imaginary) < 0.15);
        if requested == 18 {
            assert!(difference < 1.0e-7);
        }
    }
}

/// Cross-check of the two modal solve paths on one model that fits the direct path: the
/// shift-invert factorization and preconditioned LOBPCG return the same four axial-rod
/// frequencies to 1e-7 relative, and each path's factor-free Sturm count proves that no mode
/// below the reported ones was missed.
#[test]
fn solid_rod_modal_paths_agree_and_each_proves_completeness() {
    use crate::planning::resource::SubspaceSolvePath;
    let mesh = rod_mesh(4, true);
    let model = modal_model(&rod_model(&mesh, &[1.0]));
    let solve = |path| {
        super::execute_modal_core(
            &model,
            ComputeBackend::Cpu,
            ModalSolveOptions {
                solver_mesh: Some(mesh.clone()),
                truncation: ModalTruncationPolicy::requested(4),
                ..ModalSolveOptions::default()
            },
            runmat_analysis_core::AnalysisProcedure::ModalExtraction,
            None,
            path,
        )
        .unwrap()
    };
    let direct = solve(SubspaceSolvePath::ShiftInvertFactorization);
    let iterative = solve(SubspaceSolvePath::PreconditionedLobpcg);
    let frequencies = |core: &super::CompletedModalCore| -> Vec<f64> {
        core.basis
            .modes()
            .iter()
            .map(|mode| mode.frequency_hz())
            .collect()
    };
    let (a, b) = (frequencies(&direct), frequencies(&iterative));
    assert!(a.len() >= 4 && b.len() >= 4, "{a:?} {b:?}");
    for (x, y) in a.iter().zip(&b).take(4) {
        assert!((x - y).abs() <= 1e-7 * x, "direct {x} Hz vs LOBPCG {y} Hz");
    }
    // The fundamental is the quarter-wave axial mode.
    assert!((a[0] / (WAVE_SPEED / 4.0) - 1.0).abs() < 0.001, "{}", a[0]);
    for (core, method) in [
        (
            &direct,
            crate::contracts::SolverMethod::ShiftInvertBlockLanczos,
        ),
        (
            &iterative,
            crate::contracts::SolverMethod::PreconditionedLobpcg,
        ),
    ] {
        assert_eq!(core.run.solver_method, method);
        let sturm = core
            .run
            .diagnostics
            .iter()
            .find(|diagnostic| diagnostic.code == "FEA_MODAL_STURM_CHECK")
            .expect("each path runs the Sturm check");
        assert!(
            sturm.message.ends_with("complete=true"),
            "{}",
            sturm.message
        );
    }
}

/// The modes qualify on the normwise backward error and publish it beside the relative
/// residual; a policy that requires the relative residual at a level double precision cannot
/// reach refuses the same pairs.
#[test]
fn solid_rod_modes_qualify_on_the_backward_error_and_publish_both_measures() {
    use crate::planning::resource::SubspaceSolvePath;
    use runmat_analysis_core::AlgebraicAccuracyMeasure;
    let mesh = rod_mesh(4, true);
    let model = modal_model(&rod_model(&mesh, &[1.0]));
    let solve = |convergence| {
        super::execute_modal_core(
            &model,
            ComputeBackend::Cpu,
            ModalSolveOptions {
                solver_mesh: Some(mesh.clone()),
                truncation: ModalTruncationPolicy::requested(4),
                convergence,
                ..ModalSolveOptions::default()
            },
            runmat_analysis_core::AnalysisProcedure::ModalExtraction,
            None,
            SubspaceSolvePath::ShiftInvertFactorization,
        )
    };
    let production = crate::ModalConvergencePolicy::production();
    assert_eq!(
        production.algebraic_measure(),
        AlgebraicAccuracyMeasure::BackwardError
    );
    let core = solve(production).unwrap();
    let field = |id: &str| -> Vec<f64> {
        core.run
            .fields
            .iter()
            .find(|field| field.field_id == id)
            .unwrap_or_else(|| panic!("missing field {id}"))
            .as_host_f64()
            .expect("modal accuracy fields are host-backed")
            .to_vec()
    };
    let backward = field(crate::contracts::FEA_FIELD_MODAL_BACKWARD_ERROR);
    let relative = field(crate::contracts::FEA_FIELD_MODAL_RESIDUAL_NORM);
    assert_eq!(backward.len(), core.basis.modes().len());
    assert_eq!(relative.len(), backward.len());
    for (eta, residual) in backward.iter().zip(&relative) {
        assert!(*eta <= production.algebraic_tolerance(), "η = {eta}");
        assert!(
            *eta <= *residual,
            "η = {eta} above the relative residual {residual}"
        );
    }
    let accuracy = core
        .run
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code == "FEA_MODAL_ACCURACY")
        .expect("the per-mode accuracy is published");
    assert!(accuracy.message.contains("measure=backward_error"));
    assert!(accuracy.message.contains("relative_residual="));
    let strict =
        production.with_algebraic_accuracy(AlgebraicAccuracyMeasure::RelativeResidual, 1.0e-16);
    let Err(refused) = solve(strict) else {
        panic!("a relative residual of 1e-16 is below double precision and must be refused");
    };
    let refused = refused.to_string();
    assert!(
        refused.contains("did not qualify"),
        "a relative residual of 1e-16 is below double precision: {refused}"
    );
}
