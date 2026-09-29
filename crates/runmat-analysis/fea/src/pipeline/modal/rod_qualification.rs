//! Continuum and typed-execution qualification, independent of discrete K/M entries.
//! Reference wave equation and fixed/free spectrum: UT Austin, equations 5.15–5.22:
//! https://farside.ph.utexas.edu/teaching/315/Waves/node34.html
//! Reflecting its coordinate gives u(0)=0. Applying E u'(1)=T gives
//! u(x)=T sin(k x)/(E k cos(k)), k=omega sqrt(rho/E).

use runmat_analysis_core::{
    AnalysisConstraint, AnalysisModel, CoordinateFrame, Displacement, FormulationRequirement,
    IndependentDomain, MaterialElasticityModel, PhysicalSignal, PrescribedStructuralDofs,
    PrescribedValue, ProcedureStep, PropertyRegionTarget, RegionTarget, SignalExtrapolation,
    SignalInterpolation, SignalKnot, SignalOrdinate, SignalPrecision, SignalRepresentation,
    SignalValue, StructuralConstraint, StructuralConstraintKind, StructuralTranslationDof,
    TabulatedSignal, TabulatedSignalDomain, Traction, TranslationDofValue,
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
    topology::test_meshes::slab_mesh,
    ComputeBackend, ModalSolveOptions,
};

const YOUNGS: f64 = 100.0;
const WAVE_SPEED: f64 = 10.0;
const PHASOR: [f64; 2] = [1.0, 0.4];

mod execution;

fn rod_model(mesh: &SolverMeshArtifact, wave_numbers: &[f64]) -> AnalysisModel {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.material_assignments[0].target = PropertyRegionTarget::solid("slab");
    let MaterialElasticityModel::IsotropicLinear(elasticity) =
        model.materials[0].elasticity.as_mut().unwrap()
    else {
        panic!("expected isotropic material");
    };
    elasticity.youngs_modulus_pa = YOUNGS;
    // With nu=0, u=(u(x),0,0) is an exact three-dimensional solid solution.
    // Holding lateral translations excludes bending/shear modes, not axial strain.
    elasticity.poisson_ratio = 0.0;
    model.materials[0].mass_density_kg_per_m3 = Some(1.0);
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
            .filter(|dof| node.coordinates_m[0] == 0.0 || *dof != StructuralTranslationDof::Ux)
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

fn assemble(
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
    let mut error = 0.0;
    let mut norm = 0.0;
    for (index, node) in mesh.topology.nodes.iter().enumerate() {
        let expected = reference(node.coordinates_m[0], k);
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

#[test]
fn solid_rod_harmonic_face_traction_converges_to_independent_continuum() {
    let wave_numbers = [0.5, 1.0, 2.0]; // Both sides of first resonance pi/2, never exactly at it.
    let mut previous = [f64::INFINITY; 3];
    let mut previous_reaction = [f64::INFINITY; 3];
    for cells in [2, 4, 8] {
        let mesh = slab_mesh(cells);
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
                        m[0] += node.coordinates_m[2] * v[0];
                        m[1] -= node.coordinates_m[1] * v[0];
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
            eprintln!("rod cells={cells}, k={k}, complex field error={error}");
            assert!(
                error < previous[case] / 2.5,
                "cells={cells}, k={k}, error={error}, prior={}",
                previous[case]
            );
            previous[case] = error;
            if cells == 8 {
                assert!(error < 0.02);
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
            eprintln!("rod cells={cells}, k={k}, support reaction error={reaction_error}");
            assert!(reaction_error < previous_reaction[case] / 2.5);
            previous_reaction[case] = reaction_error;
            if cells == 8 {
                assert!(reaction_error < 0.02);
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

fn modal_model(model: &AnalysisModel) -> AnalysisModel {
    let mut modal = model.clone();
    modal.sequence.steps = vec![ProcedureStep::structural_modal("rod_basis")];
    replace_fixture_excitations(&mut modal, Vec::new());
    modal
}

#[test]
fn solid_rod_modal_frequency_and_retained_response_have_independent_evidence() {
    let mut previous_frequency_error = f64::INFINITY;
    for cells in [2, 4, 8] {
        let mesh = slab_mesh(cells);
        let model = rod_model(&mesh, &[1.0]);
        let modal = super::run_modal_with_options(
            &modal_model(&model),
            ComputeBackend::Cpu,
            ModalSolveOptions {
                solver_mesh: Some(mesh.clone()),
                truncation: ModalTruncationPolicy::requested(1),
                formulation_requirement: FormulationRequirement::Experimental,
                ..ModalSolveOptions::default()
            },
        )
        .unwrap();
        let expected_hz = WAVE_SPEED / 4.0;
        let error = (modal.frequencies_hz()[0] / expected_hz - 1.0).abs();
        eprintln!("rod cells={cells}, fundamental frequency error={error}");
        assert!(error < previous_frequency_error / 3.0);
        previous_frequency_error = error;
        if cells == 8 {
            assert!(error < 0.002);
        }
    }

    // On one fixed physical mesh, isolate truncation from spatial discretization.
    let mesh = slab_mesh(2);
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
        // 18 axial free DOFs on this 27-node mesh.
        let modal = super::run_modal_with_options(
            &modal_model(&model),
            ComputeBackend::Cpu,
            ModalSolveOptions {
                solver_mesh: Some(mesh.clone()),
                truncation: ModalTruncationPolicy::requested(requested),
                formulation_requirement: FormulationRequirement::Experimental,
                ..ModalSolveOptions::default()
            },
        )
        .unwrap();
        let reduced = ModalReducedSystem::project(&assembly.real, &modal.basis).unwrap();
        let response = solve_modal_harmonic(
            &reduced,
            WAVE_SPEED,
            &ModalHarmonicLoad {
                real: reduced
                    .project_vector(assembly.real.operator.load())
                    .unwrap(),
                imaginary: reduced.project_vector(&assembly.imaginary_load).unwrap(),
            },
        )
        .unwrap();
        let real = reduced.expand_vector(&response.displacement_real).unwrap();
        let imaginary = reduced
            .expand_vector(&response.displacement_imaginary)
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
            "rod retained={}, direct/modal difference={difference}",
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
