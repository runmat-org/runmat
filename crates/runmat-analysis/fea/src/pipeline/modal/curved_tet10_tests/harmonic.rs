//! Physical solid harmonic qualification against an independent scalar oscillator.
//!
//! Reference: MIT 8.03SC, The Physics of Waves, chapter 2, equations 2.14–2.24:
//! https://ocw.mit.edu/courses/8-03sc-physics-iii-vibrations-and-waves-fall-2016/782069da3820fc514c10c26ae0c15b01_MIT8_03SCF16_Text_Ch2.pdf
//! The reference uses exp(-i omega t); conjugating it gives the exp(+i omega t)
//! convention here: u = F / (k - omega^2 m + i omega c).
//! This isolates numerical operators with an explicit nodal phasor. It does not
//! qualify authored load realization or an unconstrained continuum resonance.

use runmat_analysis_core::{
    MaterialElasticityModel, MechanicalDampingModel, RayleighMassCoefficient,
    RayleighStiffnessCoefficient,
};

use super::{
    canonical_tetrahedron_solver_mesh, constrain_everything_except_corner_x,
    curved_corner_x_stiffness_reference, fixture_model, replace_fixture_excitations,
    ComputeBackend, ElementOrder, FixtureId, ModalSolveOptions, ModalTruncationPolicy,
    ProcedureStep, SolverMeshArtifact, StableDigest,
};
use crate::{
    assembly::{assemble_structural_system, StructuralFormulationRequirement},
    excitation::{realize_step_excitations, ExcitationSamplePoint},
    solve::dynamics::{
        evaluate_harmonic_physical_state, solve_direct_harmonic, solve_modal_harmonic,
        DirectHarmonicPolicy, ModalHarmonicLoad, ModalReducedSystem,
    },
};

#[derive(Clone, Copy, Debug)]
enum SolidCase {
    Tet4,
    Tet10 { warp: f64 },
}

impl SolidCase {
    fn mesh(self) -> SolverMeshArtifact {
        let (order, warp) = match self {
            Self::Tet4 => (ElementOrder::Tet4, 0.0),
            Self::Tet10 { warp } => (ElementOrder::Tet10, warp),
        };
        let mut mesh = canonical_tetrahedron_solver_mesh(order);
        if warp != 0.0 {
            {
                let quantity_change = warp;
                let quantity_index = 0;
                let quantity_slot = &mut mesh.topology.nodes[4].coordinates_m;
                let mut quantity_coordinates = quantity_slot.canonical_value();
                quantity_coordinates[quantity_index] += quantity_change;
                *quantity_slot = runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new(quantity_coordinates);
            };
            mesh.canonical_digest = StableDigest::ZERO;
            mesh.seal_canonical_digest().unwrap();
        }
        mesh
    }

    fn stiffness_and_mass(self) -> (f64, f64) {
        // E=100 Pa, nu=1/4 gives lambda=mu=40 Pa; rho=1 kg/m^3.
        // Tet4: grad(N_1)=(-1,-1,-1), V=1/6, integral(N_1^2)=1/60.
        // Tet10: N_1=2 L_1^2-L_1; straight k=20, m=1/420.
        // With the existing edge warp, det(J)=1+4 warp (L_1-L_2),
        // giving the exact mass below; stiffness uses independent 24-point integration.
        match self {
            Self::Tet4 => (100.0 / 3.0, 1.0 / 60.0),
            Self::Tet10 { warp } => (
                if warp == 0.0 {
                    20.0
                } else {
                    curved_corner_x_stiffness_reference(warp, 100.0, 0.25)
                },
                1.0 / 420.0 + warp / 315.0,
            ),
        }
    }
}

// witness: FEA-DYNAMICS-RESPONSE-001/references @
//   linear_structural_dynamics/tet4|tet10/numerical_kernel; coverage: partial;
//   boundary: Direct and modal harmonic match independent phase and power references on a curved
//   solid; oscillators, impulse and base-motion references are not covered.
// witness: FEA-DYNAMICS-RESPONSE-001/direct_modal_procedures @
//   linear_structural_dynamics/tet4|tet10/numerical_kernel; coverage: partial;
//   boundary: Direct and modal harmonic procedures agree on a curved solid.
#[test]
fn physical_solid_direct_and_modal_harmonic_match_independent_phase_and_power_references() {
    for case in [
        SolidCase::Tet4,
        SolidCase::Tet10 { warp: 0.0 },
        SolidCase::Tet10 { warp: 0.1 },
        SolidCase::Tet10 { warp: 0.2 },
    ] {
        for damped in [false, true] {
            qualify(case, damped);
        }
    }
}

fn qualify(case: SolidCase, damped: bool) {
    let mesh = case.mesh();
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.sequence.steps = vec![ProcedureStep::structural_modal("harmonic_reference_basis")];
    model.constraints = constrain_everything_except_corner_x(&mesh);
    replace_fixture_excitations(&mut model, Vec::new());
    let MaterialElasticityModel::IsotropicLinear(elasticity) =
        model.materials[0].elasticity.as_mut().unwrap()
    else {
        panic!("expected isotropic linear elasticity");
    };
    elasticity.youngs_modulus_pa = runmat_types::physical::YoungsModulus::new(100.0);
    elasticity.poisson_ratio = runmat_types::physical::PoissonRatio::new(0.25);
    model.materials[0].mass_density_kg_per_m3 =
        (Some(1.0)).map(runmat_types::physical::MassDensity::new);
    let (alpha, beta) = if damped { (0.8, 0.002) } else { (0.0, 0.0) };
    model.materials[0].damping = damped.then_some(MechanicalDampingModel::Rayleigh {
        mass_coefficient: RayleighMassCoefficient::new(alpha),
        stiffness_coefficient: RayleighStiffnessCoefficient::new(beta),
    });

    let modal = super::super::run_modal_with_options(
        &model,
        ComputeBackend::Cpu,
        ModalSolveOptions {
            truncation: ModalTruncationPolicy::requested(1),
            solver_mesh: Some(mesh.clone()),
            ..ModalSolveOptions::default()
        },
    )
    .expect("physical solid one-DOF modal basis must solve");
    let excitations = realize_step_excitations(
        &model,
        &model.sequence.steps[0],
        ExcitationSamplePoint::Static,
    )
    .unwrap();
    let mut summary = assemble_structural_system(
        &model,
        &excitations,
        Some(mesh),
        StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .expect("physical solid must assemble");
    let reduced = ModalReducedSystem::project(&summary, &modal.basis).unwrap();
    assert_eq!(reduced.mode_count(), 1);
    let (stiffness, mass) = case.stiffness_and_mass();
    let damping = alpha * mass + beta * stiffness;
    let natural_omega = (stiffness / mass).sqrt();
    let [modal_frequency_hz] = modal.frequencies_hz().try_into().unwrap();
    assert!((modal_frequency_hz * std::f64::consts::TAU / natural_omega - 1.0).abs() < 1.0e-7);

    for ratio in [0.25, 0.75, 1.0, 1.5, 3.0] {
        // The undamped scalar equation has no bounded steady response at exact resonance.
        if !damped && ratio == 1.0 {
            continue;
        }
        for scale in [1.0e-12, 1.0, 1.0e3] {
            let omega = ratio * natural_omega;
            let force = [scale, 0.4 * scale];
            let a = stiffness - omega * omega * mass;
            let b = omega * damping;
            let denominator = a * a + b * b;
            let reference = [
                (force[0] * a + force[1] * b) / denominator,
                (force[1] * a - force[0] * b) / denominator,
            ];
            let mut real_load = vec![0.0; summary.dof_count];
            let mut imaginary_load = vec![0.0; summary.dof_count];
            real_load[0] = force[0];
            imaginary_load[0] = force[1];
            summary.operator = summary.operator.with_load(real_load).unwrap();
            let direct = solve_direct_harmonic(
                &summary,
                &imaginary_load,
                omega,
                DirectHarmonicPolicy {
                    relative_tolerance: 1.0e-10,
                    ..DirectHarmonicPolicy::default()
                },
            )
            .expect("physical direct harmonic response must converge");
            let modal = solve_modal_harmonic(
                &reduced,
                omega,
                &ModalHarmonicLoad {
                    real: (reduced)
                        .load_field(summary.operator.load())
                        .and_then(|field| (reduced).project_load(&field))
                        .unwrap(),
                    imaginary: (reduced)
                        .load_field(&imaginary_load)
                        .and_then(|field| (reduced).project_load(&field))
                        .unwrap(),
                },
            )
            .unwrap();
            let modal_real = (reduced)
                .expand_state(&modal.displacement_real)
                .map(|field| field.into_canonical_storage())
                .unwrap();
            let modal_imaginary = (reduced)
                .expand_state(&modal.displacement_imaginary)
                .map(|field| field.into_canonical_storage())
                .unwrap();
            for (real, imaginary) in [
                (&direct.displacement_real, &direct.displacement_imaginary),
                (&modal_real, &modal_imaginary),
            ] {
                let actual = [real[0], imaginary[0]];
                let relative_error = (actual[0] - reference[0]).hypot(actual[1] - reference[1])
                    / reference[0].hypot(reference[1]);
                assert!(
                    relative_error < 2.0e-7,
                    "{case:?}, damped={damped}, ratio={ratio}, scale={scale}: {actual:?} vs {reference:?}, error={relative_error}"
                );
                assert!(real
                    .iter()
                    .skip(1)
                    .chain(imaginary.iter().skip(1))
                    .all(|v| *v == 0.0));
                let input_power = 0.5 * omega * (force[1] * actual[0] - force[0] * actual[1]);
                let dissipated_power =
                    0.5 * damping * omega.powi(2) * (reference[0].powi(2) + reference[1].powi(2));
                let power_scale =
                    0.5 * omega * force[0].hypot(force[1]) * reference[0].hypot(reference[1]);
                assert!((input_power - dissipated_power).abs() <= 2.0e-7 * power_scale);
                if damped {
                    assert!(input_power > 0.0);
                }
                let physical = evaluate_harmonic_physical_state(
                    &summary,
                    omega,
                    real,
                    imaginary,
                    summary.operator.load(),
                    &imaginary_load,
                )
                .unwrap();
                let (_, balance_error) = physical
                    .balance
                    .closure_metrics(&physical.reaction)
                    .unwrap();
                assert!(balance_error <= 1.0e-9);
            }
            assert!(direct.relative_residual <= 1.0e-10);
            assert!(modal.relative_residual <= 1.0e-10);
        }
    }
}
