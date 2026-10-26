use runmat_analysis_core::{
    ComplexSpectralDensity, CoordinateFrame, CrossPowerSpectralDensity, IndependentDomain,
    ProcedureStep, RandomExcitationChannel, RandomExcitationChannelId, RandomExcitationId,
    RandomExcitationQuantity, RandomExcitationSpectrum, RandomSpectrumEvidence,
    RandomSpectrumExtrapolation, RandomSpectrumInterpolation, SpectralDensityConvention,
    SpectrumDirection,
};

use crate::{
    assembly::{assemble_structural_system, AssemblySummary, StructuralFormulationRequirement},
    contracts::{ComputeBackend, ModalSolveOptions},
    excitation::RealizedExcitations,
    fixtures::{fixture_model, FixtureId},
    pipeline::{
        modal::run_modal_with_options,
        modal_basis::ModalBasisArtifact,
        modal_governance::{ModalConvergencePolicy, ModalTruncationPolicy},
    },
};

use super::*;

fn physical_basis() -> (AssemblySummary, ModalBasisArtifact) {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.excitations.clear();
    model.excitation_cases.clear();
    let mut modal_step = ProcedureStep::structural_modal("modes");
    modal_step.independent_domain = IndependentDomain::Eigenvalue { requested_count: 2 };
    model.sequence.steps = vec![modal_step];
    model.sequence.dependencies.clear();
    model.sequence.transitions.clear();
    let mesh = crate::tests::single_tetrahedron_solver_mesh_for_model(&model);
    let modal = run_modal_with_options(
        &model,
        ComputeBackend::Cpu,
        ModalSolveOptions {
            truncation: ModalTruncationPolicy::requested(2),
            convergence: ModalConvergencePolicy::exploratory(),
            residual_vector_directions: Vec::new(),
            solver_mesh: Some(mesh.clone()),
        },
    )
    .unwrap();
    let summary = assemble_structural_system(
        &model,
        &RealizedExcitations::default(),
        Some(mesh),
        StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .unwrap();
    modal.basis.validate_for(&summary).unwrap();
    (summary, modal.basis)
}

fn channel(id: &str, direction: [f64; 3]) -> RandomExcitationChannel {
    RandomExcitationChannel {
        channel_id: RandomExcitationChannelId(id.into()),
        quantity: RandomExcitationQuantity::BaseAcceleration,
        direction: SpectrumDirection {
            frame: CoordinateFrame::Global,
            vector: direction,
        },
    }
}

fn spectrum(
    frequencies_hz: Vec<f64>,
    channels: Vec<RandomExcitationChannel>,
    matrices: Vec<CrossPowerSpectralDensity>,
) -> RandomExcitationSpectrum {
    RandomExcitationSpectrum {
        excitation_id: RandomExcitationId("broadband-base".into()),
        frequencies_hz: runmat_types::physical::Quantity::new(frequencies_hz),
        channels,
        spectral_density: runmat_analysis_core::NumericalDataSource::authored(
            matrices,
            "sha256:test-broadband-base",
        ),
        convention: SpectralDensityConvention::OneSidedPerHertz,
        interpolation: RandomSpectrumInterpolation::LinearFrequency,
        extrapolation: RandomSpectrumExtrapolation::Reject,
        evidence: RandomSpectrumEvidence {
            max_absolute_density_error: (Some(0.0))
                .map(runmat_types::physical::AccelerationPowerSpectralDensity::new),
            equivalent_degrees_of_freedom: None,
        },
    }
}

fn policy() -> RandomResponseIntegrationPolicy {
    RandomResponseIntegrationPolicy {
        modal_damping_ratio: 0.04,
        minimum_cumulative_mass_ratio: 0.0,
        relative_variance_tolerance: 2.0e-6,
        maximum_refinement_levels: 18,
    }
}

#[test]
fn physical_random_response_matches_independent_dense_reference_and_round_trips() {
    let (summary, basis) = physical_basis();
    let mode = &basis.modes()[0];
    let lower = mode.frequency_hz() * 0.2;
    let upper = mode.frequency_hz() * 5.0;
    let density = 2.5e-4;
    let excitation = spectrum(
        vec![lower, upper],
        vec![channel("x", [1.0, 0.0, 0.0])],
        vec![
            CrossPowerSpectralDensity::diagonal(&runmat_types::physical::Quantity::new([density])),
            CrossPowerSpectralDensity::diagonal(&runmat_types::physical::Quantity::new([density])),
        ],
    );
    let result = solve_modal_random_response(&summary, &basis, &excitation, policy()).unwrap();
    result.validate_for(&summary, &basis, &excitation).unwrap();
    let breakpoints =
        random_response_interval_breakpoints(&summary, &basis, &excitation, policy()).unwrap();
    let mut intervals = (0..breakpoints.len() - 1)
        .map(|ordinal| {
            solve_modal_random_response_interval(&summary, &basis, &excitation, policy(), ordinal)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let first = intervals[0].clone();
    assert_eq!(
        ModalRandomResponseIntervalResult::from_canonical_bytes(&first.canonical_bytes().unwrap())
            .unwrap(),
        first
    );
    intervals.reverse();
    assert_eq!(
        assemble_modal_random_response_intervals(
            &summary,
            &basis,
            &excitation,
            policy(),
            intervals.clone(),
        )
        .unwrap(),
        result
    );
    intervals.pop();
    assert!(assemble_modal_random_response_intervals(
        &summary,
        &basis,
        &excitation,
        policy(),
        intervals,
    )
    .unwrap_err()
    .to_string()
    .contains("incomplete"));

    // Basis-independent reference: the x-channel response of every retained mode, including
    // the cross terms of repeated frequencies, so any orthonormal basis of a degenerate
    // eigenspace gives the same variance.
    let modes = basis.modes();
    let coefficient = |dof: usize| -> Vec<f64> {
        modes
            .iter()
            .map(|mode| mode.dof_shape()[dof] * mode.directional()[0].participation_factor())
            .collect()
    };
    let dof = (0..mode.dof_shape().len())
        .max_by(|left, right| {
            let total = |dof: usize| coefficient(dof).iter().sum::<f64>().abs();
            total(*left).total_cmp(&total(*right))
        })
        .unwrap();
    let weights = coefficient(dof);
    let panel_count = 400_000_usize;
    let step = (upper - lower) / panel_count as f64;
    let ordinate = |frequency_hz: f64| {
        let angular = std::f64::consts::TAU * frequency_hz;
        let (mut real, mut imaginary) = (0.0, 0.0);
        for (mode, weight) in modes.iter().zip(&weights) {
            let natural = std::f64::consts::TAU * mode.frequency_hz();
            let a = natural * natural - angular * angular;
            let b = 2.0 * policy().modal_damping_ratio * natural * angular;
            let denominator = a * a + b * b;
            real += weight * a / denominator;
            imaginary -= weight * b / denominator;
        }
        (real * real + imaginary * imaginary) * density
    };
    let mut dense = 0.5 * (ordinate(lower) + ordinate(upper));
    for index in 1..panel_count {
        dense += ordinate(lower + index as f64 * step);
    }
    dense *= step;
    let relative_error = (result.dof_variance()[dof] - dense).abs() / dense;
    assert!(relative_error < 2.0e-5, "relative error={relative_error:e}");
    assert!(result
        .field(StructuralLinearFieldQuantity::Stress)
        .is_some());
    assert_eq!(
        ModalRandomResponseResult::from_canonical_bytes(&result.canonical_bytes().unwrap())
            .unwrap(),
        result
    );
    let mut substituted = excitation.clone();
    let matrix = &mut substituted.spectral_density.values_mut().unwrap()[0];
    matrix.values = matrix
        .values
        .clone()
        .try_map_canonical_storage(|mut values| {
            values[0].real *= 2.0;
            values
        })
        .unwrap();
    assert!(result
        .validate_for(&summary, &basis, &substituted)
        .unwrap_err()
        .to_string()
        .contains("does not name"));
}

// witness: FEA-DYNAMICS-RESPONSE-001/spectrum_random_recovery @
//   linear_structural_dynamics/mesh_independent/numerical_kernel; coverage: partial;
//   boundary: A full cross-spectrum can cancel identical base-motion channels; PSD integration and
//   residual-mode terms are other tests.
#[test]
fn full_cross_spectrum_can_cancel_identical_base_motion_channels() {
    let (summary, basis) = physical_basis();
    let frequency = basis.modes()[0].frequency_hz();
    let matrix = CrossPowerSpectralDensity {
        values: runmat_types::physical::Quantity::new(vec![
            ComplexSpectralDensity::new(1.0, 0.0),
            ComplexSpectralDensity::new(-1.0, 0.0),
            ComplexSpectralDensity::new(-1.0, 0.0),
            ComplexSpectralDensity::new(1.0, 0.0),
        ]),
    };
    let excitation = spectrum(
        vec![frequency * 0.5, frequency * 2.0],
        vec![
            channel("x-a", [1.0, 0.0, 0.0]),
            channel("x-b", [1.0, 0.0, 0.0]),
        ],
        vec![matrix.clone(), matrix],
    );
    let result = solve_modal_random_response(&summary, &basis, &excitation, policy()).unwrap();
    assert!(result
        .dof_variance()
        .iter()
        .all(|variance| *variance <= 1.0e-24));
    assert!(result
        .fields()
        .iter()
        .flat_map(RandomResponseFieldStatistics::variance)
        .all(|variance| *variance <= 1.0e-18));
}

#[test]
fn insufficient_directional_mass_is_rejected() {
    let (summary, basis) = physical_basis();
    let frequency = basis.modes()[0].frequency_hz();
    let first = basis.modes()[0]
        .directional()
        .map(|metric| metric.participation_factor());
    let second = basis.modes()[1]
        .directional()
        .map(|metric| metric.participation_factor());
    let direction = [
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    ];
    assert!(direction.iter().map(|value| value * value).sum::<f64>() > f64::EPSILON);
    let excitation = spectrum(
        vec![frequency * 0.5, frequency * 2.0],
        vec![channel("missing-direction", direction)],
        vec![
            CrossPowerSpectralDensity::diagonal(&runmat_types::physical::Quantity::new([1.0])),
            CrossPowerSpectralDensity::diagonal(&runmat_types::physical::Quantity::new([1.0])),
        ],
    );
    let mut demanding = policy();
    demanding.minimum_cumulative_mass_ratio = 1.0;
    let error = solve_modal_random_response(&summary, &basis, &excitation, demanding).unwrap_err();
    assert!(error.to_string().contains("does not capture"));
}
