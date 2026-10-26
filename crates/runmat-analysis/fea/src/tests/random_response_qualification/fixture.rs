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
    solve::dynamics::RandomResponseIntegrationPolicy,
};

pub(super) fn complete_physical_basis() -> (AssemblySummary, ModalBasisArtifact) {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.excitations.clear();
    model.excitation_cases.clear();
    let mut modal_step = ProcedureStep::structural_modal("qualification-modes");
    modal_step.independent_domain = IndependentDomain::Eigenvalue { requested_count: 3 };
    model.sequence.steps = vec![modal_step];
    model.sequence.dependencies.clear();
    model.sequence.transitions.clear();

    let mesh = crate::tests::single_tetrahedron_solver_mesh_for_model(&model);
    let modal = run_modal_with_options(
        &model,
        ComputeBackend::Cpu,
        ModalSolveOptions {
            truncation: ModalTruncationPolicy::requested(3),
            convergence: ModalConvergencePolicy::exploratory(),
            residual_vector_directions: Vec::new(),
            solver_mesh: Some(mesh.clone()),
        },
    )
    .expect("the physical qualification fixture must produce a complete modal basis");
    let summary = assemble_structural_system(
        &model,
        &RealizedExcitations::default(),
        Some(mesh),
        StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .expect("the physical qualification fixture must assemble");
    modal
        .basis
        .validate_for(&summary)
        .expect("the qualification basis must bind to its physical operator");
    assert_eq!(modal.basis.modes().len(), 3);
    (summary, modal.basis)
}

pub(super) fn channel(id: &str, direction: [f64; 3]) -> RandomExcitationChannel {
    RandomExcitationChannel {
        channel_id: RandomExcitationChannelId(id.into()),
        quantity: RandomExcitationQuantity::BaseAcceleration,
        direction: SpectrumDirection {
            frame: CoordinateFrame::Global,
            vector: direction,
        },
    }
}

pub(super) fn constant_spectrum(
    bounds_hz: [f64; 2],
    channels: Vec<RandomExcitationChannel>,
    matrix: CrossPowerSpectralDensity,
) -> RandomExcitationSpectrum {
    RandomExcitationSpectrum {
        excitation_id: RandomExcitationId("qualification-spectrum".into()),
        frequencies_hz: runmat_types::physical::Quantity::new(bounds_hz.to_vec()),
        channels,
        spectral_density: runmat_analysis_core::NumericalDataSource::authored(
            vec![matrix.clone(), matrix],
            "sha256:random-response-qualification",
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

pub(super) fn hermitian_density(
    diagonal: [f64; 2],
    correlation: f64,
    phase_radians: f64,
) -> CrossPowerSpectralDensity {
    let magnitude = correlation * (diagonal[0] * diagonal[1]).sqrt();
    let cross = ComplexSpectralDensity::new(
        magnitude * phase_radians.cos(),
        magnitude * phase_radians.sin(),
    );
    CrossPowerSpectralDensity {
        values: runmat_types::physical::Quantity::new(vec![
            ComplexSpectralDensity::new(diagonal[0], 0.0),
            cross,
            ComplexSpectralDensity::new(cross.real, -cross.imaginary),
            ComplexSpectralDensity::new(diagonal[1], 0.0),
        ]),
    }
}

pub(super) fn converged_policy(mass_ratio: f64) -> RandomResponseIntegrationPolicy {
    RandomResponseIntegrationPolicy {
        modal_damping_ratio: 0.08,
        minimum_cumulative_mass_ratio: mass_ratio,
        relative_variance_tolerance: 2.0e-7,
        maximum_refinement_levels: 20,
    }
}

pub(super) fn full_frequency_bounds(basis: &ModalBasisArtifact) -> [f64; 2] {
    [
        basis.modes()[0].frequency_hz() * 0.2,
        basis.modes().last().unwrap().frequency_hz() * 3.0,
    ]
}
