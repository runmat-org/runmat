use runmat_analysis_core::{AnalysisModel, ProcedureStep};
use runmat_meshing_core::SolverMeshArtifact;

use crate::excitation::{
    realize_step_excitations, ComplexComponent, ExcitationSamplePoint, RealizedExcitations,
};

use super::{
    assemble_structural_perturbation_system, assemble_structural_system, AssemblySummary,
    LinearAssemblyError, StructuralFormulationRequirement,
};

/// One structural operator with losslessly projected real and imaginary load vectors.
pub struct ComplexStructuralAssemblySummary {
    pub real: AssemblySummary,
    pub imaginary_load: Vec<f64>,
    pub imaginary_projection_evidence: Vec<crate::excitation::SpatialProjectionEvidence>,
}

pub fn assemble_complex_structural_system(
    model: &AnalysisModel,
    step: &ProcedureStep,
    frequency_hz: f64,
    solver_mesh: Option<SolverMeshArtifact>,
    formulation: StructuralFormulationRequirement,
) -> Result<ComplexStructuralAssemblySummary, LinearAssemblyError> {
    let excitations =
        realize_step_excitations(model, step, ExcitationSamplePoint::frequency(frequency_hz))
            .map_err(|error| LinearAssemblyError::CouplingRealization(error.to_string()))?;
    assemble_realized_complex_structural_system(model, &excitations, solver_mesh, formulation)
}

fn assemble_realized_complex_structural_system(
    model: &AnalysisModel,
    excitations: &RealizedExcitations,
    solver_mesh: Option<SolverMeshArtifact>,
    formulation: StructuralFormulationRequirement,
) -> Result<ComplexStructuralAssemblySummary, LinearAssemblyError> {
    assemble_realized_with(
        model,
        excitations,
        solver_mesh,
        formulation,
        assemble_structural_system,
    )
}

/// Complex perturbation about a retained preload: assemble the elastic reference and
/// frequency-domain loads without applying the authored global thermal profile again.
pub(crate) fn assemble_complex_structural_perturbation_system(
    model: &AnalysisModel,
    step: &ProcedureStep,
    frequency_hz: f64,
    solver_mesh: Option<SolverMeshArtifact>,
    formulation: StructuralFormulationRequirement,
) -> Result<ComplexStructuralAssemblySummary, LinearAssemblyError> {
    let excitations =
        realize_step_excitations(model, step, ExcitationSamplePoint::frequency(frequency_hz))
            .map_err(|error| LinearAssemblyError::CouplingRealization(error.to_string()))?;
    assemble_realized_with(
        model,
        &excitations,
        solver_mesh,
        formulation,
        assemble_structural_perturbation_system,
    )
}

type RealStructuralAssembler = fn(
    &AnalysisModel,
    &RealizedExcitations,
    Option<SolverMeshArtifact>,
    StructuralFormulationRequirement,
) -> Result<AssemblySummary, LinearAssemblyError>;

fn assemble_realized_with(
    model: &AnalysisModel,
    excitations: &RealizedExcitations,
    solver_mesh: Option<SolverMeshArtifact>,
    formulation: StructuralFormulationRequirement,
    assemble: RealStructuralAssembler,
) -> Result<ComplexStructuralAssemblySummary, LinearAssemblyError> {
    let real = assemble(
        model,
        &excitations.component(ComplexComponent::Real),
        solver_mesh.clone(),
        formulation,
    )?;
    let imaginary = assemble(
        model,
        &excitations.component(ComplexComponent::Imaginary),
        solver_mesh,
        formulation,
    )?;
    if real.dof_count != imaginary.dof_count
        || real.operator.operators() != imaginary.operator.operators()
        || real.operator.constrained_dofs() != imaginary.operator.constrained_dofs()
        || real.constraint_matrix != imaginary.constraint_matrix
        || real.structural_dof_layout != imaginary.structural_dof_layout
        || real.structural_formulations != imaginary.structural_formulations
        || real.structural_solid_recovery != imaginary.structural_solid_recovery
        || real.structural_beam_recovery != imaginary.structural_beam_recovery
        || real.structural_shell_recovery != imaginary.structural_shell_recovery
    {
        return Err(LinearAssemblyError::SparseOperator(
            "real and imaginary structural load projection changed the canonical operator or recovery basis"
                .into(),
        ));
    }
    Ok(ComplexStructuralAssemblySummary {
        imaginary_load: imaginary.operator.load().to_vec(),
        imaginary_projection_evidence: imaginary.spatial_projection_evidence,
        real,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{fixture_model, FixtureId};

    #[test]
    fn complex_force_phase_is_projected_without_changing_the_operator() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.sequence.steps = vec![runmat_analysis_core::ProcedureStep::structural_harmonic(
            "harmonic",
        )];
        let runmat_analysis_core::PhysicalSignal::Force(signal) = &mut model.signals[0] else {
            panic!("cantilever fixture must carry a force signal");
        };
        signal.representation = runmat_analysis_core::SignalRepresentation::Tabulated(
            runmat_analysis_core::TabulatedSignal {
                domain: runmat_analysis_core::TabulatedSignalDomain::FrequencyHertz,
                knots: vec![runmat_analysis_core::SignalKnot::continuous(
                    60.0,
                    runmat_analysis_core::SignalOrdinate::Complex {
                        real: runmat_analysis_core::SignalValue::Vector3(
                            [2.0, 4.0, 6.0].map(runmat_analysis_core::Force::new),
                        ),
                        imaginary: runmat_analysis_core::SignalValue::Vector3(
                            [3.0, 6.0, 9.0].map(runmat_analysis_core::Force::new),
                        ),
                    },
                )],
                interpolation: runmat_analysis_core::SignalInterpolation::Linear,
                extrapolation: runmat_analysis_core::SignalExtrapolation::HoldEndpoint,
            },
        );
        signal.precision = runmat_analysis_core::SignalPrecision::Complex64;
        signal.evidence.sample_count = 1;
        let excitations = crate::excitation::realize_step_excitations(
            &model,
            &model.sequence.steps[0],
            crate::excitation::ExcitationSamplePoint::frequency(60.0),
        )
        .unwrap();
        let assembled = assemble_realized_complex_structural_system(
            &model,
            &excitations,
            None,
            StructuralFormulationRequirement::SyntheticOperator,
        )
        .unwrap();

        assert_eq!(assembled.real.operator.load()[1..3], [4.0, 6.0]);
        assert_eq!(assembled.imaginary_load[1..3], [6.0, 9.0]);
        assert_eq!(assembled.real.spatial_projection_evidence.len(), 0);
        assert_eq!(assembled.imaginary_projection_evidence.len(), 0);
    }
}
