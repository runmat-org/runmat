use std::collections::HashSet;

use crate::{
    AnalysisProcedure, AnalysisValidationError, CoupledPhysicsFamily, IndependentDomain,
    InitialStateConflictPolicy, PhysicsFamily, ProcedureSettings, ProcedureStep, SolutionSpace,
    StateCompleteness, StateKind, StatePersistence, StateSource, StateUse,
};

use super::step_error;

pub(super) fn validate(step: &ProcedureStep) -> Result<(), AnalysisValidationError> {
    validate_solution_space(step)?;
    validate_family_procedure(step)?;
    validate_independent_domain(step)?;
    validate_settings(step)?;
    validate_state_declarations(step)?;
    validate_required_inputs(step)
}

fn validate_settings(step: &ProcedureStep) -> Result<(), AnalysisValidationError> {
    match (&step.settings, step.procedure) {
        (ProcedureSettings::None, AnalysisProcedure::RandomResponse) => Err(step_error(
            step,
            "random response requires explicit modal damping and cumulative modal-mass controls",
        )),
        (ProcedureSettings::None, _) => Ok(()),
        (
            ProcedureSettings::StructuralRandomResponse(settings),
            AnalysisProcedure::RandomResponse,
        ) if settings.modal_damping_ratio.is_finite()
            && (0.0..1.0).contains(&settings.modal_damping_ratio)
            && settings.minimum_cumulative_mass_ratio.is_finite()
            && (0.0..=1.0).contains(&settings.minimum_cumulative_mass_ratio) =>
        {
            Ok(())
        }
        (ProcedureSettings::StructuralRandomResponse(_), AnalysisProcedure::RandomResponse) => {
            Err(step_error(
                step,
                "random-response damping must be in (0, 1) and cumulative modal mass in [0, 1]",
            ))
        }
        (ProcedureSettings::StructuralRandomResponse(_), _) => Err(step_error(
            step,
            "structural random-response settings require the random-response procedure",
        )),
        (ProcedureSettings::StructuralNonlinear(_), AnalysisProcedure::Nonlinear)
            if step.family == PhysicsFamily::Structural =>
        {
            Ok(())
        }
        (ProcedureSettings::StructuralNonlinear(_), _) => Err(step_error(
            step,
            "structural nonlinear kinematics require a structural nonlinear step",
        )),
        (
            ProcedureSettings::StructuralModalBasis(settings),
            AnalysisProcedure::ModalExtraction | AnalysisProcedure::PrestressedModalExtraction,
        ) if step.family == PhysicsFamily::Structural => {
            validate_residual_vector_directions(step, &settings.residual_vector_directions)
        }
        (ProcedureSettings::StructuralModalBasis(_), _) => Err(step_error(
            step,
            "residual vectors require a structural modal extraction step",
        )),
        (
            ProcedureSettings::StructuralModalFrequency(_),
            AnalysisProcedure::HarmonicResponse | AnalysisProcedure::PrestressedHarmonicResponse,
        ) if step.family == PhysicsFamily::Structural
            && step.solution_space == SolutionSpace::ModalReduced =>
        {
            Ok(())
        }
        (ProcedureSettings::StructuralModalFrequency(_), _) => Err(step_error(
            step,
            "static correction requires a modal-reduced structural frequency step",
        )),
    }
}

/// At most three distinct, finite, nonzero global directions (one per translational axis span).
fn validate_residual_vector_directions(
    step: &ProcedureStep,
    directions: &[crate::SpectrumDirection],
) -> Result<(), AnalysisValidationError> {
    if directions.is_empty() || directions.len() > 3 {
        return Err(step_error(
            step,
            "residual vectors name one to three directions",
        ));
    }
    let mut unit = Vec::with_capacity(directions.len());
    for direction in directions {
        let norm = direction.vector.iter().map(|value| value * value).sum::<f64>().sqrt();
        if direction.frame != crate::CoordinateFrame::Global
            || !norm.is_finite()
            || norm <= f64::EPSILON
        {
            return Err(step_error(
                step,
                "residual-vector directions are finite, nonzero and global",
            ));
        }
        let candidate = direction.vector.map(|value| value / norm);
        if unit.iter().any(|existing: &[f64; 3]| {
            existing
                .iter()
                .zip(candidate)
                .all(|(left, right)| (left - right).abs() <= 1.0e-12)
        }) {
            return Err(step_error(step, "residual-vector directions are distinct"));
        }
        unit.push(candidate);
    }
    Ok(())
}

fn validate_family_procedure(step: &ProcedureStep) -> Result<(), AnalysisValidationError> {
    // The closed family x procedure registry is the one authority: a pending identity is
    // defined (admission reports its pending id); an absent one is not a procedure of the family.
    let supported =
        crate::analysis_identity_status(step.family, step.procedure, step.solution_space).is_some();
    if supported {
        Ok(())
    } else {
        Err(step_error(
            step,
            format!(
                "procedure {:?} is not defined for physics family {:?}",
                step.procedure, step.family
            ),
        ))
    }
}

fn validate_solution_space(step: &ProcedureStep) -> Result<(), AnalysisValidationError> {
    let modal_reduced = step.solution_space == SolutionSpace::ModalReduced;
    let accepts_modal = step.family == PhysicsFamily::Structural
        && matches!(
            step.procedure,
            AnalysisProcedure::TransientResponse
                | AnalysisProcedure::HarmonicResponse
                | AnalysisProcedure::PrestressedHarmonicResponse
                | AnalysisProcedure::ModalResponseSpectrum
                | AnalysisProcedure::RandomResponse
        );
    let requires_modal = matches!(
        step.procedure,
        AnalysisProcedure::ModalResponseSpectrum | AnalysisProcedure::RandomResponse
    );
    if (modal_reduced && !accepts_modal) || (requires_modal && !modal_reduced) {
        Err(step_error(
            step,
            format!(
                "solution space {:?} is incompatible with procedure {:?} for {:?}",
                step.solution_space, step.procedure, step.family
            ),
        ))
    } else {
        Ok(())
    }
}

fn validate_independent_domain(step: &ProcedureStep) -> Result<(), AnalysisValidationError> {
    let domain_matches = match step.procedure {
        AnalysisProcedure::Static | AnalysisProcedure::LineParameters => {
            matches!(step.independent_domain, IndependentDomain::Static)
        }
        AnalysisProcedure::ModalExtraction
        | AnalysisProcedure::PrestressedModalExtraction
        | AnalysisProcedure::EigenvalueBuckling => {
            matches!(
                step.independent_domain,
                IndependentDomain::Eigenvalue { .. }
            )
        }
        AnalysisProcedure::TransientResponse => {
            matches!(
                step.independent_domain,
                IndependentDomain::TimeSeconds { .. }
            )
        }
        AnalysisProcedure::HarmonicResponse | AnalysisProcedure::PrestressedHarmonicResponse => {
            matches!(
                step.independent_domain,
                IndependentDomain::FrequencyHertz(_)
            )
        }
        AnalysisProcedure::RandomResponse => {
            matches!(
                step.independent_domain,
                IndependentDomain::RandomExcitation { .. }
            )
        }
        AnalysisProcedure::ModalResponseSpectrum => {
            matches!(
                step.independent_domain,
                IndependentDomain::ResponseSpectra { .. }
            )
        }
        AnalysisProcedure::Nonlinear => matches!(
            step.independent_domain,
            IndependentDomain::LoadFactor { .. } | IndependentDomain::TimeSeconds { .. }
        ),
        AnalysisProcedure::Adaptive => !matches!(
            step.independent_domain,
            IndependentDomain::Eigenvalue { .. }
        ),
    };
    if !domain_matches {
        return Err(step_error(
            step,
            format!(
                "independent domain {:?} is incompatible with procedure {:?}",
                step.independent_domain, step.procedure
            ),
        ));
    }

    let valid = match &step.independent_domain {
        IndependentDomain::Static => true,
        IndependentDomain::TimeSeconds {
            origin,
            start,
            end,
            interval_count,
        } => {
            origin.is_finite()
                && start.is_finite()
                && end.is_finite()
                && start >= origin
                && end > start
                && *interval_count > 0
        }
        IndependentDomain::FrequencyHertz(values) => {
            !values.is_empty()
                && values
                    .iter()
                    .all(|value| value.is_finite() && *value >= 0.0)
                && strictly_increasing(values)
        }
        IndependentDomain::LoadFactor { start, end } => {
            start.is_finite() && end.is_finite() && end > start
        }
        IndependentDomain::Eigenvalue { requested_count } => *requested_count > 0,
        IndependentDomain::StudyParameter {
            parameter_id,
            canonical_values,
        } => {
            !parameter_id.trim().is_empty()
                && !canonical_values.is_empty()
                && canonical_values.iter().all(|value| value.is_finite())
                && strictly_increasing(canonical_values)
        }
        IndependentDomain::ResponseSpectra { spectrum_ids } => {
            !spectrum_ids.is_empty()
                && spectrum_ids
                    .iter()
                    .all(|spectrum_id| !spectrum_id.0.trim().is_empty())
                && spectrum_ids.iter().collect::<HashSet<_>>().len() == spectrum_ids.len()
        }
        IndependentDomain::RandomExcitation { excitation_id } => !excitation_id.0.trim().is_empty(),
    };
    if valid {
        Ok(())
    } else {
        Err(step_error(
            step,
            "independent domain must be finite, non-empty, and strictly ordered",
        ))
    }
}

fn strictly_increasing(values: &[f64]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

fn validate_state_declarations(step: &ProcedureStep) -> Result<(), AnalysisValidationError> {
    let mut input_ids = HashSet::new();
    for input in &step.state_inputs {
        if input.input_id.trim().is_empty() || !input_ids.insert(input.input_id.as_str()) {
            return Err(step_error(
                step,
                "state input ids must be non-empty and unique",
            ));
        }
        if matches!(input.source, StateSource::InitialConditions)
            && input.conflict != InitialStateConflictPolicy::Reject
        {
            return Err(step_error(
                step,
                format!(
                    "initial-condition input `{}` cannot declare an upstream conflict override",
                    input.input_id
                ),
            ));
        }
        if !state_use_accepts_kind(input.use_as, input.required_kind) {
            return Err(step_error(
                step,
                format!(
                    "state input `{}` cannot use {:?} state as {:?}",
                    input.input_id, input.required_kind, input.use_as
                ),
            ));
        }
        if !family_can_consume_state(step.family, input.required_kind) {
            return Err(step_error(
                step,
                format!(
                    "physics family {:?} cannot consume {:?} state",
                    step.family, input.required_kind
                ),
            ));
        }
    }

    if step.state_outputs.is_empty() {
        return Err(step_error(
            step,
            "step must publish at least one typed state",
        ));
    }
    let mut output_ids = HashSet::new();
    for output in &step.state_outputs {
        let id = output.state_id.0.as_str();
        if id.trim().is_empty() || !output_ids.insert(id) {
            return Err(step_error(
                step,
                "state output ids must be non-empty and unique",
            ));
        }
        if output.completeness == StateCompleteness::Complete
            && output.persistence == StatePersistence::CheckpointOnly
        {
            return Err(step_error(
                step,
                format!("complete state `{id}` cannot be checkpoint-only"),
            ));
        }
        if output.completeness == StateCompleteness::Checkpoint
            && output.persistence != StatePersistence::CheckpointOnly
        {
            return Err(step_error(
                step,
                format!("checkpoint state `{id}` must be checkpoint-only"),
            ));
        }
        if !family_can_publish_state(step.family, output.kind) {
            return Err(step_error(
                step,
                format!(
                    "physics family {:?} cannot publish {:?} state",
                    step.family, output.kind
                ),
            ));
        }
    }
    Ok(())
}

fn validate_required_inputs(step: &ProcedureStep) -> Result<(), AnalysisValidationError> {
    let requires_prestress = matches!(
        step.procedure,
        AnalysisProcedure::PrestressedModalExtraction
            | AnalysisProcedure::EigenvalueBuckling
            | AnalysisProcedure::PrestressedHarmonicResponse
    );
    if requires_prestress
        && !step.state_inputs.iter().any(|input| {
            input.use_as == StateUse::Prestress
                && input.required_kind == StateKind::GeometricStiffness
        })
    {
        return Err(step_error(
            step,
            "procedure requires an explicit geometric-stiffness prestress state input",
        ));
    }
    let requires_basis = step.solution_space == SolutionSpace::ModalReduced;
    if requires_basis
        && !step
            .state_inputs
            .iter()
            .any(|input| input.use_as == StateUse::ModalBasis)
    {
        return Err(step_error(
            step,
            "modal-reduced procedure requires a modal-basis state input",
        ));
    }
    Ok(())
}

pub(super) fn family_can_consume_state(family: PhysicsFamily, kind: StateKind) -> bool {
    family_can_publish_state(family, kind)
        || matches!(
            (family, kind),
            (PhysicsFamily::Structural, StateKind::Thermal)
                | (PhysicsFamily::Structural, StateKind::CoupledInterface)
                | (PhysicsFamily::Thermal, StateKind::ElectromagneticStatic)
                | (PhysicsFamily::Thermal, StateKind::ElectromagneticComplex)
                | (PhysicsFamily::Thermal, StateKind::Fluid)
                | (PhysicsFamily::Thermal, StateKind::CoupledInterface)
                | (PhysicsFamily::Fluid, StateKind::CoupledInterface)
        )
}

fn family_can_publish_state(family: PhysicsFamily, kind: StateKind) -> bool {
    match family {
        PhysicsFamily::Structural => matches!(
            kind,
            StateKind::StructuralPrimary
                | StateKind::StructuralKinematics
                | StateKind::StructuralStressStrain
                | StateKind::StructuralComplex
                | StateKind::StructuralResponseEnvelope
                | StateKind::StructuralRandomStatistics
                | StateKind::GeometricStiffness
                | StateKind::MaterialHistory
                | StateKind::ContactHistory
                | StateKind::ModalBasis
                | StateKind::BucklingModes
                | StateKind::Integrator
                | StateKind::MovingMesh
        ),
        PhysicsFamily::Thermal => matches!(kind, StateKind::Thermal | StateKind::Integrator),
        PhysicsFamily::Electromagnetic => matches!(
            kind,
            StateKind::ElectromagneticStatic
                | StateKind::ElectromagneticComplex
                | StateKind::Integrator
        ),
        PhysicsFamily::Acoustic => {
            matches!(
                kind,
                StateKind::AcousticComplex
                    | StateKind::AcousticModes
                    | StateKind::AcousticPressure
                    | StateKind::Integrator
            )
        }
        PhysicsFamily::Fluid => matches!(
            kind,
            StateKind::Fluid | StateKind::MovingMesh | StateKind::Integrator
        ),
        PhysicsFamily::Coupled(coupled) => match coupled {
            CoupledPhysicsFamily::Thermomechanical => matches!(
                kind,
                StateKind::StructuralPrimary
                    | StateKind::StructuralKinematics
                    | StateKind::StructuralStressStrain
                    | StateKind::GeometricStiffness
                    | StateKind::MaterialHistory
                    | StateKind::ContactHistory
                    | StateKind::Thermal
                    | StateKind::CoupledInterface
                    | StateKind::Integrator
            ),
            CoupledPhysicsFamily::Electrothermal => matches!(
                kind,
                StateKind::Thermal
                    | StateKind::ElectromagneticStatic
                    | StateKind::ElectromagneticComplex
                    | StateKind::CoupledInterface
                    | StateKind::Integrator
            ),
            CoupledPhysicsFamily::ConjugateHeatTransfer => matches!(
                kind,
                StateKind::Thermal
                    | StateKind::Fluid
                    | StateKind::CoupledInterface
                    | StateKind::Integrator
            ),
            CoupledPhysicsFamily::FluidStructureInteraction => matches!(
                kind,
                StateKind::StructuralPrimary
                    | StateKind::StructuralKinematics
                    | StateKind::StructuralStressStrain
                    | StateKind::Fluid
                    | StateKind::MovingMesh
                    | StateKind::CoupledInterface
                    | StateKind::Integrator
            ),
        },
    }
}

fn state_use_accepts_kind(use_as: StateUse, kind: StateKind) -> bool {
    match use_as {
        StateUse::Prestress => matches!(
            kind,
            StateKind::StructuralPrimary
                | StateKind::StructuralStressStrain
                | StateKind::GeometricStiffness
                | StateKind::Thermal
        ),
        StateUse::ModalBasis => kind == StateKind::ModalBasis,
        StateUse::CoupledInterface => kind == StateKind::CoupledInterface,
        StateUse::HeatSource => matches!(
            kind,
            StateKind::ElectromagneticStatic | StateKind::ElectromagneticComplex
        ),
        StateUse::PhysicalInitialState
        | StateUse::LinearizationPoint
        | StateUse::InitialGuess
        | StateUse::Restart => true,
    }
}
