//! Pre-mesh study admission: the static rules every run path enforces, shared by `runmat check`
//! (through the runtime's checked source) and native preparation. Each rule decides from the
//! realized study (authored model, sequence and execution policy) alone; nothing here needs
//! geometry, a mesh or a solve. Kernels keep their canonical-input checks as a backstop and
//! report the same rule text from [`StaticRule`].

use std::collections::BTreeMap;

use runmat_analysis_core::{
    analysis_identity_status, family_label, procedure_label, AnalysisIdentityStatus,
    AnalysisProcedure, AnalysisSequence, AuthoredConstraintKind, AuthoredInitialConditionKind,
    AuthoredMaterialRef, AuthoredMaterialSource, AuthoredPairingPolicy, AuthoredPeriodicTransform,
    CoupledPhysicsFamily, ElectromagneticDomain, MaterialModel, PhysicsFamily, RealizedStudy,
    RestartPolicy, SolutionSpace,
};

/// One static admission rule: a stable diagnostic code, the pending-feature id when the
/// construct is planned, and the message both `check` and the run report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StaticRule {
    pub code: &'static str,
    pub pending: Option<&'static str>,
    pub message: &'static str,
}

impl StaticRule {
    pub fn issue(self) -> AdmissionIssue {
        AdmissionIssue {
            code: self.code,
            pending: self.pending,
            message: self.message.to_string(),
        }
    }
}

/// A static admission failure, reported identically by `runmat check` and `runmat run`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmissionIssue {
    pub code: &'static str,
    pub pending: Option<&'static str>,
    pub message: String,
}

impl AdmissionIssue {
    /// The message with its pending key, e.g. `... [pending:fluid.harmonic]`.
    pub fn display_message(&self) -> String {
        match self.pending {
            Some(id) => format!("{} [pending:{id}]", self.message),
            None => self.message.clone(),
        }
    }
}

impl std::fmt::Display for AdmissionIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code, self.display_message())
    }
}

impl std::error::Error for AdmissionIssue {}

pub const FAMILY_PROCEDURE_CODE: &str = "fea.admission.family_procedure";
pub const PROCEDURE_GRAPH_CODE: &str = "fea.admission.procedure_graph";

pub const FLUID_PERIODIC: StaticRule = StaticRule {
    code: "fea.admission.constraint",
    pending: Some("fluid.periodic"),
    message: "periodic fluid boundary (fluid.periodic) is not supported by the fluid solver",
};
pub const FLUID_INITIAL_CONDITION: StaticRule = StaticRule {
    code: "fea.admission.initial_condition",
    pending: None,
    message: "nonfluid initial condition: a fluid analysis starts from fluid velocity and pressure initial conditions only",
};
pub const MAGNETIC_STATIC_CONSTRAINT: StaticRule = StaticRule {
    code: "fea.admission.constraint",
    pending: Some("magnetic.periodic"),
    message: "nonzero, periodic or phase magnetic boundary: magnetostatics admits magnetic.vector_potential_ground and magnetic.coulomb_gauge",
};
pub const MAGNETIC_STATIC_GAUGE: StaticRule = StaticRule {
    code: "fea.admission.gauge",
    pending: None,
    message: "the curl-curl operator has no unique vector potential without a gauge: add magnetic.coulomb_gauge",
};
pub const MAXWELL_CONSTRAINT: StaticRule = StaticRule {
    code: "fea.admission.constraint",
    pending: Some("magnetic.harmonic.vector-potential"),
    message: "harmonic analysis admits magnetic.vector_potential_ground; conductivity and permittivity fix the gauge, so magnetic.coulomb_gauge, prescribed or periodic vector potentials are not admitted",
};
pub const MAXWELL_NONELECTROMAGNETIC_CONSTRAINT: StaticRule = StaticRule {
    code: "fea.admission.constraint",
    pending: None,
    message: "non-electromagnetic constraint in a harmonic analysis",
};
pub const MAXWELL_FAR_FIELD: StaticRule = StaticRule {
    code: "fea.admission.far_field",
    pending: None,
    message: "a harmonic analysis needs a far field: model the surrounding air and ground its outer boundary with magnetic.vector_potential_ground",
};
pub const ACOUSTIC_ROTATED_PERIODIC: StaticRule = StaticRule {
    code: "fea.admission.constraint",
    pending: Some("acoustic.periodic.rotational"),
    message: "nonmatching or rotated periodic pressure relation",
};
pub const ACOUSTIC_TRANSIENT_INITIAL_CONDITION: StaticRule = StaticRule {
    code: "fea.admission.initial_condition",
    pending: None,
    message:
        "acoustic transients start from rest; drive them with boundary pressure or velocity signals",
};
pub const ACOUSTIC_TRANSIENT_CONSTRAINT: StaticRule = StaticRule {
    code: "fea.admission.constraint",
    pending: None,
    message: "periodic or non-acoustic constraints in a pressure transient",
};
pub const ELECTROMAGNETIC_INITIAL_CONDITION: StaticRule = StaticRule {
    code: "fea.admission.initial_condition",
    pending: None,
    message: "electromagnetic static and harmonic analyses take no initial conditions",
};
pub const MATERIAL_MASS_DENSITY_CODE: &str = "fea.admission.material_mass_density";
pub const OBSERVATION_CODE: &str = "fea.admission.observation";
pub const ACCELERATOR_REQUIRED_CODE: &str = "fea.admission.accelerator_required";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmittedFamily {
    Acoustic,
    AcousticResponse,
    Electrical,
    Fluid,
    /// Per-length line parameters of a cross-section: solved in place on the section's
    /// triangle mesh, with no exact geometry or volume mesh.
    LineParameters,
    Nonlinear,
    Magnetic,
    Maxwell,
    Structural,
    Thermal,
}

/// Every static rule for one realized study, in order: family x procedure and procedure graph,
/// then the selected family's constraint, initial-condition and observation admission, then
/// strict accelerator placement. Returns the family the run prepares, or every issue found.
pub fn admit_study(
    realized: &RealizedStudy,
    materials: &BTreeMap<AuthoredMaterialRef, MaterialModel>,
) -> Result<AdmittedFamily, Vec<AdmissionIssue>> {
    let family = select_family(&realized.sequence, realized.model.electromagnetic.as_ref())
        .map_err(|issue| vec![issue])?;
    let mut issues = Vec::new();
    admit_materials(realized, materials, &mut issues);
    admit_constraints(realized, family, &mut issues);
    admit_initial_conditions(realized, family, &mut issues);
    admit_observations(realized, family, &mut issues);
    if issues.is_empty() {
        Ok(family)
    } else {
        Err(issues)
    }
}

fn admit_constraints(
    realized: &RealizedStudy,
    family: AdmittedFamily,
    issues: &mut Vec<AdmissionIssue>,
) {
    let constraints = &realized.model.constraints;
    let mut push = |rule: StaticRule| {
        let issue = rule.issue();
        if !issues.contains(&issue) {
            issues.push(issue);
        }
    };
    match family {
        AdmittedFamily::Fluid => {
            if constraints
                .iter()
                .any(|c| matches!(c.kind, AuthoredConstraintKind::FluidPeriodic(_)))
            {
                push(FLUID_PERIODIC);
            }
        }
        AdmittedFamily::Magnetic => {
            let mut gauged = false;
            for constraint in constraints {
                match &constraint.kind {
                    AuthoredConstraintKind::MagneticVectorPotentialGround { .. } => {}
                    AuthoredConstraintKind::MagneticCoulombGauge { .. } => gauged = true,
                    _ => push(MAGNETIC_STATIC_CONSTRAINT),
                }
            }
            if !gauged {
                push(MAGNETIC_STATIC_GAUGE);
            }
        }
        AdmittedFamily::Maxwell => {
            let mut far_field = false;
            for constraint in constraints {
                match &constraint.kind {
                    AuthoredConstraintKind::MagneticVectorPotentialGround { .. }
                    | AuthoredConstraintKind::ElectricPerfectConductor { .. } => far_field = true,
                    // A prescribed potential on a conductor face is a drive terminal
                    // (physics/maxwell/drive.rs), not a field constraint.
                    AuthoredConstraintKind::ElectricalPotential { .. } => {}
                    AuthoredConstraintKind::MagneticVectorPotential { .. }
                    | AuthoredConstraintKind::MagneticCoulombGauge { .. }
                    | AuthoredConstraintKind::MagneticPeriodic(_)
                    | AuthoredConstraintKind::MagneticPhaseShiftedPeriodic(_) => {
                        push(MAXWELL_CONSTRAINT)
                    }
                    _ => push(MAXWELL_NONELECTROMAGNETIC_CONSTRAINT),
                }
            }
            if !far_field {
                push(MAXWELL_FAR_FIELD);
            }
        }
        AdmittedFamily::Acoustic | AdmittedFamily::AcousticResponse => {
            let transient = realized
                .sequence
                .steps
                .iter()
                .any(|step| step.procedure == AnalysisProcedure::TransientResponse);
            for constraint in constraints {
                let periodic = match &constraint.kind {
                    AuthoredConstraintKind::AcousticPressure { .. } => continue,
                    AuthoredConstraintKind::AcousticPeriodic(periodic) => periodic,
                    AuthoredConstraintKind::AcousticPhaseShiftedPeriodic(bloch) => &bloch.periodic,
                    // Other families' constraints are rejected by family composition checks.
                    _ => continue,
                };
                if transient {
                    push(ACOUSTIC_TRANSIENT_CONSTRAINT);
                } else if !matches!(periodic.pair.pairing, AuthoredPairingPolicy::Matching)
                    || matches!(
                        periodic.transform,
                        AuthoredPeriodicTransform::Rotation { .. }
                    )
                {
                    push(ACOUSTIC_ROTATED_PERIODIC);
                }
            }
        }
        _ => {}
    }
}

fn admit_initial_conditions(
    realized: &RealizedStudy,
    family: AdmittedFamily,
    issues: &mut Vec<AdmissionIssue>,
) {
    let conditions = &realized.model.initial_conditions;
    if conditions.is_empty() {
        return;
    }
    let rule = match family {
        AdmittedFamily::Fluid => {
            // Conjugate heat transfer also starts its solid and fluid temperatures here.
            let conjugate = realized.sequence.steps.iter().any(|step| {
                step.family == PhysicsFamily::Coupled(CoupledPhysicsFamily::ConjugateHeatTransfer)
            });
            conditions
                .iter()
                .any(|condition| {
                    !matches!(
                        condition.kind,
                        AuthoredInitialConditionKind::FluidVelocity(_)
                            | AuthoredInitialConditionKind::FluidPressure(_)
                    ) && !(conjugate
                        && matches!(condition.kind, AuthoredInitialConditionKind::Temperature(_)))
                })
                .then_some(FLUID_INITIAL_CONDITION)
        }
        AdmittedFamily::AcousticResponse
            if realized
                .sequence
                .steps
                .iter()
                .any(|step| step.procedure == AnalysisProcedure::TransientResponse) =>
        {
            Some(ACOUSTIC_TRANSIENT_INITIAL_CONDITION)
        }
        AdmittedFamily::Magnetic | AdmittedFamily::Maxwell => {
            Some(ELECTROMAGNETIC_INITIAL_CONDITION)
        }
        _ => None,
    };
    if let Some(rule) = rule {
        issues.push(rule.issue());
    }
}

/// Structural procedures with inertia (modal, transient, harmonic, spectrum and random
/// response) assemble a mass matrix, so every assigned material must carry `mass_density`.
fn needs_mass(step: &runmat_analysis_core::ProcedureStep) -> bool {
    step.family == PhysicsFamily::Structural
        && matches!(
            step.procedure,
            AnalysisProcedure::ModalExtraction
                | AnalysisProcedure::PrestressedModalExtraction
                | AnalysisProcedure::TransientResponse
                | AnalysisProcedure::HarmonicResponse
                | AnalysisProcedure::PrestressedHarmonicResponse
                | AnalysisProcedure::ModalResponseSpectrum
                | AnalysisProcedure::RandomResponse
        )
}

fn admit_materials(
    realized: &RealizedStudy,
    materials: &BTreeMap<AuthoredMaterialRef, MaterialModel>,
    issues: &mut Vec<AdmissionIssue>,
) {
    let Some(step) = realized.sequence.steps.iter().find(|step| needs_mass(step)) else {
        return;
    };
    let assigned = std::iter::once(&realized.model)
        .chain(&realized.included_models)
        .flat_map(|model| &model.material_assignments)
        .filter_map(|assignment| match &assignment.material {
            AuthoredMaterialSource::Declaration(reference) => Some(reference),
            AuthoredMaterialSource::ModelInput(_) => None,
        })
        .collect::<std::collections::BTreeSet<_>>();
    for reference in assigned {
        let Some(material) = materials.get(reference) else {
            continue;
        };
        if material.mass_density_kg_per_m3.is_none() {
            issues.push(AdmissionIssue {
                code: MATERIAL_MASS_DENSITY_CODE,
                pending: None,
                message: format!(
                    "analysis `{}` ({} {}) assembles mass, but material `{}` declares no mass_density",
                    step.step_id.as_str(),
                    family_label(step.family),
                    procedure_label(step.procedure),
                    reference.material_id
                ),
            });
        }
    }
}

fn admit_observations(
    realized: &RealizedStudy,
    family: AdmittedFamily,
    issues: &mut Vec<AdmissionIssue>,
) {
    // Families that publish sampled native fields select observations from them only.
    if !matches!(family, AdmittedFamily::Magnetic | AdmittedFamily::Maxwell) {
        return;
    }
    for step in &realized.sequence.steps {
        if let Err(error) = crate::results::observation::validate_sampled_requests(step) {
            issues.push(AdmissionIssue {
                code: OBSERVATION_CODE,
                pending: None,
                message: format!("analysis `{}`: {error}", step.step_id.as_str()),
            });
        }
    }
}

/// Strict accelerator placement (`accelerators = required`) on a study whose families have no
/// device solve path cannot run anywhere: reported before meshing. Whether a host can allocate
/// a device stays a run-time decision.
pub fn admit_accelerator_policy(
    steps: &[runmat_analysis_core::ProcedureStep],
    policy: Option<&runmat_analysis_core::AuthoredExecutionPolicy>,
) -> Result<(), AdmissionIssue> {
    use crate::contracts::accelerator_admission::{
        first_family_without_device_path, AcceleratorFamily,
    };
    use runmat_analysis_core::{
        AcceleratorAdmissionPolicy, AcceleratorPreference, BackendPreference,
    };
    let Some(policy) = policy else {
        return Ok(());
    };
    if policy.resources.accelerator_admission() != AcceleratorAdmissionPolicy::Required
        || policy.backend == BackendPreference::Cpu
        || policy.resources.accelerators == AcceleratorPreference::Disabled
    {
        return Ok(());
    }
    match first_family_without_device_path(steps.iter().map(AcceleratorFamily::of_step)) {
        None => Ok(()),
        Some(family) => Err(AdmissionIssue {
            code: ACCELERATOR_REQUIRED_CODE,
            pending: None,
            message: format!(
                "the study requires the accelerator (`accelerators = required`), but {} execution has no device path: {}",
                family.label(),
                family.unavailable_reason().unwrap_or_default()
            ),
        }),
    }
}

/// Family dispatch for the native run. Exact material/source/mesh/numerical admission stays with
/// the compiled study product and canonical worker.
pub fn select_family(
    sequence: &AnalysisSequence,
    electromagnetic: Option<&ElectromagneticDomain>,
) -> Result<AdmittedFamily, AdmissionIssue> {
    for step in &sequence.steps {
        match analysis_identity_status(step.family, step.procedure, step.solution_space) {
            Some(AnalysisIdentityStatus::Runs) => {}
            status => {
                return Err(AdmissionIssue {
                    code: FAMILY_PROCEDURE_CODE,
                    pending: match status {
                        Some(AnalysisIdentityStatus::Pending(id)) => Some(id),
                        _ => None,
                    },
                    message: format!(
                        "analysis `{}`: {} {} ({}) does not run in this release",
                        step.step_id.as_str(),
                        family_label(step.family),
                        procedure_label(step.procedure),
                        match step.solution_space {
                            SolutionSpace::Direct => "direct",
                            SolutionSpace::ModalReduced => "modal reduced",
                        }
                    ),
                });
            }
        }
    }
    if let [step] = sequence.steps.as_slice() {
        if step.family == PhysicsFamily::Structural
            && step.procedure == AnalysisProcedure::Nonlinear
            && step.solution_space == SolutionSpace::Direct
            && step.restart == RestartPolicy::Fresh
            && step.state_inputs.is_empty()
            && step.activations.is_empty()
            && sequence.dependencies.is_empty()
            && sequence.transitions.is_empty()
            && sequence.branches.is_empty()
        {
            return Ok(AdmittedFamily::Nonlinear);
        }
    }
    if sequence.steps.iter().any(|step| {
        step.family == PhysicsFamily::Structural && step.procedure == AnalysisProcedure::Nonlinear
    }) {
        return Err(AdmissionIssue {
            code: PROCEDURE_GRAPH_CODE,
            pending: Some("structural.nonlinear.sequence"),
            message: "a structural nonlinear analysis runs as one fresh direct analysis; nonlinear steps inside procedure graphs are not supported".into(),
        });
    }
    if structural_root_sequence(sequence) {
        return Ok(AdmittedFamily::Structural);
    }
    let [step] = sequence.steps.as_slice() else {
        return Err(unsupported());
    };
    // Flow coupled to heat transfer or a structure runs the same flow study lifecycle.
    let conjugate_heat = step.family
        == PhysicsFamily::Coupled(
            runmat_analysis_core::CoupledPhysicsFamily::ConjugateHeatTransfer,
        )
        || step.family
            == PhysicsFamily::Coupled(
                runmat_analysis_core::CoupledPhysicsFamily::FluidStructureInteraction,
            );
    if (step.family == PhysicsFamily::Fluid || conjugate_heat)
        && matches!(
            (step.procedure, &step.independent_domain),
            (
                AnalysisProcedure::Static,
                runmat_analysis_core::IndependentDomain::Static
            ) | (
                AnalysisProcedure::TransientResponse,
                runmat_analysis_core::IndependentDomain::TimeSeconds { .. }
            )
        )
        && step.solution_space == SolutionSpace::Direct
        && step.restart == RestartPolicy::Fresh
        && step.state_inputs.is_empty()
        && step.activations.is_empty()
        && sequence.dependencies.is_empty()
        && sequence.transitions.is_empty()
        && sequence.branches.is_empty()
    {
        return Ok(AdmittedFamily::Fluid);
    }
    if step.family == PhysicsFamily::Electromagnetic
        && step.procedure == AnalysisProcedure::HarmonicResponse
        && step.solution_space == SolutionSpace::Direct
        && step.restart == RestartPolicy::Fresh
        && step.state_inputs.is_empty()
        && step.activations.is_empty()
        && sequence.dependencies.is_empty()
        && sequence.transitions.is_empty()
        && sequence.branches.is_empty()
    {
        if !matches!(
            electromagnetic,
            None | Some(ElectromagneticDomain::Harmonic { .. })
        ) {
            return Err(unsupported());
        }
        // A multi-frequency study runs one case per frequency (fanned out before preparation).
        let runmat_analysis_core::IndependentDomain::FrequencyHertz(frequencies) =
            &step.independent_domain
        else {
            return Err(unsupported());
        };
        if frequencies.is_empty()
            || frequencies
                .iter()
                .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
        {
            return Err(unsupported());
        }
        return Ok(AdmittedFamily::Maxwell);
    }
    if step.family == PhysicsFamily::Electromagnetic
        && step.procedure == AnalysisProcedure::Static
        && step.solution_space == SolutionSpace::Direct
        && step.restart == RestartPolicy::Fresh
        && step.state_inputs.is_empty()
        && step.activations.is_empty()
        && sequence.dependencies.is_empty()
        && sequence.transitions.is_empty()
        && sequence.branches.is_empty()
    {
        return match electromagnetic {
            Some(ElectromagneticDomain::Static { .. }) => Ok(AdmittedFamily::Electrical),
            Some(ElectromagneticDomain::Magnetostatic { .. }) => Ok(AdmittedFamily::Magnetic),
            Some(ElectromagneticDomain::Harmonic { .. }) | None => Err(unsupported()),
        };
    }
    if step.family == PhysicsFamily::Acoustic
        && step.procedure == AnalysisProcedure::HarmonicResponse
        && step.solution_space == SolutionSpace::Direct
        && step.restart == RestartPolicy::Fresh
        && step.state_inputs.is_empty()
        && step.activations.is_empty()
        && sequence.dependencies.is_empty()
        && sequence.transitions.is_empty()
        && sequence.branches.is_empty()
    {
        return Ok(AdmittedFamily::Acoustic);
    }
    if step.family == PhysicsFamily::Electromagnetic
        && step.procedure == AnalysisProcedure::LineParameters
        && step.solution_space == SolutionSpace::Direct
        && step.restart == RestartPolicy::Fresh
        && step.state_inputs.is_empty()
        && step.activations.is_empty()
        && sequence.dependencies.is_empty()
        && sequence.transitions.is_empty()
        && sequence.branches.is_empty()
    {
        return Ok(AdmittedFamily::LineParameters);
    }
    if is_electrothermal(sequence) {
        return Ok(AdmittedFamily::Thermal);
    }
    if step.family == PhysicsFamily::Acoustic
        && matches!(
            step.procedure,
            AnalysisProcedure::ModalExtraction | AnalysisProcedure::TransientResponse
        )
        && step.solution_space == SolutionSpace::Direct
        && step.restart == RestartPolicy::Fresh
        && step.state_inputs.is_empty()
        && step.activations.is_empty()
        && sequence.dependencies.is_empty()
        && sequence.transitions.is_empty()
        && sequence.branches.is_empty()
    {
        return Ok(AdmittedFamily::AcousticResponse);
    }
    if step.family != PhysicsFamily::Thermal
        || !matches!(
            step.procedure,
            AnalysisProcedure::Static | AnalysisProcedure::TransientResponse
        )
        || step.solution_space != SolutionSpace::Direct
        || step.restart != RestartPolicy::Fresh
        || !crate::study::thermal_step_initial_selection_supported(step)
        || !sequence.dependencies.is_empty()
        || !sequence.transitions.is_empty()
        || !sequence.branches.is_empty()
    {
        return Err(unsupported());
    }
    Ok(AdmittedFamily::Thermal)
}

/// One fresh electrothermal (Joule heating) analysis, steady or transient.
pub fn is_electrothermal(sequence: &AnalysisSequence) -> bool {
    matches!(sequence.steps.as_slice(), [step]
        if step.family == PhysicsFamily::Coupled(runmat_analysis_core::CoupledPhysicsFamily::Electrothermal)
            && matches!(step.procedure, AnalysisProcedure::Static | AnalysisProcedure::TransientResponse)
            && step.solution_space == SolutionSpace::Direct
            && step.restart == RestartPolicy::Fresh
            && sequence.dependencies.is_empty()
            && sequence.transitions.is_empty()
            && sequence.branches.is_empty())
}

/// The structural study root owns structural graphs, thermomechanical steps, and a thermal
/// producer that feeds a thermomechanical step. The exact thermal-to-preload binding and the
/// supported consumers are validated by structural workload planning, not by this family gate.
fn structural_root_sequence(sequence: &AnalysisSequence) -> bool {
    let mechanical = |step: &runmat_analysis_core::ProcedureStep| {
        matches!(
            step.family,
            PhysicsFamily::Structural
                | PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical)
        )
    };
    let thermal = sequence
        .steps
        .iter()
        .filter(|step| step.family == PhysicsFamily::Thermal)
        .count();
    let thermomechanical = sequence
        .steps
        .iter()
        .any(|step| step.family == PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical));
    sequence
        .steps
        .iter()
        .all(|step| mechanical(step) || step.family == PhysicsFamily::Thermal)
        && (thermal == 0 || (thermal == 1 && thermomechanical))
}

fn unsupported() -> AdmissionIssue {
    AdmissionIssue {
        code: PROCEDURE_GRAPH_CODE,
        pending: None,
        message: "native execution runs structural studies (including a thermal prefix feeding a thermomechanical step), or one fresh independent direct thermal steady/transient, fluid steady/transient, conjugate heat transfer or fluid-structure interaction steady/transient, equation-selected static electrical/magnetic, Maxwell harmonic, electrothermal steady/transient, or acoustic harmonic, modal or transient analysis".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_analysis_core::ProcedureStep;

    #[test]
    fn nonlinear_dispatch_preserves_fresh_standalone_authority() {
        let step = ProcedureStep::structural_nonlinear("plastic");
        assert_eq!(
            super::select_family(
                &AnalysisSequence::from_steps("plastic", vec![step.clone()]),
                None
            )
            .unwrap(),
            AdmittedFamily::Nonlinear
        );
        let mut resumed = step.clone();
        resumed.restart = RestartPolicy::ResumeCheckpoint {
            checkpoint_id: "foreign".into(),
            required_state_digest: "foreign".into(),
        };
        let mut reduced = step.clone();
        reduced.solution_space = SolutionSpace::ModalReduced;
        for steps in [vec![resumed], vec![reduced], vec![step.clone(), step]] {
            assert!(
                super::select_family(&AnalysisSequence::from_steps("plastic", steps), None)
                    .is_err()
            );
        }
    }

    #[test]
    fn fluid_dispatch_requires_one_fresh_independent_direct_steady_or_timed_step() {
        let step = ProcedureStep::fluid_static("flow");
        let sequence = AnalysisSequence::from_steps("flow", vec![step.clone()]);
        assert_eq!(
            super::select_family(&sequence, None).unwrap(),
            AdmittedFamily::Fluid
        );
        let mut transient = step.clone();
        transient.procedure = AnalysisProcedure::TransientResponse;
        let mut timed = transient.clone();
        timed.independent_domain = runmat_analysis_core::IndependentDomain::TimeSeconds {
            origin: 0.0,
            start: 0.0,
            end: 1.0,
            interval_count: 10,
        };
        assert_eq!(
            super::select_family(&AnalysisSequence::from_steps("flow", vec![timed]), None).unwrap(),
            AdmittedFamily::Fluid
        );
        let mut resumed = step.clone();
        resumed.restart = RestartPolicy::ResumeCheckpoint {
            checkpoint_id: "foreign".into(),
            required_state_digest: "foreign".into(),
        };
        let mut reduced = step.clone();
        reduced.solution_space = SolutionSpace::ModalReduced;
        for steps in [
            vec![transient],
            vec![resumed],
            vec![reduced],
            vec![step.clone(), step],
        ] {
            assert!(
                super::select_family(&AnalysisSequence::from_steps("flow", steps), None).is_err()
            );
        }
    }

    fn select_family(sequence: &AnalysisSequence) -> Result<AdmittedFamily, AdmissionIssue> {
        super::select_family(
            sequence,
            Some(&ElectromagneticDomain::Static {
                equation: runmat_analysis_core::StaticElectricalEquation::Electrostatics,
            }),
        )
    }

    #[test]
    fn maxwell_dispatch_requires_exact_singleton_fresh_harmonic_intent() {
        let domain = ElectromagneticDomain::Harmonic {
            reference_frequency: runmat_analysis_core::Frequency::new(2000.0),
        };
        let mut step = ProcedureStep::electromagnetic_harmonic("field");
        step.independent_domain =
            runmat_analysis_core::IndependentDomain::FrequencyHertz(vec![2000.0]);
        let sequence = AnalysisSequence::from_steps("field", vec![step.clone()]);
        assert_eq!(
            super::select_family(&sequence, Some(&domain)).unwrap(),
            AdmittedFamily::Maxwell
        );
        // The analysis's own frequencies define a harmonic study; no domain block is needed.
        assert_eq!(
            super::select_family(&sequence, None).unwrap(),
            AdmittedFamily::Maxwell
        );
        assert!(super::select_family(
            &sequence,
            Some(&ElectromagneticDomain::Magnetostatic {
                formulation:
                    runmat_analysis_core::MagnetostaticFormulation::MagneticVectorPotential,
            })
        )
        .is_err());
        assert_eq!(
            {
                let mut fanned = step.clone();
                fanned.independent_domain =
                    runmat_analysis_core::IndependentDomain::FrequencyHertz(vec![2000.0, 3000.0]);
                super::select_family(
                    &AnalysisSequence::from_steps("field", vec![fanned]),
                    Some(&domain),
                )
                .unwrap()
            },
            AdmittedFamily::Maxwell
        );
        for frequencies in [vec![], vec![2000.0, f64::NAN], vec![0.0]] {
            let mut changed = step.clone();
            changed.independent_domain =
                runmat_analysis_core::IndependentDomain::FrequencyHertz(frequencies);
            assert!(super::select_family(
                &AnalysisSequence::from_steps("field", vec![changed]),
                Some(&domain)
            )
            .is_err());
        }
        let mut resumed = step.clone();
        resumed.restart = RestartPolicy::ResumeCheckpoint {
            checkpoint_id: "missing".into(),
            required_state_digest: "missing".into(),
        };
        let mut reduced = step.clone();
        reduced.solution_space = SolutionSpace::ModalReduced;
        let mut static_step = step.clone();
        static_step.procedure = AnalysisProcedure::Static;
        for steps in [
            vec![resumed],
            vec![reduced],
            vec![static_step],
            vec![step.clone(), step],
        ] {
            assert!(super::select_family(
                &AnalysisSequence::from_steps("field", steps),
                Some(&domain)
            )
            .is_err());
        }
    }

    #[test]
    fn static_em_dispatch_requires_exact_authored_equation() {
        let sequence = AnalysisSequence::from_steps(
            "field",
            vec![ProcedureStep::electromagnetic_static("field")],
        );
        assert!(super::select_family(&sequence, None).is_err());
        assert!(super::select_family(
            &sequence,
            Some(&ElectromagneticDomain::Harmonic {
                reference_frequency: runmat_analysis_core::Frequency::new(1.0),
            })
        )
        .is_err());
        assert_eq!(
            super::select_family(
                &sequence,
                Some(&ElectromagneticDomain::Magnetostatic {
                    formulation:
                        runmat_analysis_core::MagnetostaticFormulation::MagneticVectorPotential,
                })
            )
            .unwrap(),
            AdmittedFamily::Magnetic
        );
        for equation in [
            runmat_analysis_core::StaticElectricalEquation::Electrostatics,
            runmat_analysis_core::StaticElectricalEquation::SteadyConduction,
        ] {
            assert_eq!(
                super::select_family(&sequence, Some(&ElectromagneticDomain::Static { equation }))
                    .unwrap(),
                AdmittedFamily::Electrical
            );
        }
    }

    #[test]
    fn static_electrical_family_selection_keeps_restart_and_mixed_sequences_closed() {
        let step = ProcedureStep::electromagnetic_static("potential");
        assert_eq!(
            select_family(&AnalysisSequence::from_steps(
                "potential",
                vec![step.clone()]
            ))
            .unwrap(),
            AdmittedFamily::Electrical
        );
        let mut resumed = step.clone();
        resumed.restart = RestartPolicy::ResumeCheckpoint {
            checkpoint_id: "checkpoint".into(),
            required_state_digest: "digest".into(),
        };
        for steps in [
            vec![resumed],
            vec![step.clone(), step],
            vec![ProcedureStep::electromagnetic_harmonic("field")],
        ] {
            assert!(select_family(&AnalysisSequence::from_steps("unsupported", steps)).is_err());
        }
    }

    #[test]
    fn fresh_acoustic_harmonic_selects_without_admitting_restart_or_cross_physics() {
        let acoustic = ProcedureStep::acoustic_harmonic("sound");
        assert_eq!(
            select_family(&AnalysisSequence::from_steps(
                "sound",
                vec![acoustic.clone()]
            ))
            .unwrap(),
            AdmittedFamily::Acoustic
        );
        let mut resumed = acoustic.clone();
        resumed.restart = RestartPolicy::ResumeCheckpoint {
            checkpoint_id: "checkpoint".into(),
            required_state_digest: "digest".into(),
        };
        let mut modal = acoustic.clone();
        modal.solution_space = SolutionSpace::ModalReduced;
        for steps in [
            vec![resumed],
            vec![modal],
            vec![acoustic, ProcedureStep::thermal_steady("heat")],
        ] {
            assert!(select_family(&AnalysisSequence::from_steps("unsupported", steps)).is_err());
        }
    }

    #[test]
    fn standalone_thermal_and_existing_structural_families_select_without_cad() {
        for step in [
            ProcedureStep::thermal_steady("heat"),
            ProcedureStep::thermal_transient("heat"),
        ] {
            assert_eq!(
                select_family(&AnalysisSequence::from_steps("heat", vec![step])).unwrap(),
                AdmittedFamily::Thermal
            );
        }
        assert_eq!(
            select_family(&AnalysisSequence::from_steps(
                "solid",
                vec![ProcedureStep::structural_static("solid")]
            ))
            .unwrap(),
            AdmittedFamily::Structural
        );
    }

    #[test]
    fn thermal_checkpoint_mixed_and_duplicate_steps_stay_closed() {
        let thermal = ProcedureStep::thermal_steady("heat");
        let mut restart = thermal.clone();
        restart.restart = RestartPolicy::ResumeCheckpoint {
            checkpoint_id: "checkpoint".into(),
            required_state_digest: "digest".into(),
        };
        for steps in [
            vec![restart],
            vec![thermal.clone(), ProcedureStep::structural_static("solid")],
            vec![thermal.clone(), ProcedureStep::thermal_transient("other")],
            vec![
                thermal.clone(),
                ProcedureStep::thermal_transient("other"),
                ProcedureStep::thermomechanical_preload("preload", "heat", "heated"),
            ],
            vec![
                ProcedureStep::structural_preload("preload"),
                ProcedureStep::acoustic_harmonic("sound"),
            ],
        ] {
            assert!(
                select_family(&AnalysisSequence::from_steps("unsupported", steps))
                    .unwrap_err()
                    .code
                    == super::PROCEDURE_GRAPH_CODE
            );
        }
        let mut wrong = thermal;
        wrong.family = PhysicsFamily::Acoustic;
        assert!(select_family(&AnalysisSequence::from_steps("acoustic", vec![wrong])).is_err());
    }

    #[test]
    fn thermal_prefix_and_thermomechanical_graphs_select_the_structural_root() {
        let chained = vec![
            ProcedureStep::thermal_steady("heat"),
            ProcedureStep::thermomechanical_preload("preload", "heat", "heated"),
            ProcedureStep::structural_prestressed_modal("modes", "preload", "geometric_stiffness"),
        ];
        let mut prescribed = ProcedureStep::structural_static("coupled");
        prescribed.family = PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical);
        for steps in [chained, vec![prescribed]] {
            assert_eq!(
                select_family(&AnalysisSequence::from_steps("thermal_stress", steps)).unwrap(),
                AdmittedFamily::Structural
            );
        }
    }

    #[test]
    fn physical_thermal_initial_selection_is_admitted_but_other_state_lifecycles_are_not() {
        use runmat_analysis_core::{
            ActivationLifecycle, InitialStateConflictPolicy, ProcedureStepId, StateId, StateInput,
            StateKind, StateSource, StateUse, StepObjectActivation, StepObjectRef,
        };
        let mut selected = ProcedureStep::thermal_transient("heat");
        selected.state_inputs.push(StateInput {
            input_id: "temperature".into(),
            required_kind: StateKind::Thermal,
            source: StateSource::InitialConditions,
            use_as: StateUse::PhysicalInitialState,
            conflict: InitialStateConflictPolicy::Reject,
        });
        selected.activations.push(StepObjectActivation {
            object: StepObjectRef::InitialCondition("chosen".into()),
            lifecycle: ActivationLifecycle::Activate,
        });
        assert_eq!(
            select_family(&AnalysisSequence::from_steps(
                "heat",
                vec![selected.clone()]
            ))
            .unwrap(),
            AdmittedFamily::Thermal
        );
        let mut variants = Vec::new();
        let mut wrong = selected.clone();
        wrong.state_inputs[0].source = StateSource::Upstream {
            producer_step: ProcedureStepId("other".into()),
            state_id: StateId("state".into()),
        };
        variants.push(wrong);
        for kind in [StateKind::StructuralPrimary, StateKind::AcousticComplex] {
            let mut wrong = selected.clone();
            wrong.state_inputs[0].required_kind = kind;
            variants.push(wrong);
        }
        for use_as in [StateUse::InitialGuess, StateUse::Restart] {
            let mut wrong = selected.clone();
            wrong.state_inputs[0].use_as = use_as;
            variants.push(wrong);
        }
        let mut wrong = selected.clone();
        wrong.state_inputs[0].conflict = InitialStateConflictPolicy::AugmentDisjointQuantities;
        variants.push(wrong);
        for lifecycle in [
            ActivationLifecycle::Continue {
                from_step: ProcedureStepId("other".into()),
            },
            ActivationLifecycle::Deactivate {
                from_step: ProcedureStepId("other".into()),
            },
            ActivationLifecycle::Modify {
                prior_step: ProcedureStepId("other".into()),
                prior_object: StepObjectRef::InitialCondition("prior".into()),
            },
        ] {
            let mut wrong = selected.clone();
            wrong.activations[0].lifecycle = lifecycle;
            variants.push(wrong);
        }
        // Non-initial objects are validated by shared model/sequence planning, not
        // interpreted as thermal state selections by this pre-mesh family gate.
        let mut parameter = selected;
        parameter.activations[0].object = StepObjectRef::MaterialParameter("parameter".into());
        assert_eq!(
            select_family(&AnalysisSequence::from_steps("heat", vec![parameter])).unwrap(),
            AdmittedFamily::Thermal
        );
        for step in variants {
            assert!(select_family(&AnalysisSequence::from_steps("heat", vec![step])).is_err());
        }
    }
}
