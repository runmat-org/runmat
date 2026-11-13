//! A harmonic electromagnetic analysis whose cycle-averaged loss heats a following thermal
//! analysis (`heat_source = analysis.<harmonic>.state.<name>`), as in induction heating.
//!
//! The authored study is split into its two analyses. The harmonic analysis runs and
//! publishes as its own case; its element loss density (1/V)∫(1/2)sigma|E|^2 then becomes the
//! heat source bound into the thermal analysis, which runs and publishes as the next case
//! on the same mesh. Each case carries only the outputs and assessments of its analysis.
use runmat_analysis_core::{AnalysisProcedure, PhysicsFamily, StateSource, StateUse};
use runmat_analysis_language::ResolvedAnalysisStudy;

pub(super) struct HeatingChain {
    pub harmonic: String,
    pub thermal: String,
    pub state_id: String,
}

pub(super) fn heating_chain(study: &ResolvedAnalysisStudy) -> Option<HeatingChain> {
    let steps = &study.study.sequence.steps;
    let thermal = steps.iter().find(|step| {
        step.family == PhysicsFamily::Thermal
            && step
                .state_inputs
                .iter()
                .any(|input| input.use_as == StateUse::HeatSource)
    })?;
    let input = thermal
        .state_inputs
        .iter()
        .find(|input| input.use_as == StateUse::HeatSource)?;
    let StateSource::Upstream {
        producer_step,
        state_id,
    } = &input.source
    else {
        return None;
    };
    let harmonic = steps.iter().find(|step| &step.step_id == producer_step)?;
    (harmonic.family == PhysicsFamily::Electromagnetic
        && harmonic.procedure == AnalysisProcedure::HarmonicResponse)
        .then(|| HeatingChain {
            harmonic: harmonic.step_id.0.clone(),
            thermal: thermal.step_id.0.clone(),
            state_id: state_id.0.clone(),
        })
}

/// The study restricted to one analysis, with only the outputs/assessments it owns. A thermal
/// analysis keeps its authored definition except the upstream heat-source binding, which the
/// caller replaces with the solved loss field.
pub(super) fn sub_study(study: &ResolvedAnalysisStudy, step_id: &str) -> ResolvedAnalysisStudy {
    let mut part = study.clone();
    let sequence = &mut part.study.sequence;
    sequence
        .steps
        .retain(|step| step.step_id.as_str() == step_id);
    for step in &mut sequence.steps {
        step.state_inputs
            .retain(|input| input.use_as != StateUse::HeatSource);
    }
    sequence.dependencies.clear();
    sequence.transitions.clear();
    sequence.branches.clear();
    let study = &mut part.study;
    runmat_analysis_core::retain_authored_outputs(
        &mut study.outputs,
        &mut study.assessments,
        |owner, _| owner == step_id,
    );
    part
}

/// The analysis name of a single electrothermal (Joule heating) analysis, if that is the study.
pub(super) fn electrothermal_analysis(study: &ResolvedAnalysisStudy) -> Option<String> {
    match study.study.sequence.steps.as_slice() {
        [step]
            if step.family
                == PhysicsFamily::Coupled(
                    runmat_analysis_core::CoupledPhysicsFamily::Electrothermal,
                ) =>
        {
            Some(step.step_id.0.clone())
        }
        _ => None,
    }
}
