use std::collections::HashSet;

use crate::{
    AccuracyTarget, AccuracyTolerance, ActivationLifecycle, AnalysisModel, AnalysisValidationError,
    ErrorContributionKind, ProcedureStep, StepObjectRef,
};

use super::{graph::ValidatedGraph, step_error};

pub(super) fn validate_objects(
    model: &AnalysisModel,
    step: &ProcedureStep,
) -> Result<(), AnalysisValidationError> {
    if let crate::IndependentDomain::ResponseSpectra { spectrum_ids } = &step.independent_domain {
        let required_captured_mass_ratio = step
            .accuracy
            .criteria
            .iter()
            .filter(|criterion| {
                criterion.mandatory
                    && criterion.target == AccuracyTarget::NumericalResult
                    && criterion.contribution == ErrorContributionKind::ModalTruncation
            })
            .filter_map(|criterion| match criterion.tolerance {
                AccuracyTolerance::Relative(tolerance) => Some((1.0 - tolerance).max(0.0)),
                AccuracyTolerance::Absolute { .. } => None,
            })
            .fold(0.0_f64, f64::max);
        for spectrum_id in spectrum_ids {
            let Some(spectrum) = model
                .response_spectra
                .iter()
                .find(|spectrum| spectrum.spectrum_id == *spectrum_id)
            else {
                return Err(step_error(
                    step,
                    format!(
                        "response-spectrum procedure references unknown spectrum `{}`",
                        spectrum_id.0
                    ),
                ));
            };
            if spectrum.truncation.minimum_cumulative_mass_ratio + 64.0 * f64::EPSILON
                < required_captured_mass_ratio
            {
                return Err(step_error(
                    step,
                    format!(
                        "response spectrum `{}` stops at captured-mass ratio {} but the accuracy plan requires at least {}",
                        spectrum_id.0,
                        spectrum.truncation.minimum_cumulative_mass_ratio,
                        required_captured_mass_ratio
                    ),
                ));
            }
        }
    }
    if let crate::IndependentDomain::RandomExcitation { excitation_id } = &step.independent_domain {
        if !model
            .random_excitations
            .iter()
            .any(|excitation| excitation.excitation_id == *excitation_id)
        {
            return Err(step_error(
                step,
                format!(
                    "random-response procedure references unknown excitation `{}`",
                    excitation_id.0
                ),
            ));
        }
    }
    let mut seen = HashSet::new();
    for activation in &step.activations {
        if !seen.insert(&activation.object) {
            return Err(step_error(
                step,
                format!("object {:?} is activated more than once", activation.object),
            ));
        }
        if !object_exists(model, &activation.object) {
            return Err(step_error(
                step,
                format!(
                    "activation references unknown object {:?}",
                    activation.object
                ),
            ));
        }
        if let ActivationLifecycle::Modify { prior_object, .. } = &activation.lifecycle {
            if prior_object == &activation.object {
                return Err(step_error(
                    step,
                    "modification must produce a new authored object identity",
                ));
            }
            if !object_exists(model, prior_object) {
                return Err(step_error(
                    step,
                    format!("modification references unknown prior object {prior_object:?}"),
                ));
            }
        }
    }
    Ok(())
}

pub(super) fn validate_lifecycles<'a>(
    step: &'a ProcedureStep,
    graph: &ValidatedGraph<'a>,
) -> Result<(), AnalysisValidationError> {
    for activation in &step.activations {
        let prior = match &activation.lifecycle {
            ActivationLifecycle::Activate => continue,
            ActivationLifecycle::Continue { from_step }
            | ActivationLifecycle::Deactivate { from_step } => {
                (from_step.as_str(), &activation.object)
            }
            ActivationLifecycle::Modify {
                prior_step,
                prior_object,
            } => (prior_step.as_str(), prior_object),
        };
        let Some(prior_step) = graph.steps.get(prior.0) else {
            return Err(step_error(
                step,
                format!("activation lifecycle references unknown step `{}`", prior.0),
            ));
        };
        if !graph.is_ancestor(prior.0, step.step_id.as_str()) {
            return Err(step_error(
                step,
                format!("activation lifecycle step `{}` is not upstream", prior.0),
            ));
        }
        if !prior_step
            .activations
            .iter()
            .any(|value| &value.object == prior.1 && leaves_active(&value.lifecycle))
        {
            return Err(step_error(
                step,
                format!("object {:?} was not active in step `{}`", prior.1, prior.0),
            ));
        }
    }
    Ok(())
}

fn object_exists(model: &AnalysisModel, object: &StepObjectRef) -> bool {
    match object {
        StepObjectRef::Constraint(id) => model
            .constraints
            .iter()
            .any(|value| value.constraint_id() == id),
        StepObjectRef::Boundary(id) => model
            .boundaries
            .iter()
            .any(|value| value.boundary_id == *id),
        StepObjectRef::Connector(id) => model
            .connectors
            .iter()
            .any(|value| value.connector_id == *id),
        StepObjectRef::Interface(id) => model
            .interfaces
            .iter()
            .any(|value| value.interface_id == *id),
        StepObjectRef::Excitation(id) => model
            .excitations
            .iter()
            .any(|value| value.excitation().excitation_id.0 == *id),
        StepObjectRef::InitialCondition(id) => model
            .initial_conditions
            .iter()
            .any(|value| value.initial_condition_id() == id),
        StepObjectRef::MaterialParameter(id) => {
            let Some((material_id, parameter)) = id.split_once('.') else {
                return false;
            };
            !parameter.trim().is_empty()
                && model
                    .materials
                    .iter()
                    .any(|material| material.material_id == material_id)
        }
    }
}

fn leaves_active(lifecycle: &ActivationLifecycle) -> bool {
    !matches!(lifecycle, ActivationLifecycle::Deactivate { .. })
}
