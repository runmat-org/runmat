//! Observation binding at static admission: every authored field, history, probe and
//! reduction must bind one field its family declares. The rule is
//! [`crate::results::observation::bind_request_field`], the same function plan resolution
//! calls after the solve, run here against each family's mesh-free native inventory so
//! `runmat check` and run preparation reject an unbound observation before meshing.

use runmat_analysis_core::{
    CoupledPhysicsFamily, ElectromagneticDomain, IndependentDomain, PhysicsFamily, ProcedureStep, RealizedStudy,
};

use super::study::{AdmissionIssue, AdmittedFamily, OBSERVATION_CODE};
use crate::execution::FluidCoupling;
use crate::results::field::RecoveredFieldDefinition;

/// The native fields a family declares for `step`, independent of the mesh. `None` leaves the
/// binding to the family's run-time plan (structural, thermal and acoustic catalogs depend
/// on the mesh, connectors or recovery request).
fn declared_fields(
    realized: &RealizedStudy,
    family: AdmittedFamily,
    step: &ProcedureStep,
) -> Option<Vec<RecoveredFieldDefinition>> {
    match family {
        AdmittedFamily::Electrical => match realized.model.electromagnetic.as_ref() {
            Some(ElectromagneticDomain::Static { equation }) => {
                crate::execution::electrical_native_field_definition_inventory(*equation).ok()
            }
            _ => None,
        },
        AdmittedFamily::Magnetic => crate::execution::magnetic_native_field_definition_inventory().ok(),
        AdmittedFamily::Maxwell => crate::execution::maxwell_native_field_definition_inventory().ok(),
        AdmittedFamily::Fluid => {
            let coupling = match step.family {
                PhysicsFamily::Coupled(CoupledPhysicsFamily::ConjugateHeatTransfer) => {
                    FluidCoupling::ConjugateHeat
                }
                PhysicsFamily::Coupled(CoupledPhysicsFamily::FluidStructureInteraction) => {
                    FluidCoupling::FluidStructure
                }
                _ => FluidCoupling::Standalone,
            };
            crate::execution::coupled_fluid_field_definition_inventory(coupling).ok()
        }
        _ => None,
    }
}

pub(super) fn admit_observation_bindings(
    realized: &RealizedStudy,
    family: AdmittedFamily,
    issues: &mut Vec<AdmissionIssue>,
) {
    for step in &realized.sequence.steps {
        if step.observations.is_empty() {
            continue;
        }
        let Some(fields) = declared_fields(realized, family, step) else {
            continue;
        };
        // A multi-frequency Maxwell study runs one prepared case per frequency; bind against
        // the step each case prepares.
        let case;
        let step = match &step.independent_domain {
            IndependentDomain::FrequencyHertz(frequencies)
                if family == AdmittedFamily::Maxwell && frequencies.len() > 1 =>
            {
                let mut single = step.clone();
                single.independent_domain = IndependentDomain::FrequencyHertz(vec![frequencies[0]]);
                case = single;
                &case
            }
            _ => step,
        };
        for request in &step.observations {
            if let Err(error) =
                crate::results::observation::bind_request_field(step, request, fields.iter())
            {
                let issue = AdmissionIssue {
                    code: OBSERVATION_CODE,
                    pending: None,
                    message: format!("analysis `{}`: {error}", step.step_id.as_str()),
                };
                if !issues.contains(&issue) {
                    issues.push(issue);
                }
            }
        }
    }
}
