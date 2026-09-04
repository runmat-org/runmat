mod identity;
mod realizations;
mod registry;
mod structural;

use std::fmt;

use runmat_analysis_core::{
    AnalysisProcedure, ExactFormulationRealization, FormulationAdmissionRejectionReason,
    FormulationMaturity, FormulationRequirement, PhysicsFamily, SolutionSpace,
};
use serde::{Deserialize, Serialize};

pub use identity::AnalysisRunIdentity;
pub use realizations::{registered_formulations, RegisteredFormulation};
pub use structural::{
    structural_exact_realization, thermomechanical_exact_realization, StructuralRealizationInput,
    ThermomechanicalRealizationInput,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FormulationAdmissionRequest {
    pub family: PhysicsFamily,
    pub procedure: AnalysisProcedure,
    pub solution_space: SolutionSpace,
    pub requirement: FormulationRequirement,
    pub realization: ExactFormulationRealization,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AdmittedFormulationReceipt {
    family: PhysicsFamily,
    procedure: AnalysisProcedure,
    solution_space: SolutionSpace,
    realization: ExactFormulationRealization,
    qualification: FormulationMaturity,
}

impl AdmittedFormulationReceipt {
    pub const fn family(&self) -> PhysicsFamily {
        self.family
    }

    pub const fn procedure(&self) -> AnalysisProcedure {
        self.procedure
    }

    pub const fn solution_space(&self) -> SolutionSpace {
        self.solution_space
    }

    pub fn realization(&self) -> &ExactFormulationRealization {
        &self.realization
    }

    pub const fn qualification(&self) -> FormulationMaturity {
        self.qualification
    }

    pub fn into_identity(self) -> AnalysisRunIdentity {
        AnalysisRunIdentity::from_receipt(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormulationAdmissionRejection {
    pub request: FormulationAdmissionRequest,
    pub reason: FormulationAdmissionRejectionReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FormulationAvailabilityRejection {
    pub family: PhysicsFamily,
    pub procedure: AnalysisProcedure,
    pub solution_space: SolutionSpace,
    pub requirement: FormulationRequirement,
    pub reason: FormulationAdmissionRejectionReason,
}

impl fmt::Display for FormulationAvailabilityRejection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "formulation availability rejected: family={:?} procedure={:?} solution_space={:?} requirement={:?} reason={:?}",
            self.family, self.procedure, self.solution_space, self.requirement, self.reason
        )
    }
}

impl std::error::Error for FormulationAvailabilityRejection {}

pub fn require_qualified_formulation(
    family: PhysicsFamily,
    procedure: AnalysisProcedure,
    solution_space: SolutionSpace,
    requirement: FormulationRequirement,
) -> Result<(), FormulationAvailabilityRejection> {
    registry::require_available(family, procedure, solution_space, requirement).map_err(|reason| {
        FormulationAvailabilityRejection {
            family,
            procedure,
            solution_space,
            requirement,
            reason,
        }
    })
}

impl fmt::Display for FormulationAdmissionRejection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "formulation admission rejected: family={:?} procedure={:?} solution_space={:?} reason={:?}",
            self.request.family, self.request.procedure, self.request.solution_space, self.reason
        )
    }
}

impl std::error::Error for FormulationAdmissionRejection {}

pub fn admit_formulation(
    request: FormulationAdmissionRequest,
) -> Result<AdmittedFormulationReceipt, FormulationAdmissionRejection> {
    let qualification = match registry::qualification(&request) {
        Ok(qualification) => qualification,
        Err(reason) => return Err(FormulationAdmissionRejection { request, reason }),
    };
    if request.requirement == FormulationRequirement::Production
        && qualification != FormulationMaturity::Production
    {
        return Err(FormulationAdmissionRejection {
            request,
            reason: FormulationAdmissionRejectionReason::ProductionQualificationMissing,
        });
    }
    Ok(AdmittedFormulationReceipt {
        family: request.family,
        procedure: request.procedure,
        solution_space: request.solution_space,
        realization: request.realization,
        qualification,
    })
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
