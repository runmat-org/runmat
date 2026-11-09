mod electrical;
mod fluid;
mod fluid_solid;
mod identity;
mod magnetic;
mod maxwell;
mod nonstructural;
mod realizations;
mod registry;
mod structural;
mod study;

use std::fmt;

use runmat_analysis_core::{
    AnalysisProcedure, ExactFormulationRealization, FormulationAdmissionRejectionReason,
    PhysicsFamily, SolutionSpace,
};
use serde::{Deserialize, Serialize};

pub use electrical::admit_canonical_electrical_static;
pub(crate) use electrical::realization as canonical_electrical_realization;
pub use fluid::{
    admit_canonical_conjugate_heat_steady, admit_canonical_conjugate_heat_transient,
    admit_canonical_fluid_steady, admit_canonical_fluid_structure_steady,
    admit_canonical_fluid_structure_transient, admit_canonical_fluid_transient,
    canonical_fluid_steady_realization, canonical_fluid_transient_realization,
};
pub use fluid_solid::{
    classify_fluid_solid_face, is_authored_fluid_wall, is_fluid_wall, FluidSolidFaceRole,
    FLUID_SOLID_FACE_CODE,
};
pub use identity::AnalysisRunIdentity;
pub use magnetic::admit_canonical_magnetostatic;
pub(crate) use magnetic::realization as canonical_magnetic_realization;
pub use maxwell::{admit_canonical_maxwell, canonical_maxwell_realization};
pub use nonstructural::admit_nonstructural;
pub(crate) use nonstructural::admit_nonstructural_with_temporal;
pub use realizations::{registered_formulations, RegisteredFormulation};
pub use structural::{
    structural_exact_realization, thermomechanical_exact_realization, StructuralRealizationInput,
    ThermomechanicalRealizationInput,
};
pub use study::{
    admit_accelerator_policy, admit_study, is_electrothermal, select_family, AdmissionIssue,
    AdmittedFamily, StaticRule, ACCELERATOR_REQUIRED_CODE, ACOUSTIC_ROTATED_PERIODIC,
    ACOUSTIC_TRANSIENT_CONSTRAINT, ACOUSTIC_TRANSIENT_INITIAL_CONDITION,
    ELECTROMAGNETIC_INITIAL_CONDITION, FAMILY_PROCEDURE_CODE, FLUID_INITIAL_CONDITION,
    FLUID_PERIODIC, MAGNETIC_STATIC_CONSTRAINT, MAGNETIC_STATIC_GAUGE, MAXWELL_CONSTRAINT,
    MAXWELL_FAR_FIELD, MAXWELL_NONELECTROMAGNETIC_CONSTRAINT, OBSERVATION_CODE,
    PROCEDURE_GRAPH_CODE,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FormulationAdmissionRequest {
    pub family: PhysicsFamily,
    pub procedure: AnalysisProcedure,
    pub solution_space: SolutionSpace,
    pub realization: ExactFormulationRealization,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AdmittedFormulationReceipt {
    family: PhysicsFamily,
    procedure: AnalysisProcedure,
    solution_space: SolutionSpace,
    realization: ExactFormulationRealization,
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
    pub reason: FormulationAdmissionRejectionReason,
}

impl fmt::Display for FormulationAvailabilityRejection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "formulation availability rejected: family={:?} procedure={:?} solution_space={:?} reason={:?}",
            self.family, self.procedure, self.solution_space, self.reason
        )
    }
}

impl std::error::Error for FormulationAvailabilityRejection {}

pub fn require_available_formulation(
    family: PhysicsFamily,
    procedure: AnalysisProcedure,
    solution_space: SolutionSpace,
) -> Result<(), FormulationAvailabilityRejection> {
    registry::require_available(family, procedure, solution_space).map_err(|reason| {
        FormulationAvailabilityRejection {
            family,
            procedure,
            solution_space,
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
    if let Err(reason) = registry::admit(&request) {
        return Err(FormulationAdmissionRejection { request, reason });
    }
    Ok(AdmittedFormulationReceipt {
        family: request.family,
        procedure: request.procedure,
        solution_space: request.solution_space,
        realization: request.realization,
    })
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
