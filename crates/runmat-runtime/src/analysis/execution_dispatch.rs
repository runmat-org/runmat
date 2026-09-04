use runmat_analysis_core::{
    AnalysisProcedure, CoupledPhysicsFamily, PhysicsFamily, ProcedureStep, SolutionSpace,
};
use runmat_analysis_fea::AnalysisRunIdentity;

use super::contracts::AnalysisExecutionStep;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum AnalysisOperation {
    LinearStatic,
    Modal,
    Acoustic,
    Transient,
    Thermal,
    Nonlinear,
    Electromagnetic,
    Cfd,
    Cht,
    Fsi,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AnalysisOperationDescriptor {
    name: &'static str,
    version: &'static str,
    invalid_model_error_code: &'static str,
    cancelled_error_code: &'static str,
    assembly_error_code: &'static str,
    formulation_admission_error_code: &'static str,
    invalid_electro_thermal_options_error_code: &'static str,
}

impl AnalysisOperation {
    #[cfg(test)]
    const ALL: [Self; 10] = [
        Self::LinearStatic,
        Self::Modal,
        Self::Acoustic,
        Self::Transient,
        Self::Thermal,
        Self::Nonlinear,
        Self::Electromagnetic,
        Self::Cfd,
        Self::Cht,
        Self::Fsi,
    ];

    const fn descriptor(self) -> AnalysisOperationDescriptor {
        match self {
            Self::LinearStatic => AnalysisOperationDescriptor {
                name: "fea.run_linear_static",
                version: "fea.run_linear_static/v1",
                invalid_model_error_code: "RM.FEA.RUN_LINEAR_STATIC.SOLVER_MODEL_INVALID",
                cancelled_error_code: "RM.FEA.RUN_LINEAR_STATIC.CANCELLED",
                assembly_error_code: "RM.FEA.RUN_LINEAR_STATIC.ASSEMBLY_FAILED",
                formulation_admission_error_code:
                    "RM.FEA.RUN_LINEAR_STATIC.FORMULATION_NOT_ADMITTED",
                invalid_electro_thermal_options_error_code:
                    "RM.FEA.RUN_LINEAR_STATIC.INVALID_ELECTRO_THERMAL_OPTIONS",
            },
            Self::Modal => AnalysisOperationDescriptor {
                name: "fea.run_modal",
                version: "fea.run_modal/v1",
                invalid_model_error_code: "RM.FEA.RUN_MODAL.SOLVER_MODEL_INVALID",
                cancelled_error_code: "RM.FEA.RUN_MODAL.CANCELLED",
                assembly_error_code: "RM.FEA.RUN_MODAL.ASSEMBLY_FAILED",
                formulation_admission_error_code: "RM.FEA.RUN_MODAL.FORMULATION_NOT_ADMITTED",
                invalid_electro_thermal_options_error_code:
                    "RM.FEA.RUN_MODAL.INVALID_ELECTRO_THERMAL_OPTIONS",
            },
            Self::Acoustic => AnalysisOperationDescriptor {
                name: "fea.run_acoustic",
                version: "fea.run_acoustic/v1",
                invalid_model_error_code: "RM.FEA.RUN_ACOUSTIC.SOLVER_MODEL_INVALID",
                cancelled_error_code: "RM.FEA.RUN_ACOUSTIC.CANCELLED",
                assembly_error_code: "RM.FEA.RUN_ACOUSTIC.ASSEMBLY_FAILED",
                formulation_admission_error_code: "RM.FEA.RUN_ACOUSTIC.FORMULATION_NOT_ADMITTED",
                invalid_electro_thermal_options_error_code:
                    "RM.FEA.RUN_ACOUSTIC.INVALID_ELECTRO_THERMAL_OPTIONS",
            },
            Self::Transient => AnalysisOperationDescriptor {
                name: "fea.run_transient",
                version: "fea.run_transient/v1",
                invalid_model_error_code: "RM.FEA.RUN_TRANSIENT.SOLVER_MODEL_INVALID",
                cancelled_error_code: "RM.FEA.RUN_TRANSIENT.CANCELLED",
                assembly_error_code: "RM.FEA.RUN_TRANSIENT.ASSEMBLY_FAILED",
                formulation_admission_error_code: "RM.FEA.RUN_TRANSIENT.FORMULATION_NOT_ADMITTED",
                invalid_electro_thermal_options_error_code:
                    "RM.FEA.RUN_TRANSIENT.INVALID_ELECTRO_THERMAL_OPTIONS",
            },
            Self::Thermal => AnalysisOperationDescriptor {
                name: "fea.run_thermal",
                version: "fea.run_thermal/v1",
                invalid_model_error_code: "RM.FEA.RUN_THERMAL.SOLVER_MODEL_INVALID",
                cancelled_error_code: "RM.FEA.RUN_THERMAL.CANCELLED",
                assembly_error_code: "RM.FEA.RUN_THERMAL.ASSEMBLY_FAILED",
                formulation_admission_error_code: "RM.FEA.RUN_THERMAL.FORMULATION_NOT_ADMITTED",
                invalid_electro_thermal_options_error_code:
                    "RM.FEA.RUN_THERMAL.INVALID_ELECTRO_THERMAL_OPTIONS",
            },
            Self::Nonlinear => AnalysisOperationDescriptor {
                name: "fea.run_nonlinear",
                version: "fea.run_nonlinear/v1",
                invalid_model_error_code: "RM.FEA.RUN_NONLINEAR.SOLVER_MODEL_INVALID",
                cancelled_error_code: "RM.FEA.RUN_NONLINEAR.CANCELLED",
                assembly_error_code: "RM.FEA.RUN_NONLINEAR.ASSEMBLY_FAILED",
                formulation_admission_error_code: "RM.FEA.RUN_NONLINEAR.FORMULATION_NOT_ADMITTED",
                invalid_electro_thermal_options_error_code:
                    "RM.FEA.RUN_NONLINEAR.INVALID_ELECTRO_THERMAL_OPTIONS",
            },
            Self::Electromagnetic => AnalysisOperationDescriptor {
                name: "fea.run_electromagnetic",
                version: "fea.run_electromagnetic/v1",
                invalid_model_error_code: "RM.FEA.RUN_ELECTROMAGNETIC.SOLVER_MODEL_INVALID",
                cancelled_error_code: "RM.FEA.RUN_ELECTROMAGNETIC.CANCELLED",
                assembly_error_code: "RM.FEA.RUN_ELECTROMAGNETIC.ASSEMBLY_FAILED",
                formulation_admission_error_code:
                    "RM.FEA.RUN_ELECTROMAGNETIC.FORMULATION_NOT_ADMITTED",
                invalid_electro_thermal_options_error_code:
                    "RM.FEA.RUN_ELECTROMAGNETIC.INVALID_ELECTRO_THERMAL_OPTIONS",
            },
            Self::Cfd => AnalysisOperationDescriptor {
                name: "fea.run_cfd",
                version: "fea.run_cfd/v1",
                invalid_model_error_code: "RM.FEA.RUN_CFD.SOLVER_MODEL_INVALID",
                cancelled_error_code: "RM.FEA.RUN_CFD.CANCELLED",
                assembly_error_code: "RM.FEA.RUN_CFD.ASSEMBLY_FAILED",
                formulation_admission_error_code: "RM.FEA.RUN_CFD.FORMULATION_NOT_ADMITTED",
                invalid_electro_thermal_options_error_code:
                    "RM.FEA.RUN_CFD.INVALID_ELECTRO_THERMAL_OPTIONS",
            },
            Self::Cht => AnalysisOperationDescriptor {
                name: "fea.run_cht",
                version: "fea.run_cht/v1",
                invalid_model_error_code: "RM.FEA.RUN_CHT.SOLVER_MODEL_INVALID",
                cancelled_error_code: "RM.FEA.RUN_CHT.CANCELLED",
                assembly_error_code: "RM.FEA.RUN_CHT.ASSEMBLY_FAILED",
                formulation_admission_error_code: "RM.FEA.RUN_CHT.FORMULATION_NOT_ADMITTED",
                invalid_electro_thermal_options_error_code:
                    "RM.FEA.RUN_CHT.INVALID_ELECTRO_THERMAL_OPTIONS",
            },
            Self::Fsi => AnalysisOperationDescriptor {
                name: "fea.run_fsi",
                version: "fea.run_fsi/v1",
                invalid_model_error_code: "RM.FEA.RUN_FSI.SOLVER_MODEL_INVALID",
                cancelled_error_code: "RM.FEA.RUN_FSI.CANCELLED",
                assembly_error_code: "RM.FEA.RUN_FSI.ASSEMBLY_FAILED",
                formulation_admission_error_code: "RM.FEA.RUN_FSI.FORMULATION_NOT_ADMITTED",
                invalid_electro_thermal_options_error_code:
                    "RM.FEA.RUN_FSI.INVALID_ELECTRO_THERMAL_OPTIONS",
            },
        }
    }

    pub(super) const fn name(self) -> &'static str {
        self.descriptor().name
    }

    pub(super) const fn version(self) -> &'static str {
        self.descriptor().version
    }

    pub(super) const fn invalid_model_error_code(self) -> &'static str {
        self.descriptor().invalid_model_error_code
    }

    pub(super) const fn cancelled_error_code(self) -> &'static str {
        self.descriptor().cancelled_error_code
    }

    pub(super) const fn assembly_error_code(self) -> &'static str {
        self.descriptor().assembly_error_code
    }

    pub(super) const fn formulation_admission_error_code(self) -> &'static str {
        self.descriptor().formulation_admission_error_code
    }

    pub(super) const fn invalid_electro_thermal_options_error_code(self) -> &'static str {
        self.descriptor().invalid_electro_thermal_options_error_code
    }
}

pub(super) fn execution_operation(
    execution: &AnalysisExecutionStep,
    procedure_step: &ProcedureStep,
) -> Option<AnalysisOperation> {
    if !execution.solver_options.supports(procedure_step) {
        return None;
    }
    run_operation_for_procedure(
        procedure_step.family,
        procedure_step.procedure,
        procedure_step.solution_space,
    )
}

pub(super) fn run_operation_for_identity(
    identity: &AnalysisRunIdentity,
) -> Option<AnalysisOperation> {
    run_operation_for_procedure(
        identity.family(),
        identity.procedure(),
        identity.solution_space(),
    )
}

fn run_operation_for_procedure(
    family: PhysicsFamily,
    procedure: AnalysisProcedure,
    solution_space: SolutionSpace,
) -> Option<AnalysisOperation> {
    if solution_space != SolutionSpace::Direct {
        return None;
    }
    use AnalysisProcedure::{
        HarmonicResponse, ModalExtraction, Nonlinear, Static, TransientResponse,
    };
    use PhysicsFamily::{Acoustic, Coupled, Electromagnetic, Fluid, Structural, Thermal};

    match (family, procedure) {
        (Structural, Static) | (Coupled(CoupledPhysicsFamily::Thermomechanical), Static) => {
            Some(AnalysisOperation::LinearStatic)
        }
        (Structural, ModalExtraction) => Some(AnalysisOperation::Modal),
        (Acoustic, HarmonicResponse) => Some(AnalysisOperation::Acoustic),
        (Thermal, TransientResponse) => Some(AnalysisOperation::Thermal),
        (Structural, TransientResponse)
        | (Coupled(CoupledPhysicsFamily::Thermomechanical), TransientResponse) => {
            Some(AnalysisOperation::Transient)
        }
        (Fluid, Static | TransientResponse) => Some(AnalysisOperation::Cfd),
        (Coupled(CoupledPhysicsFamily::ConjugateHeatTransfer), Static | TransientResponse) => {
            Some(AnalysisOperation::Cht)
        }
        (Coupled(CoupledPhysicsFamily::FluidStructureInteraction), TransientResponse) => {
            Some(AnalysisOperation::Fsi)
        }
        (Structural, Nonlinear) | (Coupled(CoupledPhysicsFamily::Thermomechanical), Nonlinear) => {
            Some(AnalysisOperation::Nonlinear)
        }
        (Electromagnetic, HarmonicResponse) => Some(AnalysisOperation::Electromagnetic),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    #[test]
    fn operation_descriptors_have_complete_unique_boundary_identities_and_error_codes() {
        let mut names = BTreeSet::new();
        let mut versions = BTreeSet::new();
        let mut invalid_model_codes = BTreeSet::new();
        let mut cancelled_codes = BTreeSet::new();
        let mut assembly_codes = BTreeSet::new();
        let mut admission_codes = BTreeSet::new();
        let mut electro_thermal_codes = BTreeSet::new();

        for operation in AnalysisOperation::ALL {
            assert!(names.insert(operation.name()));
            assert!(versions.insert(operation.version()));
            assert_eq!(operation.version(), format!("{}/v1", operation.name()));
            let error_namespace = format!(
                "RM.FEA.{}.",
                operation
                    .name()
                    .strip_prefix("fea.")
                    .expect("analysis operation namespace")
                    .to_ascii_uppercase()
            );
            assert_eq!(
                operation.invalid_model_error_code(),
                format!("{error_namespace}SOLVER_MODEL_INVALID")
            );
            assert_eq!(
                operation.cancelled_error_code(),
                format!("{error_namespace}CANCELLED")
            );
            assert_eq!(
                operation.assembly_error_code(),
                format!("{error_namespace}ASSEMBLY_FAILED")
            );
            assert_eq!(
                operation.formulation_admission_error_code(),
                format!("{error_namespace}FORMULATION_NOT_ADMITTED")
            );
            assert_eq!(
                operation.invalid_electro_thermal_options_error_code(),
                format!("{error_namespace}INVALID_ELECTRO_THERMAL_OPTIONS")
            );
            assert!(invalid_model_codes.insert(operation.invalid_model_error_code()));
            assert!(cancelled_codes.insert(operation.cancelled_error_code()));
            assert!(assembly_codes.insert(operation.assembly_error_code()));
            assert!(admission_codes.insert(operation.formulation_admission_error_code()));
            assert!(electro_thermal_codes
                .insert(operation.invalid_electro_thermal_options_error_code()));
        }
    }

    #[test]
    fn supported_canonical_procedure_tuples_select_typed_operations() {
        let direct = SolutionSpace::Direct;
        let cases = [
            (
                PhysicsFamily::Structural,
                AnalysisProcedure::Static,
                AnalysisOperation::LinearStatic,
            ),
            (
                PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical),
                AnalysisProcedure::Static,
                AnalysisOperation::LinearStatic,
            ),
            (
                PhysicsFamily::Structural,
                AnalysisProcedure::ModalExtraction,
                AnalysisOperation::Modal,
            ),
            (
                PhysicsFamily::Acoustic,
                AnalysisProcedure::HarmonicResponse,
                AnalysisOperation::Acoustic,
            ),
            (
                PhysicsFamily::Thermal,
                AnalysisProcedure::TransientResponse,
                AnalysisOperation::Thermal,
            ),
            (
                PhysicsFamily::Structural,
                AnalysisProcedure::TransientResponse,
                AnalysisOperation::Transient,
            ),
            (
                PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical),
                AnalysisProcedure::TransientResponse,
                AnalysisOperation::Transient,
            ),
            (
                PhysicsFamily::Electromagnetic,
                AnalysisProcedure::HarmonicResponse,
                AnalysisOperation::Electromagnetic,
            ),
            (
                PhysicsFamily::Fluid,
                AnalysisProcedure::Static,
                AnalysisOperation::Cfd,
            ),
            (
                PhysicsFamily::Fluid,
                AnalysisProcedure::TransientResponse,
                AnalysisOperation::Cfd,
            ),
            (
                PhysicsFamily::Coupled(CoupledPhysicsFamily::ConjugateHeatTransfer),
                AnalysisProcedure::Static,
                AnalysisOperation::Cht,
            ),
            (
                PhysicsFamily::Coupled(CoupledPhysicsFamily::ConjugateHeatTransfer),
                AnalysisProcedure::TransientResponse,
                AnalysisOperation::Cht,
            ),
            (
                PhysicsFamily::Coupled(CoupledPhysicsFamily::FluidStructureInteraction),
                AnalysisProcedure::TransientResponse,
                AnalysisOperation::Fsi,
            ),
            (
                PhysicsFamily::Structural,
                AnalysisProcedure::Nonlinear,
                AnalysisOperation::Nonlinear,
            ),
            (
                PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical),
                AnalysisProcedure::Nonlinear,
                AnalysisOperation::Nonlinear,
            ),
        ];
        for (family, procedure, operation) in cases {
            assert_eq!(
                run_operation_for_procedure(family, procedure, direct),
                Some(operation)
            );
        }
    }

    #[test]
    fn unsupported_procedure_tuples_have_no_operation_identity() {
        assert_eq!(
            run_operation_for_procedure(
                PhysicsFamily::Thermal,
                AnalysisProcedure::Static,
                SolutionSpace::Direct,
            ),
            None
        );
        assert_eq!(
            run_operation_for_procedure(
                PhysicsFamily::Electromagnetic,
                AnalysisProcedure::Static,
                SolutionSpace::Direct,
            ),
            None
        );
        assert_eq!(
            run_operation_for_procedure(
                PhysicsFamily::Coupled(CoupledPhysicsFamily::Electrothermal),
                AnalysisProcedure::TransientResponse,
                SolutionSpace::Direct,
            ),
            None
        );
        assert_eq!(
            run_operation_for_procedure(
                PhysicsFamily::Structural,
                AnalysisProcedure::Static,
                SolutionSpace::ModalReduced,
            ),
            None
        );
    }
}
