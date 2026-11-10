use runmat_analysis_core::{
    AnalysisProcedure, CoupledPhysicsFamily, PhysicsFamily, ProcedureStep, SolutionSpace,
};
use runmat_analysis_fea::AnalysisRunIdentity;

use super::contracts::AnalysisExecutionStep;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum AnalysisOperation {
    LinearStatic,
    StructuralFrequency,
    StructuralResponseSpectrum,
    StructuralRandomResponse,
    Modal,
    Buckling,
    Acoustic,
    Transient,
    Thermal,
    Nonlinear,
    Electromagnetic,
    Cfd,
    Cht,
    Fsi,
    LineParameters,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AnalysisOperationDescriptor {
    name: &'static str,
    version: &'static str,
    invalid_model_error_code: &'static str,
    cancelled_error_code: &'static str,
    assembly_error_code: &'static str,
    formulation_admission_error_code: &'static str,
}

impl AnalysisOperation {
    #[cfg(test)]
    const ALL: [Self; 15] = [
        Self::LinearStatic,
        Self::StructuralFrequency,
        Self::StructuralResponseSpectrum,
        Self::StructuralRandomResponse,
        Self::Modal,
        Self::Buckling,
        Self::Acoustic,
        Self::Transient,
        Self::Thermal,
        Self::Nonlinear,
        Self::Electromagnetic,
        Self::Cfd,
        Self::Cht,
        Self::Fsi,
        Self::LineParameters,
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
            },
            Self::StructuralFrequency => AnalysisOperationDescriptor {
                name: "fea.run_structural_frequency",
                version: "fea.run_structural_frequency/v1",
                invalid_model_error_code: "RM.FEA.RUN_STRUCTURAL_FREQUENCY.SOLVER_MODEL_INVALID",
                cancelled_error_code: "RM.FEA.RUN_STRUCTURAL_FREQUENCY.CANCELLED",
                assembly_error_code: "RM.FEA.RUN_STRUCTURAL_FREQUENCY.ASSEMBLY_FAILED",
                formulation_admission_error_code:
                    "RM.FEA.RUN_STRUCTURAL_FREQUENCY.FORMULATION_NOT_ADMITTED",
            },
            Self::StructuralResponseSpectrum => AnalysisOperationDescriptor {
                name: "fea.run_structural_response_spectrum",
                version: "fea.run_structural_response_spectrum/v1",
                invalid_model_error_code:
                    "RM.FEA.RUN_STRUCTURAL_RESPONSE_SPECTRUM.SOLVER_MODEL_INVALID",
                cancelled_error_code: "RM.FEA.RUN_STRUCTURAL_RESPONSE_SPECTRUM.CANCELLED",
                assembly_error_code: "RM.FEA.RUN_STRUCTURAL_RESPONSE_SPECTRUM.ASSEMBLY_FAILED",
                formulation_admission_error_code:
                    "RM.FEA.RUN_STRUCTURAL_RESPONSE_SPECTRUM.FORMULATION_NOT_ADMITTED",
            },
            Self::StructuralRandomResponse => AnalysisOperationDescriptor {
                name: "fea.run_structural_random_response",
                version: "fea.run_structural_random_response/v1",
                invalid_model_error_code:
                    "RM.FEA.RUN_STRUCTURAL_RANDOM_RESPONSE.SOLVER_MODEL_INVALID",
                cancelled_error_code: "RM.FEA.RUN_STRUCTURAL_RANDOM_RESPONSE.CANCELLED",
                assembly_error_code: "RM.FEA.RUN_STRUCTURAL_RANDOM_RESPONSE.ASSEMBLY_FAILED",
                formulation_admission_error_code:
                    "RM.FEA.RUN_STRUCTURAL_RANDOM_RESPONSE.FORMULATION_NOT_ADMITTED",
            },
            Self::Modal => AnalysisOperationDescriptor {
                name: "fea.run_modal",
                version: "fea.run_modal/v1",
                invalid_model_error_code: "RM.FEA.RUN_MODAL.SOLVER_MODEL_INVALID",
                cancelled_error_code: "RM.FEA.RUN_MODAL.CANCELLED",
                assembly_error_code: "RM.FEA.RUN_MODAL.ASSEMBLY_FAILED",
                formulation_admission_error_code: "RM.FEA.RUN_MODAL.FORMULATION_NOT_ADMITTED",
            },
            Self::Buckling => AnalysisOperationDescriptor {
                name: "fea.run_buckling",
                version: "fea.run_buckling/v1",
                invalid_model_error_code: "RM.FEA.RUN_BUCKLING.SOLVER_MODEL_INVALID",
                cancelled_error_code: "RM.FEA.RUN_BUCKLING.CANCELLED",
                assembly_error_code: "RM.FEA.RUN_BUCKLING.ASSEMBLY_FAILED",
                formulation_admission_error_code: "RM.FEA.RUN_BUCKLING.FORMULATION_NOT_ADMITTED",
            },
            Self::Acoustic => AnalysisOperationDescriptor {
                name: "fea.run_acoustic",
                version: "fea.run_acoustic/v1",
                invalid_model_error_code: "RM.FEA.RUN_ACOUSTIC.SOLVER_MODEL_INVALID",
                cancelled_error_code: "RM.FEA.RUN_ACOUSTIC.CANCELLED",
                assembly_error_code: "RM.FEA.RUN_ACOUSTIC.ASSEMBLY_FAILED",
                formulation_admission_error_code: "RM.FEA.RUN_ACOUSTIC.FORMULATION_NOT_ADMITTED",
            },
            Self::Transient => AnalysisOperationDescriptor {
                name: "fea.run_transient",
                version: "fea.run_transient/v1",
                invalid_model_error_code: "RM.FEA.RUN_TRANSIENT.SOLVER_MODEL_INVALID",
                cancelled_error_code: "RM.FEA.RUN_TRANSIENT.CANCELLED",
                assembly_error_code: "RM.FEA.RUN_TRANSIENT.ASSEMBLY_FAILED",
                formulation_admission_error_code: "RM.FEA.RUN_TRANSIENT.FORMULATION_NOT_ADMITTED",
            },
            Self::Thermal => AnalysisOperationDescriptor {
                name: "fea.run_thermal",
                version: "fea.run_thermal/v1",
                invalid_model_error_code: "RM.FEA.RUN_THERMAL.SOLVER_MODEL_INVALID",
                cancelled_error_code: "RM.FEA.RUN_THERMAL.CANCELLED",
                assembly_error_code: "RM.FEA.RUN_THERMAL.ASSEMBLY_FAILED",
                formulation_admission_error_code: "RM.FEA.RUN_THERMAL.FORMULATION_NOT_ADMITTED",
            },
            Self::Nonlinear => AnalysisOperationDescriptor {
                name: "fea.run_nonlinear",
                version: "fea.run_nonlinear/v1",
                invalid_model_error_code: "RM.FEA.RUN_NONLINEAR.SOLVER_MODEL_INVALID",
                cancelled_error_code: "RM.FEA.RUN_NONLINEAR.CANCELLED",
                assembly_error_code: "RM.FEA.RUN_NONLINEAR.ASSEMBLY_FAILED",
                formulation_admission_error_code: "RM.FEA.RUN_NONLINEAR.FORMULATION_NOT_ADMITTED",
            },
            Self::Electromagnetic => AnalysisOperationDescriptor {
                name: "fea.run_electromagnetic",
                version: "fea.run_electromagnetic/v1",
                invalid_model_error_code: "RM.FEA.RUN_ELECTROMAGNETIC.SOLVER_MODEL_INVALID",
                cancelled_error_code: "RM.FEA.RUN_ELECTROMAGNETIC.CANCELLED",
                assembly_error_code: "RM.FEA.RUN_ELECTROMAGNETIC.ASSEMBLY_FAILED",
                formulation_admission_error_code:
                    "RM.FEA.RUN_ELECTROMAGNETIC.FORMULATION_NOT_ADMITTED",
            },
            Self::Cfd => AnalysisOperationDescriptor {
                name: "fea.run_cfd",
                version: "fea.run_cfd/v1",
                invalid_model_error_code: "RM.FEA.RUN_CFD.SOLVER_MODEL_INVALID",
                cancelled_error_code: "RM.FEA.RUN_CFD.CANCELLED",
                assembly_error_code: "RM.FEA.RUN_CFD.ASSEMBLY_FAILED",
                formulation_admission_error_code: "RM.FEA.RUN_CFD.FORMULATION_NOT_ADMITTED",
            },
            Self::Cht => AnalysisOperationDescriptor {
                name: "fea.run_cht",
                version: "fea.run_cht/v1",
                invalid_model_error_code: "RM.FEA.RUN_CHT.SOLVER_MODEL_INVALID",
                cancelled_error_code: "RM.FEA.RUN_CHT.CANCELLED",
                assembly_error_code: "RM.FEA.RUN_CHT.ASSEMBLY_FAILED",
                formulation_admission_error_code: "RM.FEA.RUN_CHT.FORMULATION_NOT_ADMITTED",
            },
            Self::Fsi => AnalysisOperationDescriptor {
                name: "fea.run_fsi",
                version: "fea.run_fsi/v1",
                invalid_model_error_code: "RM.FEA.RUN_FSI.SOLVER_MODEL_INVALID",
                cancelled_error_code: "RM.FEA.RUN_FSI.CANCELLED",
                assembly_error_code: "RM.FEA.RUN_FSI.ASSEMBLY_FAILED",
                formulation_admission_error_code: "RM.FEA.RUN_FSI.FORMULATION_NOT_ADMITTED",
            },
            Self::LineParameters => AnalysisOperationDescriptor {
                name: "fea.run_line_parameters",
                version: "fea.run_line_parameters/v1",
                invalid_model_error_code: "RM.FEA.RUN_LINE_PARAMETERS.SOLVER_MODEL_INVALID",
                cancelled_error_code: "RM.FEA.RUN_LINE_PARAMETERS.CANCELLED",
                assembly_error_code: "RM.FEA.RUN_LINE_PARAMETERS.ASSEMBLY_FAILED",
                formulation_admission_error_code:
                    "RM.FEA.RUN_LINE_PARAMETERS.FORMULATION_NOT_ADMITTED",
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
    use AnalysisProcedure::{
        EigenvalueBuckling, HarmonicResponse, ModalExtraction, ModalResponseSpectrum, Nonlinear,
        PrestressedHarmonicResponse, PrestressedModalExtraction, RandomResponse, Static,
        TransientResponse,
    };
    use PhysicsFamily::{Acoustic, Coupled, Electromagnetic, Fluid, Structural, Thermal};

    match (family, procedure, solution_space) {
        (Structural, Static, SolutionSpace::Direct)
        | (Coupled(CoupledPhysicsFamily::Thermomechanical), Static, SolutionSpace::Direct) => {
            Some(AnalysisOperation::LinearStatic)
        }
        (Structural, ModalExtraction | PrestressedModalExtraction, SolutionSpace::Direct) => {
            Some(AnalysisOperation::Modal)
        }
        (Structural, ModalResponseSpectrum, SolutionSpace::ModalReduced) => {
            Some(AnalysisOperation::StructuralResponseSpectrum)
        }
        (Structural, RandomResponse, SolutionSpace::ModalReduced) => {
            Some(AnalysisOperation::StructuralRandomResponse)
        }
        (Structural, EigenvalueBuckling, SolutionSpace::Direct) => {
            Some(AnalysisOperation::Buckling)
        }
        (
            Structural,
            HarmonicResponse | PrestressedHarmonicResponse,
            SolutionSpace::Direct | SolutionSpace::ModalReduced,
        ) => Some(AnalysisOperation::StructuralFrequency),
        (
            Acoustic,
            HarmonicResponse | ModalExtraction | TransientResponse,
            SolutionSpace::Direct,
        ) => Some(AnalysisOperation::Acoustic),
        (Thermal, Static | TransientResponse, SolutionSpace::Direct) => {
            Some(AnalysisOperation::Thermal)
        }
        (Structural, TransientResponse, SolutionSpace::Direct | SolutionSpace::ModalReduced) => {
            Some(AnalysisOperation::Transient)
        }
        (Fluid, Static | TransientResponse, SolutionSpace::Direct) => Some(AnalysisOperation::Cfd),
        (
            Coupled(CoupledPhysicsFamily::ConjugateHeatTransfer),
            Static | TransientResponse,
            SolutionSpace::Direct,
        ) => Some(AnalysisOperation::Cht),
        (
            Coupled(CoupledPhysicsFamily::FluidStructureInteraction),
            Static | TransientResponse,
            SolutionSpace::Direct,
        ) => Some(AnalysisOperation::Fsi),
        (Structural, Nonlinear, SolutionSpace::Direct)
        | (Coupled(CoupledPhysicsFamily::Thermomechanical), Nonlinear, SolutionSpace::Direct) => {
            Some(AnalysisOperation::Nonlinear)
        }
        (Electromagnetic, Static | HarmonicResponse, SolutionSpace::Direct) => {
            Some(AnalysisOperation::Electromagnetic)
        }
        (Electromagnetic, AnalysisProcedure::LineParameters, SolutionSpace::Direct) => {
            Some(AnalysisOperation::LineParameters)
        }
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
            assert!(invalid_model_codes.insert(operation.invalid_model_error_code()));
            assert!(cancelled_codes.insert(operation.cancelled_error_code()));
            assert!(assembly_codes.insert(operation.assembly_error_code()));
            assert!(admission_codes.insert(operation.formulation_admission_error_code()));
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
                PhysicsFamily::Structural,
                AnalysisProcedure::PrestressedModalExtraction,
                AnalysisOperation::Modal,
            ),
            (
                PhysicsFamily::Structural,
                AnalysisProcedure::HarmonicResponse,
                AnalysisOperation::StructuralFrequency,
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
                PhysicsFamily::Electromagnetic,
                AnalysisProcedure::HarmonicResponse,
                AnalysisOperation::Electromagnetic,
            ),
            (
                PhysicsFamily::Electromagnetic,
                AnalysisProcedure::Static,
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
        assert_eq!(
            run_operation_for_procedure(
                PhysicsFamily::Structural,
                AnalysisProcedure::HarmonicResponse,
                SolutionSpace::ModalReduced,
            ),
            Some(AnalysisOperation::StructuralFrequency)
        );
    }

    #[test]
    fn unsupported_procedure_tuples_have_no_operation_identity() {
        assert_eq!(
            run_operation_for_procedure(
                PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical),
                AnalysisProcedure::TransientResponse,
                SolutionSpace::Direct,
            ),
            None
        );
        assert_eq!(
            run_operation_for_procedure(
                PhysicsFamily::Electromagnetic,
                AnalysisProcedure::ModalExtraction,
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
