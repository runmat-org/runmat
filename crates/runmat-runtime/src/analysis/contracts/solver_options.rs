//! Closed solver-option selection; numerical execution remains with each physics owner.
use super::*;

impl AnalysisSolverOptions {
    pub(in crate::analysis) fn solver_mesh_artifact_path(&self) -> Option<&str> {
        match self {
            Self::StructuralLinear(o) => o.solver_mesh_artifact_path.as_deref(),
            Self::StructuralFrequency(o) => o.solver_mesh_artifact_path.as_deref(),
            Self::StructuralModes(o) => o.solver_mesh_artifact_path.as_deref(),
            Self::StructuralResponseSpectrum(o) => o.solver_mesh_artifact_path.as_deref(),
            Self::StructuralRandomResponse(o) => o.solver_mesh_artifact_path.as_deref(),
            Self::StructuralBuckling(o) => o.solver_mesh_artifact_path.as_deref(),
            Self::ThermalDiffusion(o) => o.solver_mesh_artifact_path.as_deref(),
            Self::StructuralTransient(o) => o.solver_mesh_artifact_path.as_deref(),
            Self::StructuralNonlinear(o) => o.solver_mesh_artifact_path.as_deref(),
            _ => None,
        }
    }
    pub(in crate::analysis) fn clear_solver_mesh_artifact_path(&mut self) {
        match self {
            Self::ElectricalStatic(_) | Self::Magnetostatic(_) | Self::MaxwellHarmonic(_) => {}
            Self::StructuralLinear(options) => options.solver_mesh_artifact_path = None,
            Self::StructuralFrequency(options) => options.solver_mesh_artifact_path = None,
            Self::StructuralModes(options) => options.solver_mesh_artifact_path = None,
            Self::StructuralResponseSpectrum(options) => {
                options.solver_mesh_artifact_path = None;
            }
            Self::StructuralRandomResponse(options) => {
                options.solver_mesh_artifact_path = None;
            }
            Self::StructuralBuckling(options) => options.solver_mesh_artifact_path = None,
            Self::AcousticHelmholtz(_) | Self::AcousticResponse(_) => {}
            Self::ThermalDiffusion(options) => options.solver_mesh_artifact_path = None,
            Self::StructuralTransient(options) => options.solver_mesh_artifact_path = None,
            Self::IncompressibleFlow(_) => {}
            Self::StructuralNonlinear(options) => options.solver_mesh_artifact_path = None,
        }
    }

    pub fn defaults_for(step: &ProcedureStep) -> Result<Self, String> {
        use AnalysisProcedure::{
            EigenvalueBuckling, HarmonicResponse, ModalExtraction, ModalResponseSpectrum,
            Nonlinear, PrestressedHarmonicResponse, PrestressedModalExtraction, RandomResponse,
            Static, TransientResponse,
        };
        use PhysicsFamily::{Acoustic, Coupled, Electromagnetic, Fluid, Structural, Thermal};

        match (step.family, step.procedure, step.solution_space) {
            (Electromagnetic, Static, SolutionSpace::Direct) => Err(
                "static electromagnetic execution requires explicit equation-bound solver options".into(),
            ),
            (Structural, Static, SolutionSpace::Direct)
            | (
                Coupled(CoupledPhysicsFamily::Thermomechanical),
                Static,
                SolutionSpace::Direct,
            ) => {
                Ok(Self::StructuralLinear(AnalysisRunOptions::default()))
            }
            (
                Structural,
                ModalExtraction | PrestressedModalExtraction,
                SolutionSpace::Direct,
            ) => {
                let runmat_analysis_core::IndependentDomain::Eigenvalue { requested_count } =
                    &step.independent_domain
                else {
                    return Err(format!(
                        "modal procedure step `{}` requires an eigenvalue domain",
                        step.step_id.as_str()
                    ));
                };
                let options = AnalysisModalRunOptions {
                    truncation: runmat_analysis_fea::ModalTruncationPolicy::requested(
                        *requested_count,
                    ),
                    ..AnalysisModalRunOptions::default()
                };
                Ok(Self::StructuralModes(options))
            }
            (Structural, ModalResponseSpectrum, SolutionSpace::ModalReduced) => Ok(
                Self::StructuralResponseSpectrum(AnalysisResponseSpectrumRunOptions::default()),
            ),
            (Structural, RandomResponse, SolutionSpace::ModalReduced) => Ok(
                Self::StructuralRandomResponse(AnalysisRandomResponseRunOptions::default()),
            ),
            (Structural, EigenvalueBuckling, SolutionSpace::Direct) => Ok(
                Self::StructuralBuckling(AnalysisBucklingRunOptions::default()),
            ),
            // Direct and modal-reduced (modal -> harmonic) sweeps share the frequency options.
            (
                Structural,
                HarmonicResponse | PrestressedHarmonicResponse,
                SolutionSpace::Direct | SolutionSpace::ModalReduced,
            ) => Ok(Self::StructuralFrequency(
                AnalysisStructuralFrequencyRunOptions::default(),
            )),
            (Acoustic, ModalExtraction | TransientResponse, SolutionSpace::Direct) => Ok(
                Self::AcousticResponse(super::super::AnalysisAcousticResponseRunOptions::default()),
            ),
            (Acoustic, HarmonicResponse, SolutionSpace::Direct) => Ok(Self::AcousticHelmholtz(
                AnalysisAcousticRunOptions::default(),
            )),
            (Thermal, Static | TransientResponse, SolutionSpace::Direct) => Ok(Self::ThermalDiffusion(
                AnalysisThermalRunOptions::default(),
            )),
            (
                Structural,
                TransientResponse,
                SolutionSpace::Direct | SolutionSpace::ModalReduced,
            ) => Ok(Self::StructuralTransient(AnalysisTransientRunOptions::default())),
            // Flow studies, standalone or coupled to heat transfer or a structure, run the one
            // canonical finite-volume flow lifecycle and share its solver options: there is no
            // separate conjugate-heat-transfer or fluid-structure solver-options variant.
            (Fluid, Static | TransientResponse, SolutionSpace::Direct)
            | (
                Coupled(
                    CoupledPhysicsFamily::ConjugateHeatTransfer
                    | CoupledPhysicsFamily::FluidStructureInteraction,
                ),
                Static | TransientResponse,
                SolutionSpace::Direct,
            ) => Ok(Self::IncompressibleFlow(AnalysisCfdRunOptions::default())),
            (Structural, Nonlinear, SolutionSpace::Direct)
            | (
                Coupled(CoupledPhysicsFamily::Thermomechanical),
                Nonlinear,
                SolutionSpace::Direct,
            ) => Ok(Self::StructuralNonlinear(
                AnalysisNonlinearRunOptions::default(),
            )),
            (Electromagnetic, HarmonicResponse, SolutionSpace::Direct) => Ok(
                Self::MaxwellHarmonic(super::super::AnalysisMaxwellRunOptions::default()),
            ),
            (family, procedure, solution_space) => Err(format!(
                "procedure step `{}` selects unsupported execution combination {family:?}/{procedure:?}/{solution_space:?}",
                step.step_id.as_str()
            )),
        }
    }

    pub fn supports(&self, step: &ProcedureStep) -> bool {
        if matches!(self, Self::ElectricalStatic(_) | Self::Magnetostatic(_)) {
            return step.family == PhysicsFamily::Electromagnetic
                && step.procedure == AnalysisProcedure::Static
                && step.solution_space == SolutionSpace::Direct
                && step.independent_domain == runmat_analysis_core::IndependentDomain::Static;
        }
        Self::defaults_for(step)
            .map(|expected| std::mem::discriminant(&expected) == std::mem::discriminant(self))
            .unwrap_or(false)
    }
}
