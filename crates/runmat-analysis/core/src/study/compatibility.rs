use super::identity::{AnalysisProcedure, CoupledPhysicsFamily, PhysicsFamily, SolutionSpace};

pub fn is_exposed_analysis_identity(
    family: PhysicsFamily,
    procedure: AnalysisProcedure,
    solution_space: SolutionSpace,
) -> bool {
    if solution_space != SolutionSpace::Direct {
        return false;
    }
    matches!(
        (family, procedure),
        (PhysicsFamily::Structural, AnalysisProcedure::Static)
            | (
                PhysicsFamily::Structural,
                AnalysisProcedure::ModalExtraction
            )
            | (
                PhysicsFamily::Structural,
                AnalysisProcedure::TransientResponse
            )
            | (PhysicsFamily::Structural, AnalysisProcedure::Nonlinear)
            | (PhysicsFamily::Thermal, AnalysisProcedure::Static)
            | (PhysicsFamily::Thermal, AnalysisProcedure::TransientResponse)
            | (
                PhysicsFamily::Electromagnetic,
                AnalysisProcedure::HarmonicResponse
            )
            | (PhysicsFamily::Acoustic, AnalysisProcedure::HarmonicResponse)
            | (PhysicsFamily::Fluid, AnalysisProcedure::Static)
            | (PhysicsFamily::Fluid, AnalysisProcedure::TransientResponse)
            | (
                PhysicsFamily::Coupled(CoupledPhysicsFamily::ConjugateHeatTransfer),
                AnalysisProcedure::Static
            )
            | (
                PhysicsFamily::Coupled(CoupledPhysicsFamily::ConjugateHeatTransfer),
                AnalysisProcedure::TransientResponse
            )
            | (
                PhysicsFamily::Coupled(CoupledPhysicsFamily::FluidStructureInteraction),
                AnalysisProcedure::TransientResponse
            )
            | (
                PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical),
                AnalysisProcedure::Static
            )
            | (
                PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical),
                AnalysisProcedure::Nonlinear
            )
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_current_exposed_identity_is_in_the_closed_table() {
        let identities = [
            (PhysicsFamily::Structural, AnalysisProcedure::Static),
            (
                PhysicsFamily::Structural,
                AnalysisProcedure::ModalExtraction,
            ),
            (
                PhysicsFamily::Structural,
                AnalysisProcedure::TransientResponse,
            ),
            (PhysicsFamily::Structural, AnalysisProcedure::Nonlinear),
            (PhysicsFamily::Thermal, AnalysisProcedure::Static),
            (PhysicsFamily::Thermal, AnalysisProcedure::TransientResponse),
            (
                PhysicsFamily::Electromagnetic,
                AnalysisProcedure::HarmonicResponse,
            ),
            (PhysicsFamily::Acoustic, AnalysisProcedure::HarmonicResponse),
            (PhysicsFamily::Fluid, AnalysisProcedure::Static),
            (PhysicsFamily::Fluid, AnalysisProcedure::TransientResponse),
            (
                PhysicsFamily::Coupled(CoupledPhysicsFamily::ConjugateHeatTransfer),
                AnalysisProcedure::Static,
            ),
            (
                PhysicsFamily::Coupled(CoupledPhysicsFamily::ConjugateHeatTransfer),
                AnalysisProcedure::TransientResponse,
            ),
            (
                PhysicsFamily::Coupled(CoupledPhysicsFamily::FluidStructureInteraction),
                AnalysisProcedure::TransientResponse,
            ),
            (
                PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical),
                AnalysisProcedure::Static,
            ),
            (
                PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical),
                AnalysisProcedure::Nonlinear,
            ),
        ];

        for (family, procedure) in identities {
            assert!(is_exposed_analysis_identity(
                family,
                procedure,
                SolutionSpace::Direct
            ));
        }
        assert!(!is_exposed_analysis_identity(
            PhysicsFamily::Acoustic,
            AnalysisProcedure::Static,
            SolutionSpace::Direct
        ));
        assert!(!is_exposed_analysis_identity(
            PhysicsFamily::Structural,
            AnalysisProcedure::Static,
            SolutionSpace::ModalReduced
        ));
        assert!(!is_exposed_analysis_identity(
            PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical),
            AnalysisProcedure::TransientResponse,
            SolutionSpace::Direct
        ));
    }
}
