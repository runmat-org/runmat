//! Resource phases that construct or recover state without a numerical solve.

use super::FeaResourcePhase;

pub(super) fn requires_no_solve_policy(phase: &FeaResourcePhase) -> bool {
    matches!(
        phase,
        FeaResourcePhase::StructuralNonlinearRecovery
            | FeaResourcePhase::StructuralFrequencyFieldRecovery(_)
            | FeaResourcePhase::StructuralTransientScopePreparation(_)
            | FeaResourcePhase::StructuralTransientFieldRecovery(_)
            | FeaResourcePhase::StructuralModalResponseSpectrumDirection(_)
            | FeaResourcePhase::StructuralRandomResponseInterval(_)
    )
}
