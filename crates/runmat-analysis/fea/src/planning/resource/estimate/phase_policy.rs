//! Resource phases that construct or recover state without a numerical solve.

use super::FeaResourcePhase;

pub(super) fn requires_no_solve_policy(phase: &FeaResourcePhase) -> bool {
    matches!(
        phase,
        FeaResourcePhase::StructuralFrequencyFieldRecovery(_)
            | FeaResourcePhase::StructuralTransientScopePreparation(_)
            | FeaResourcePhase::StructuralModalResponseSpectrumDirection(_)
            | FeaResourcePhase::StructuralRandomResponseInterval(_)
    )
}
