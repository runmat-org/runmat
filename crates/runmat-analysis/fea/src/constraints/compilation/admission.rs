//! Caller-supplied component ceiling; allocation lifetimes remain compiler-owned.
use super::ConstraintPlanError;
use crate::constraints::workspace::{add, check_cancelled};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub enum ConstraintCompilationPhase {
    Preparation,
    Selection,
    Lowering,
    Group,
    Accumulation,
    RankProfile,
    Finalization,
}

/// Construction evidence for requested additional heap, excluding borrowed input, stack,
/// allocator overhead and RSS. Phase sums are conservative; not a serialized proof.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConstraintCompilationWorkspace {
    phase_bytes: [u64; 7],
}
impl ConstraintCompilationWorkspace {
    pub fn peak_requested_bytes(&self) -> u64 {
        self.phase_bytes.into_iter().max().unwrap_or(0)
    }
    pub fn phase_requested_bytes(&self, phase: ConstraintCompilationPhase) -> u64 {
        self.phase_bytes[phase as usize]
    }
}

pub(super) struct CompilationAdmission {
    maximum: u64,
    workspace: ConstraintCompilationWorkspace,
}
impl CompilationAdmission {
    pub(super) fn new(maximum: u64) -> Self {
        Self {
            maximum,
            workspace: ConstraintCompilationWorkspace {
                phase_bytes: [0; 7],
            },
        }
    }
    pub(super) fn require(
        &mut self,
        phase: ConstraintCompilationPhase,
        retained: u64,
        additional: u64,
    ) -> Result<(), ConstraintPlanError> {
        check_cancelled()?;
        let required = add(retained, additional)?;
        if required > self.maximum {
            return Err(ConstraintPlanError::WorkspaceLimit {
                phase,
                required,
                maximum: self.maximum,
            });
        }
        self.workspace.phase_bytes[phase as usize] =
            self.workspace.phase_bytes[phase as usize].max(required);
        Ok(())
    }
    pub(super) fn finish(self) -> ConstraintCompilationWorkspace {
        self.workspace
    }
}

#[cfg(test)]
mod tests;
