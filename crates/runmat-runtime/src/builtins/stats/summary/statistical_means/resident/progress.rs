//! One request's provider-effect boundary across resident mean stages.

use super::{contract_failure, provider_failure, Failure};
use crate::builtins::stats::summary::statistical_means::options::ReductionKind;
use runmat_accelerate_api::ProviderExecutionError;

#[derive(Default)]
pub(super) struct ResidentStageProgress {
    phase: Phase,
}

#[derive(Clone, Copy, Default)]
enum Phase {
    #[default]
    BeforeEffect,
    AfterEffect,
}

impl ResidentStageProgress {
    pub(super) fn observe_optional<T>(
        &mut self,
        kind: ReductionKind,
        result: runmat_accelerate_api::ProviderExecutionResult<Option<T>>,
        detail: &'static str,
    ) -> Result<T, Failure> {
        match result {
            Ok(Some(output)) => self.observe(kind, Ok::<T, ProviderExecutionError>(output)),
            Ok(None) => Err(self.unavailable(kind, detail)),
            Err(error) => self.observe(kind, Err(error.with_unproven_effects())),
        }
    }

    /// Observe a real provider hook, not static validation or a local identity
    /// return. A successful hook crosses the replay boundary before its output
    /// is admitted; a pending hook never reaches this method.
    pub(super) fn observe<T, E: Into<ProviderExecutionError>>(
        &mut self,
        kind: ReductionKind,
        result: Result<T, E>,
    ) -> Result<T, Failure> {
        match result {
            Ok(value) => {
                self.phase = Phase::AfterEffect;
                Ok(value)
            }
            Err(source) => {
                let source = source.into();
                let source = match self.phase {
                    Phase::BeforeEffect => source,
                    Phase::AfterEffect => source.after_effect(),
                };
                Err(provider_failure(kind, source))
            }
        }
    }

    /// A nested static rejection cannot turn completed provider work into a
    /// host retry. It has no provider cause to fabricate.
    pub(super) fn unavailable(&self, kind: ReductionKind, detail: &'static str) -> Failure {
        match self.phase {
            Phase::BeforeEffect => Failure::Unsupported,
            Phase::AfterEffect => contract_failure(kind, detail),
        }
    }
}

pub(super) trait ObserveResidentHook<T> {
    fn observe(
        self,
        progress: &mut ResidentStageProgress,
        kind: ReductionKind,
    ) -> Result<T, Failure>;
}

impl<T, E: Into<ProviderExecutionError>> ObserveResidentHook<T> for Result<T, E> {
    fn observe(
        self,
        progress: &mut ResidentStageProgress,
        kind: ReductionKind,
    ) -> Result<T, Failure> {
        progress.observe(kind, self)
    }
}
