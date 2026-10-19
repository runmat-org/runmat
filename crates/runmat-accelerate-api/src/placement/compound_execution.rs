//! Shared cancellation and accumulated-effect adaptation for actual compound
//! physical children. No resource, numerical or completion grant is minted.
use crate::{ProviderExecutionError, ProviderExecutionResult};
use runmat_execution::cancellation::CancellationToken;

/// Only for an authenticated private output after protected borrowed aliases
/// have been excluded. Failure retirement keeps both the original execution
/// cause and any cleanup owner, rather than relying on silent final Drop.
pub(crate) fn retire_output_after_failure(
    output: crate::GpuTensorHandle,
    primary: ProviderExecutionError,
) -> ProviderExecutionError {
    let primary = primary.after_effect();
    let cleanup = output.release().err().into_iter().collect::<Vec<_>>();
    if cleanup.is_empty() {
        primary
    } else {
        ProviderExecutionError::from_source(crate::ProviderExecutionCompletionError::new(
            Some(primary),
            cleanup,
        ))
        .after_effect()
    }
}

#[derive(Debug)]
pub struct ProviderCompoundCancelled {
    pub reason: Option<runmat_execution::CancellationReason>,
}

impl std::fmt::Display for ProviderCompoundCancelled {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "compound provider execution cancelled ({:?})",
            self.reason
        )
    }
}

impl std::error::Error for ProviderCompoundCancelled {}

pub(crate) fn with_prior_effect<T>(
    result: ProviderExecutionResult<T>,
    effected: bool,
) -> ProviderExecutionResult<T> {
    result.map_err(|cause| {
        if effected {
            cause.after_effect()
        } else {
            cause
        }
    })
}

pub(crate) fn check_cancelled(
    token: Option<&CancellationToken>,
    after_effect: bool,
) -> ProviderExecutionResult<()> {
    if let Some(token) = token.filter(|token| token.is_cancelled()) {
        let error = ProviderExecutionError::from_source(ProviderCompoundCancelled {
            reason: token.reason(),
        });
        return Err(if after_effect {
            error.after_effect()
        } else {
            error
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn accumulated_probe_effects_dominate_child_applicability() {
        for prior_effect in [false, true] {
            for child_effect in [false, true] {
                let cause = ProviderExecutionError::from(crate::unsupported_provider_operation(
                    "child recipe unavailable",
                ));
                let cause = if child_effect {
                    cause.after_effect()
                } else {
                    cause
                };
                let result: ProviderExecutionResult<()> = Err(cause);
                let cause = with_prior_effect(result, prior_effect).unwrap_err();
                assert_eq!(cause.is_after_effect(), prior_effect || child_effect);
                assert_eq!(cause.is_unsupported(), !prior_effect && !child_effect);
                assert!(std::error::Error::source(&cause)
                    .is_some_and(|source| source.is::<crate::UnsupportedProviderOperation>()));
            }
        }
        let cause = ProviderExecutionError::from(crate::unsupported_provider_operation(
            "unproved child outcome",
        ))
        .with_unproven_effects();
        let result: ProviderExecutionResult<()> = Err(cause);
        assert!(!with_prior_effect(result, false)
            .unwrap_err()
            .is_unsupported());
    }
}
