//! Original terminal decisions and worker accounting. Driver closure is not proof
//! that a lost process stopped, nor a measurement of its resource consumption.
use runmat_execution::state::AttemptState;
use serde::{Deserialize, Serialize};

use super::{AttemptFailureKind, AttemptRecord, AttemptResourceAccounting};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case", deny_unknown_fields)]
pub enum AttemptDisposition {
    Succeeded,
    Failed { kind: AttemptFailureKind },
    Lost,
    CertifiedNotStarted,
    Cancelled,
}

impl AttemptDisposition {
    pub(crate) fn state(self) -> AttemptState {
        match self {
            Self::Failed {
                kind: AttemptFailureKind::Rejected,
            } => AttemptState::Rejected,
            Self::Succeeded | Self::Failed { .. } => AttemptState::Completed,
            Self::Lost | Self::CertifiedNotStarted => AttemptState::Lost,
            Self::Cancelled => AttemptState::Cancelled,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttemptTerminal {
    pub disposition: AttemptDisposition,
    #[serde(deserialize_with = "runmat_execution::console::deserialize_transcript")]
    pub console: Option<Box<runmat_execution::console::ConsoleTranscript>>,
    #[serde(deserialize_with = "runmat_execution::console::deserialize_scopes")]
    pub console_scopes: Vec<runmat_execution::console::ScopedConsoleTranscript>,
    #[serde(deserialize_with = "runmat_execution::console::deserialize_source_identity")]
    pub delegated_console_root: Option<runmat_execution::console::ConsoleSourceIdentity>,
    /// Saved logical driver clock, not worker elapsed time or a cessation fence.
    pub closed_at_driver_millis: u64,
    /// Original report; None explicitly leaves consumption unaccounted.
    pub resources: Option<Box<AttemptResourceAccounting>>,
    pub execution: Option<Box<super::AttemptExecutionObservation>>,
    /// Original failure/loss report text, or explicit absence of diagnostic evidence.
    pub diagnostic: Option<String>,
}

/// An original provider report, not a cessation proof or measured final total.
/// Kept on the same attempt while its admitted allocation remains reserved.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttemptRetirementUnverified {
    pub observed_at_driver_millis: u64,
    pub diagnostic: String,
    pub failure: Option<Box<runmat_execution::ProgramRuntimeFailure>>,
    pub failure_kind: Option<AttemptFailureKind>,
    pub resources: Option<Box<AttemptResourceAccounting>>,
    pub execution: Option<Box<super::AttemptExecutionObservation>>,
    #[serde(deserialize_with = "runmat_execution::console::deserialize_transcript")]
    pub console: Option<Box<runmat_execution::console::ConsoleTranscript>>,
    #[serde(deserialize_with = "runmat_execution::console::deserialize_scopes")]
    pub console_scopes: Vec<runmat_execution::console::ScopedConsoleTranscript>,
    #[serde(deserialize_with = "runmat_execution::console::deserialize_source_identity")]
    pub delegated_console_root: Option<runmat_execution::console::ConsoleSourceIdentity>,
    /// Set only by the later exact fenced backend retirement certificate.
    pub retired_at_driver_millis: Option<u64>,
}

impl AttemptTerminal {
    pub(crate) fn validate_against(&self, attempt: &AttemptRecord) -> Result<(), String> {
        if attempt
            .unverified_retirement
            .as_ref()
            .is_some_and(|original| {
                original
                    .resources
                    .as_ref()
                    .is_some_and(|saved| self.resources.as_ref() != Some(saved))
                    || original
                        .execution
                        .as_ref()
                        .is_some_and(|saved| self.execution.as_ref() != Some(saved))
                    || original
                        .console
                        .as_ref()
                        .is_some_and(|saved| self.console.as_ref() != Some(saved))
                    || (!original.console_scopes.is_empty()
                        && self.console_scopes != original.console_scopes)
                    || (original.delegated_console_root.is_some()
                        && self.delegated_console_root != original.delegated_console_root)
            })
        {
            return Err("terminal report replaced original uncertain attempt evidence".into());
        }
        if self.disposition == AttemptDisposition::CertifiedNotStarted
            && (self.resources.is_some()
                || self.execution.is_some()
                || self.console.is_some()
                || !self.console_scopes.is_empty()
                || self.delegated_console_root.is_some()
                || self.diagnostic.is_some()
                || attempt
                    .unverified_retirement
                    .as_ref()
                    .is_none_or(|uncertainty| {
                        uncertainty.resources.is_some()
                            || uncertainty.execution.is_some()
                            || uncertainty.console.is_some()
                            || !uncertainty.console_scopes.is_empty()
                            || uncertainty.delegated_console_root.is_some()
                            || uncertainty.failure.is_some()
                            || uncertainty.failure_kind.is_some()
                    }))
        {
            return Err("certified never-started attempt carries execution evidence or lacks original uncertainty".into());
        }
        if let Some(console) = &self.console {
            console.validate().map_err(|error| error.to_string())?;
            if Some(console.limits()) != attempt.request.task.console {
                return Err("console evidence differs from original attempt controls".into());
            }
        }
        runmat_execution::console::validate_scopes(&self.console_scopes)
            .map_err(|error| error.to_string())?;
        runmat_execution::console::validate_delegated_root(
            self.console.as_deref(),
            &self.console_scopes,
            self.delegated_console_root,
            self.disposition == AttemptDisposition::Succeeded,
        )
        .map_err(|error| error.to_string())?;
        runmat_execution::console::validate_source_controls(
            self.console.as_deref(),
            &self.console_scopes,
            self.delegated_console_root,
            attempt.request.task.console,
            attempt.request.id,
        )
        .map_err(|error| error.to_string())?;
        if self.disposition == AttemptDisposition::Succeeded
            && attempt.request.task.console.is_some()
            && self.delegated_console_root.is_none()
            && self.console.as_ref().is_none_or(|console| {
                console.completion() != runmat_execution::console::ConsoleCompletion::Complete
            })
        {
            return Err("successful attempt lacks complete requested console evidence".into());
        }
        if self.closed_at_driver_millis < attempt.assigned_at_millis {
            return Err("attempt closure precedes its assignment".into());
        }
        if let Some(execution) = &self.execution {
            execution.validate_against(&attempt.request)?;
            if self.disposition == AttemptDisposition::Succeeded
                && !matches!(
                    execution.termination,
                    super::AttemptExecutionTermination::ProcessExited { success: true, .. }
                )
            {
                return Err("successful attempt contradicts process exit evidence".into());
            }
            if self.resources.as_ref().is_some_and(|resources| {
                resources.resources.usage.wall_millis != execution.wall_millis
            }) {
                return Err("original accounting differs from observed execution interval".into());
            }
        }
        if let Some(resources) = &self.resources {
            resources.validate_against(&attempt.request)?;
        }
        // Value successes are admitted only with worker accounting (see
        // AttemptSuccess::validate_against); graph and SPMD successes carry none.
        Ok(())
    }
}

impl AttemptRecord {
    pub(crate) fn validate_terminal(&self, now_millis: u64) -> Result<(), String> {
        if let Some(uncertainty) = &self.unverified_retirement {
            if let Some(failure) = &uncertainty.failure {
                failure.validate().map_err(|error| error.to_string())?;
            }
            if uncertainty.diagnostic.is_empty()
                || uncertainty.observed_at_driver_millis < self.assigned_at_millis
                || uncertainty.observed_at_driver_millis > now_millis
                || uncertainty
                    .retired_at_driver_millis
                    .is_some_and(|at| at < uncertainty.observed_at_driver_millis || at > now_millis)
                || uncertainty.retired_at_driver_millis.is_some() != self.terminal.is_some()
                || uncertainty.retired_at_driver_millis.is_some_and(|at| {
                    self.terminal
                        .as_ref()
                        .is_none_or(|terminal| terminal.closed_at_driver_millis != at)
                })
            {
                return Err("attempt retirement uncertainty contradicts original history".into());
            }
            if let Some(resources) = &uncertainty.resources {
                resources.validate_against(&self.request)?;
            }
            if let Some(execution) = &uncertainty.execution {
                execution.validate_against(&self.request)?;
                if uncertainty.resources.as_ref().is_some_and(|resources| {
                    resources.resources.usage.wall_millis != execution.wall_millis
                }) {
                    return Err("uncertain retirement accounting differs from observation".into());
                }
            }
            if let Some(console) = &uncertainty.console {
                console.validate().map_err(|error| error.to_string())?;
                if Some(console.limits()) != self.request.task.console {
                    return Err(
                        "uncertain retirement console differs from original controls".into(),
                    );
                }
            }
            runmat_execution::console::validate_scopes(&uncertainty.console_scopes)
                .map_err(|error| error.to_string())?;
            runmat_execution::console::validate_delegated_root(
                uncertainty.console.as_deref(),
                &uncertainty.console_scopes,
                uncertainty.delegated_console_root,
                false,
            )
            .map_err(|error| error.to_string())?;
            runmat_execution::console::validate_source_controls(
                uncertainty.console.as_deref(),
                &uncertainty.console_scopes,
                uncertainty.delegated_console_root,
                self.request.task.console,
                self.request.id,
            )
            .map_err(|error| error.to_string())?;
        }
        if self.state == AttemptState::Lost && self.unverified_retirement.is_none() {
            return Err("lost attempt lacks its original uncertainty and retirement proof".into());
        }
        if self.assigned_at_millis > now_millis
            || self
                .cancellation_requested_at
                .is_some_and(|at| at < self.assigned_at_millis || at > now_millis)
        {
            return Err("attempt timing is outside the saved driver interval".into());
        }
        match (&self.terminal, self.state) {
            (Some(terminal), state) if terminal.disposition.state() == state => {
                terminal.validate_against(self)?;
                if terminal.closed_at_driver_millis > now_millis {
                    return Err("attempt closure exceeds the saved driver clock".into());
                }
                Ok(())
            }
            (None, AttemptState::Assigned | AttemptState::Starting | AttemptState::Running) => {
                Ok(())
            }
            _ => Err("attempt state differs from its terminal evidence".into()),
        }
    }
}
