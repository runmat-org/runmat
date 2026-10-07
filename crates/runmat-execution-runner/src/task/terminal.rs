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
    Cancelled,
}

impl AttemptDisposition {
    pub(crate) fn state(self) -> AttemptState {
        match self {
            Self::Failed {
                kind: AttemptFailureKind::Rejected,
            } => AttemptState::Rejected,
            Self::Succeeded | Self::Failed { .. } => AttemptState::Completed,
            Self::Lost => AttemptState::Lost,
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
    /// Saved logical driver clock, not worker elapsed time or a cessation fence.
    pub closed_at_driver_millis: u64,
    /// Original report; None explicitly leaves consumption unaccounted.
    pub resources: Option<Box<AttemptResourceAccounting>>,
    pub execution: Option<Box<super::AttemptExecutionObservation>>,
    /// Original failure/loss report text, or explicit absence of diagnostic evidence.
    pub diagnostic: Option<String>,
}

impl AttemptTerminal {
    pub(crate) fn validate_against(&self, attempt: &AttemptRecord) -> Result<(), String> {
        if let Some(console) = &self.console {
            console.validate().map_err(|error| error.to_string())?;
            if Some(console.limits()) != attempt.request.task.console {
                return Err("console evidence differs from original attempt controls".into());
            }
        }
        if self.disposition == AttemptDisposition::Succeeded
            && attempt.request.task.console.is_some()
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
        if self.disposition == AttemptDisposition::Succeeded && self.resources.is_none() {
            return Err("successful attempt lacks original accounting".into());
        }
        Ok(())
    }
}

impl AttemptRecord {
    pub(crate) fn validate_terminal(&self, now_millis: u64) -> Result<(), String> {
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
