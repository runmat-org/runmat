use runmat_analysis_core::{
    AccuracyObservationEvent, ObservationEvent, SeriesObservationEvent, SolverObservationEvent,
};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use crate::{
    execution::{AnalysisAttemptIdentity, SolverResultIdentity},
    results::{accuracy::AccuracyReportIdentity, field::RecoveredFieldSample},
};

use super::super::contract::ObservationError;
use super::{contract::ObservationHistoryIdentity, schedule::validate_coordinate};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "source")]
pub enum SolverEventSource {
    CompletedResult {
        result: SolverResultIdentity,
    },
    FailedAttempt {
        attempt: AnalysisAttemptIdentity,
        failure_evidence: Digest,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverEventAttribution {
    pub event: SolverObservationEvent,
    pub sample: RecoveredFieldSample,
    pub occurrence_index: u64,
    pub source: SolverEventSource,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeriesEventAttribution {
    pub event: SeriesObservationEvent,
    pub sample: RecoveredFieldSample,
    pub source_history: ObservationHistoryIdentity,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AccuracyEventAttribution {
    pub event: AccuracyObservationEvent,
    pub sample: RecoveredFieldSample,
    pub source_report: AccuracyReportIdentity,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "owner")]
pub enum ObservationEventAttribution {
    Solver(SolverEventAttribution),
    Series(SeriesEventAttribution),
    Accuracy(AccuracyEventAttribution),
}

impl ObservationEventAttribution {
    pub const fn event(self) -> ObservationEvent {
        match self {
            Self::Solver(attribution) => ObservationEvent::Solver(attribution.event),
            Self::Series(attribution) => ObservationEvent::Series(attribution.event),
            Self::Accuracy(attribution) => ObservationEvent::Accuracy(attribution.event),
        }
    }

    pub const fn sample(self) -> RecoveredFieldSample {
        match self {
            Self::Solver(attribution) => attribution.sample,
            Self::Series(attribution) => attribution.sample,
            Self::Accuracy(attribution) => attribution.sample,
        }
    }

    pub fn validate(self) -> Result<(), ObservationError> {
        validate_coordinate(self.sample())?;
        match self {
            Self::Solver(SolverEventAttribution {
                event: SolverObservationEvent::ConvergenceFailure,
                occurrence_index: 0,
                source: SolverEventSource::FailedAttempt { .. },
                ..
            })
            | Self::Solver(SolverEventAttribution {
                event: SolverObservationEvent::FirstYield,
                occurrence_index: 0,
                source: SolverEventSource::CompletedResult { .. },
                ..
            })
            | Self::Solver(SolverEventAttribution {
                event: SolverObservationEvent::ContactStatusChange,
                source: SolverEventSource::CompletedResult { .. },
                ..
            })
            | Self::Series(_)
            | Self::Accuracy(_) => Ok(()),
            Self::Solver(_) => Err(ObservationError::Invalid(
                "solver event attribution uses an incompatible source or occurrence index".into(),
            )),
        }
    }

    pub const fn occurrence_index(self) -> u64 {
        match self {
            Self::Solver(attribution) => attribution.occurrence_index,
            Self::Series(_) | Self::Accuracy(_) => 0,
        }
    }
}
