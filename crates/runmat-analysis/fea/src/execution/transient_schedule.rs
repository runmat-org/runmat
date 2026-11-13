use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use serde::{Deserialize, Serialize};

use super::{AnalysisExecutionError, AnalysisExecutionResult, StructuralTransientScheduleIdentity};

const TRANSIENT_SCHEDULE_PREFIX: &[u8] = b"runmat-structural-transient-schedule/v1\0";
const TRANSIENT_SCHEDULE_LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 32));

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuralTransientIntegrator {
    NewmarkAverageAcceleration,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuralTransientPrecision {
    Real64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuralTransientScheduleArrival {
    Initial,
    Continuous,
    LeftLimit,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuralTransientContinuationApproach {
    At,
    RightLimit,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralTransientSchedulePoint {
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub time_s: runmat_types::physical::Time,
    pub arrival: StructuralTransientScheduleArrival,
    pub continuation: StructuralTransientContinuationApproach,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "StructuralTransientScheduleWire")]
pub struct StructuralTransientSchedule {
    pub schema_version: u16,
    pub identity: StructuralTransientScheduleIdentity,
    pub points: Vec<StructuralTransientSchedulePoint>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralTransientScheduleWire {
    schema_version: u16,
    identity: StructuralTransientScheduleIdentity,
    points: Vec<StructuralTransientSchedulePoint>,
}

impl TryFrom<StructuralTransientScheduleWire> for StructuralTransientSchedule {
    type Error = AnalysisExecutionError;

    fn try_from(wire: StructuralTransientScheduleWire) -> Result<Self, Self::Error> {
        let schedule = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            points: wire.points,
        };
        schedule.validate()?;
        Ok(schedule)
    }
}

impl StructuralTransientSchedule {
    pub fn new(points: Vec<StructuralTransientSchedulePoint>) -> AnalysisExecutionResult<Self> {
        let identity = schedule_identity(1, &points)?;
        let schedule = Self {
            schema_version: 1,
            identity,
            points,
        };
        schedule.validate()?;
        Ok(schedule)
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        if self.schema_version != 1 || self.points.len() < 2 {
            return Err(AnalysisExecutionError::Invalid(
                "structural transient schedule must contain an initial point and at least one accepted interval"
                    .into(),
            ));
        }
        for (ordinal, point) in self.points.iter().enumerate() {
            if !point.time_s.canonical_value().is_finite()
                || (point.time_s.canonical_value() == 0.0
                    && point.time_s.canonical_value().is_sign_negative())
            {
                return Err(AnalysisExecutionError::Invalid(
                    "structural transient schedule contains a noncanonical time".into(),
                ));
            }
            if ordinal == 0 {
                if point.arrival != StructuralTransientScheduleArrival::Initial
                    || point.continuation != StructuralTransientContinuationApproach::At
                {
                    return Err(AnalysisExecutionError::Invalid(
                        "the first structural transient schedule point must be the initial At state"
                            .into(),
                    ));
                }
                continue;
            }
            if point.time_s.canonical_value() <= self.points[ordinal - 1].time_s.canonical_value() {
                return Err(AnalysisExecutionError::Invalid(
                    "structural transient schedule points must be strictly ordered".into(),
                ));
            }
            let expected = match point.arrival {
                StructuralTransientScheduleArrival::LeftLimit => {
                    StructuralTransientContinuationApproach::RightLimit
                }
                StructuralTransientScheduleArrival::Continuous => {
                    StructuralTransientContinuationApproach::At
                }
                StructuralTransientScheduleArrival::Initial => {
                    return Err(AnalysisExecutionError::Invalid(
                        "only the first structural transient schedule point may be initial".into(),
                    ));
                }
            };
            if point.continuation != expected {
                return Err(AnalysisExecutionError::Invalid(
                    "structural transient schedule arrival and continuation sides disagree".into(),
                ));
            }
        }
        if self.identity != schedule_identity(self.schema_version, &self.points)? {
            return Err(AnalysisExecutionError::Identity(
                "structural transient schedule identity does not converge",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            TRANSIENT_SCHEDULE_PREFIX,
            "structural-transient-schedule",
            self,
            TRANSIENT_SCHEDULE_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            TRANSIENT_SCHEDULE_PREFIX,
            "structural-transient-schedule",
            bytes,
            TRANSIENT_SCHEDULE_LIMITS,
        )?)
    }
}

fn schedule_identity(
    schema_version: u16,
    points: &[StructuralTransientSchedulePoint],
) -> AnalysisExecutionResult<StructuralTransientScheduleIdentity> {
    StructuralTransientScheduleIdentity::derive(
        "structural-transient-schedule",
        &(schema_version, points),
    )
}
