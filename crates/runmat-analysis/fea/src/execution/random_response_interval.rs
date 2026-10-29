use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use serde::{Deserialize, Serialize};

use crate::solve::dynamics::ModalRandomResponseIntervalResult;

use super::{
    AnalysisExecutionError, AnalysisExecutionResult,
    StructuralRandomResponseIntervalRequestIdentity,
    StructuralRandomResponseIntervalResultIdentity, StructuralRandomResponseScope,
    StructuralRandomResponseWorkload,
};

const INTERVAL_PREFIX: &[u8] = b"runmat-structural-random-response-interval-task\0";
const INTERVAL_LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    try_from = "StructuralRandomResponseIntervalRequestWire"
)]
pub struct StructuralRandomResponseIntervalRequest {
    schema_version: u16,
    identity: StructuralRandomResponseIntervalRequestIdentity,
    scope: StructuralRandomResponseScope,
    task_ordinal: usize,
    interval_range: [usize; 2],
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralRandomResponseIntervalRequestWire {
    schema_version: u16,
    identity: StructuralRandomResponseIntervalRequestIdentity,
    scope: StructuralRandomResponseScope,
    task_ordinal: usize,
    interval_range: [usize; 2],
}

impl TryFrom<StructuralRandomResponseIntervalRequestWire>
    for StructuralRandomResponseIntervalRequest
{
    type Error = AnalysisExecutionError;

    fn try_from(wire: StructuralRandomResponseIntervalRequestWire) -> Result<Self, Self::Error> {
        let request = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            scope: wire.scope,
            task_ordinal: wire.task_ordinal,
            interval_range: wire.interval_range,
        };
        request.validate()?;
        Ok(request)
    }
}

impl StructuralRandomResponseIntervalRequest {
    pub fn new(
        workload: &StructuralRandomResponseWorkload,
        scope: StructuralRandomResponseScope,
        task_ordinal: usize,
    ) -> AnalysisExecutionResult<Self> {
        scope.validate_for(workload)?;
        let interval_range = *scope
            .task_ranges()
            .get(task_ordinal)
            .ok_or_else(|| invalid("random-response task ordinal is outside the plan"))?;
        let mut request = Self {
            schema_version: 1,
            identity: StructuralRandomResponseIntervalRequestIdentity::derive(
                "structural-random-response-interval-request-placeholder",
                &1_u8,
            )?,
            scope,
            task_ordinal,
            interval_range,
        };
        request.identity = request.derive_identity()?;
        request.validate_for(workload)?;
        Ok(request)
    }

    pub const fn identity(&self) -> StructuralRandomResponseIntervalRequestIdentity {
        self.identity
    }

    pub fn scope(&self) -> &StructuralRandomResponseScope {
        &self.scope
    }

    pub const fn task_ordinal(&self) -> usize {
        self.task_ordinal
    }

    pub const fn interval_range(&self) -> [usize; 2] {
        self.interval_range
    }

    pub fn validate_for(
        &self,
        workload: &StructuralRandomResponseWorkload,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        self.scope.validate_for(workload)
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            INTERVAL_PREFIX,
            "structural-random-response-interval-request",
            self,
            INTERVAL_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            INTERVAL_PREFIX,
            "structural-random-response-interval-request",
            bytes,
            INTERVAL_LIMITS,
        )?)
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        self.scope.validate()?;
        if self.schema_version != 1
            || self
                .scope
                .task_ranges()
                .get(self.task_ordinal)
                .is_none_or(|expected| *expected != self.interval_range)
            || self.identity != self.derive_identity()?
        {
            return Err(invalid("random-response interval request is not canonical"));
        }
        Ok(())
    }

    fn derive_identity(
        &self,
    ) -> AnalysisExecutionResult<StructuralRandomResponseIntervalRequestIdentity> {
        StructuralRandomResponseIntervalRequestIdentity::derive(
            "structural-random-response-interval-request",
            &(
                self.schema_version,
                self.scope.identity(),
                self.task_ordinal,
                self.interval_range,
            ),
        )
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    try_from = "StructuralRandomResponseIntervalResultWire"
)]
pub struct StructuralRandomResponseIntervalResult {
    schema_version: u16,
    identity: StructuralRandomResponseIntervalResultIdentity,
    request: StructuralRandomResponseIntervalRequest,
    intervals: Vec<ModalRandomResponseIntervalResult>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralRandomResponseIntervalResultWire {
    schema_version: u16,
    identity: StructuralRandomResponseIntervalResultIdentity,
    request: StructuralRandomResponseIntervalRequest,
    intervals: Vec<ModalRandomResponseIntervalResult>,
}

impl TryFrom<StructuralRandomResponseIntervalResultWire>
    for StructuralRandomResponseIntervalResult
{
    type Error = AnalysisExecutionError;

    fn try_from(wire: StructuralRandomResponseIntervalResultWire) -> Result<Self, Self::Error> {
        let result = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            request: wire.request,
            intervals: wire.intervals,
        };
        result.validate()?;
        Ok(result)
    }
}

impl StructuralRandomResponseIntervalResult {
    pub fn new(
        request: StructuralRandomResponseIntervalRequest,
        intervals: Vec<ModalRandomResponseIntervalResult>,
    ) -> AnalysisExecutionResult<Self> {
        let mut result = Self {
            schema_version: 1,
            identity: StructuralRandomResponseIntervalResultIdentity::derive(
                "structural-random-response-interval-result-placeholder",
                &1_u8,
            )?,
            request,
            intervals,
        };
        result.identity = result.derive_identity()?;
        result.validate()?;
        Ok(result)
    }

    pub const fn identity(&self) -> StructuralRandomResponseIntervalResultIdentity {
        self.identity
    }

    pub fn request(&self) -> &StructuralRandomResponseIntervalRequest {
        &self.request
    }

    pub fn intervals(&self) -> &[ModalRandomResponseIntervalResult] {
        &self.intervals
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            INTERVAL_PREFIX,
            "structural-random-response-interval-result",
            self,
            INTERVAL_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            INTERVAL_PREFIX,
            "structural-random-response-interval-result",
            bytes,
            INTERVAL_LIMITS,
        )?)
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        self.request.validate()?;
        let range = self.request.interval_range;
        if self.schema_version != 1
            || self.intervals.len() != range[1] - range[0]
            || self.intervals.iter().enumerate().any(|(offset, interval)| {
                interval.validate().is_err()
                    || interval.ordinal() != range[0] + offset
                    || interval.basis() != self.request.scope().basis()
                    || interval.excitation() != self.request.scope().plan().excitation()
                    || interval.excitation_digest()
                        != self.request.scope().plan().excitation_digest()
                    || interval.policy() != self.request.scope().plan().policy()
                    || interval.frequency_bounds_hz()
                        != [
                            self.request.scope().plan().breakpoints_hz()[range[0] + offset],
                            self.request.scope().plan().breakpoints_hz()[range[0] + offset + 1],
                        ]
            })
            || self.identity != self.derive_identity()?
        {
            return Err(invalid(
                "random-response interval result is incomplete or inconsistent",
            ));
        }
        Ok(())
    }

    fn derive_identity(
        &self,
    ) -> AnalysisExecutionResult<StructuralRandomResponseIntervalResultIdentity> {
        StructuralRandomResponseIntervalResultIdentity::derive(
            "structural-random-response-interval-result",
            &(
                self.schema_version,
                self.request.identity(),
                self.intervals
                    .iter()
                    .map(ModalRandomResponseIntervalResult::identity)
                    .collect::<Vec<_>>(),
            ),
        )
    }
}

fn invalid(message: impl Into<String>) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(message.into())
}
