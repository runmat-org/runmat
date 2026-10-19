use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, AnalysisResultIdentity, StudyIdentity,
    SweepIdentity, SweepResultIdentity,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SweepCompositionPolicy {
    FailFast,
    CollectAll,
}

impl SweepCompositionPolicy {
    pub const fn from_fail_fast(fail_fast: bool) -> Self {
        if fail_fast {
            Self::FailFast
        } else {
            Self::CollectAll
        }
    }

    pub const fn fail_fast(self) -> bool {
        matches!(self, Self::FailFast)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SweepDefinition {
    sweep_contract_digest: Digest,
    studies: Vec<StudyIdentity>,
    composition: SweepCompositionPolicy,
}

impl SweepDefinition {
    pub fn new(
        sweep_contract_digest: Digest,
        studies: Vec<StudyIdentity>,
        composition: SweepCompositionPolicy,
    ) -> AnalysisExecutionResult<Self> {
        if studies.is_empty() {
            return Err(AnalysisExecutionError::Invalid(
                "sweep definition requires at least one study".into(),
            ));
        }
        let mut unique = studies.clone();
        unique.sort_unstable();
        unique.dedup();
        if unique.len() != studies.len() {
            return Err(AnalysisExecutionError::Invalid(
                "sweep definition contains duplicate studies".into(),
            ));
        }
        Ok(Self {
            sweep_contract_digest,
            studies,
            composition,
        })
    }

    pub fn identity(&self) -> AnalysisExecutionResult<SweepIdentity> {
        SweepIdentity::from_definition(self)
    }

    pub const fn sweep_contract_digest(&self) -> Digest {
        self.sweep_contract_digest
    }

    pub fn studies(&self) -> &[StudyIdentity] {
        &self.studies
    }

    pub const fn composition(&self) -> SweepCompositionPolicy {
        self.composition
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "disposition")]
pub enum SweepEntryDisposition {
    Succeeded { result: AnalysisResultIdentity },
    Failed { failure_evidence: Digest },
    Cancelled,
    Indeterminate { recovery_evidence: Digest },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SweepEntryOutcome {
    pub ordinal: u32,
    pub study: StudyIdentity,
    pub disposition: SweepEntryDisposition,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComposedSweepResult {
    pub sweep: SweepIdentity,
    pub identity: SweepResultIdentity,
    pub entries: Vec<SweepEntryOutcome>,
}

impl ComposedSweepResult {
    /// Canonically composes completed entries in authored order, independent of
    /// worker completion order.
    pub fn new(
        sweep: SweepIdentity,
        expected_studies: &[StudyIdentity],
        mut entries: Vec<SweepEntryOutcome>,
    ) -> AnalysisExecutionResult<Self> {
        if expected_studies.is_empty() || entries.len() != expected_studies.len() {
            return Err(AnalysisExecutionError::Invalid(
                "sweep outcomes do not cover every authored study".into(),
            ));
        }
        entries.sort_unstable_by_key(|entry| entry.ordinal);
        for (ordinal, (entry, expected_study)) in entries.iter().zip(expected_studies).enumerate() {
            if usize::try_from(entry.ordinal).ok() != Some(ordinal)
                || entry.study != *expected_study
            {
                return Err(AnalysisExecutionError::Invalid(
                    "sweep outcomes differ from authored ordinal or study identity".into(),
                ));
            }
        }
        let identity = SweepResultIdentity::from_entries(sweep, &entries)?;
        Ok(Self {
            sweep,
            identity,
            entries,
        })
    }
}

impl SweepResultIdentity {
    fn from_entries(
        sweep: SweepIdentity,
        entries: &[SweepEntryOutcome],
    ) -> AnalysisExecutionResult<Self> {
        Self::derive("sweep-result", &(sweep, entries))
    }
}
