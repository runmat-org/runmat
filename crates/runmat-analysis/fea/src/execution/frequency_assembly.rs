use std::collections::BTreeMap;

use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use serde::{Deserialize, Serialize};

use crate::frequency::FrequencyResponsePlanEntryIdentity;

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, StructuralFrequencyEntryResult,
    StructuralFrequencyEntryResultIdentity, StructuralFrequencyNumericalReceipt,
    StructuralFrequencyResultAssemblyRequestIdentity, StructuralFrequencyScope,
    StructuralFrequencySolutionSpace, StructuralFrequencySolverResult,
};

const FREQUENCY_ASSEMBLY_PREFIX: &[u8] = b"runmat-structural-frequency-assembly/v1\0";
const FREQUENCY_ASSEMBLY_LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralFrequencyAssemblyEntry {
    entry: FrequencyResponsePlanEntryIdentity,
    result: StructuralFrequencyEntryResultIdentity,
}

impl StructuralFrequencyAssemblyEntry {
    pub const fn entry(self) -> FrequencyResponsePlanEntryIdentity {
        self.entry
    }

    pub const fn result(self) -> StructuralFrequencyEntryResultIdentity {
        self.result
    }
}

/// Exact plan-ordered fan-in request over independently executable frequency results.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    try_from = "StructuralFrequencyResultAssemblyRequestWire"
)]
pub struct StructuralFrequencyResultAssemblyRequest {
    schema_version: u16,
    identity: StructuralFrequencyResultAssemblyRequestIdentity,
    scope: StructuralFrequencyScope,
    solution_space: StructuralFrequencySolutionSpace,
    entries: Vec<StructuralFrequencyAssemblyEntry>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralFrequencyResultAssemblyRequestWire {
    schema_version: u16,
    identity: StructuralFrequencyResultAssemblyRequestIdentity,
    scope: StructuralFrequencyScope,
    solution_space: StructuralFrequencySolutionSpace,
    entries: Vec<StructuralFrequencyAssemblyEntry>,
}

impl TryFrom<StructuralFrequencyResultAssemblyRequestWire>
    for StructuralFrequencyResultAssemblyRequest
{
    type Error = AnalysisExecutionError;

    fn try_from(wire: StructuralFrequencyResultAssemblyRequestWire) -> Result<Self, Self::Error> {
        let request = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            scope: wire.scope,
            solution_space: wire.solution_space,
            entries: wire.entries,
        };
        request.validate_identity()?;
        Ok(request)
    }
}

pub struct StructuralFrequencyAssemblyOutput {
    pub result: StructuralFrequencySolverResult,
    pub receipt: StructuralFrequencyNumericalReceipt,
}

impl StructuralFrequencyResultAssemblyRequest {
    pub fn new(results: &[StructuralFrequencyEntryResult]) -> AnalysisExecutionResult<Self> {
        let first = results.first().ok_or_else(|| {
            AnalysisExecutionError::Invalid(
                "structural frequency result assembly requires entry results".into(),
            )
        })?;
        first.validate()?;
        let scope = first.request().scope().clone();
        let solution_space = first.solution_space().clone();
        let mut by_entry = BTreeMap::new();
        for result in results {
            result.validate()?;
            if result.request().scope() != &scope
                || result.solution_space() != &solution_space
                || by_entry
                    .insert(result.request().entry().identity(), result.identity())
                    .is_some()
            {
                return Err(AnalysisExecutionError::Identity(
                    "structural frequency assembly contains incompatible or duplicate entries",
                ));
            }
        }
        let entries = scope
            .plan()
            .entries()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?
            .into_iter()
            .map(|entry| {
                by_entry
                    .remove(&entry.identity())
                    .map(|result| StructuralFrequencyAssemblyEntry {
                        entry: entry.identity(),
                        result,
                    })
                    .ok_or(AnalysisExecutionError::Identity(
                        "structural frequency assembly is missing a plan entry",
                    ))
            })
            .collect::<AnalysisExecutionResult<Vec<_>>>()?;
        if !by_entry.is_empty() {
            return Err(AnalysisExecutionError::Identity(
                "structural frequency assembly contains an unrelated entry",
            ));
        }
        let mut request = Self {
            schema_version: 1,
            identity: StructuralFrequencyResultAssemblyRequestIdentity::derive(
                "structural-frequency-assembly-placeholder",
                &1_u8,
            )?,
            scope,
            solution_space,
            entries,
        };
        request.identity = request.derive_identity()?;
        request.validate_for(results)?;
        Ok(request)
    }

    pub const fn identity(&self) -> StructuralFrequencyResultAssemblyRequestIdentity {
        self.identity
    }

    pub fn scope(&self) -> &StructuralFrequencyScope {
        &self.scope
    }

    pub fn solution_space(&self) -> &StructuralFrequencySolutionSpace {
        &self.solution_space
    }

    pub fn entries(&self) -> &[StructuralFrequencyAssemblyEntry] {
        &self.entries
    }

    pub fn validate_for(
        &self,
        results: &[StructuralFrequencyEntryResult],
    ) -> AnalysisExecutionResult<()> {
        self.validate_identity()?;
        if results.len() != self.entries.len() {
            return Err(AnalysisExecutionError::Identity(
                "structural frequency assembly result count differs from its request",
            ));
        }
        let mut by_identity = BTreeMap::new();
        for result in results {
            result.validate()?;
            if by_identity.insert(result.identity(), result).is_some() {
                return Err(AnalysisExecutionError::Identity(
                    "structural frequency assembly contains a duplicate result",
                ));
            }
        }
        for expected in &self.entries {
            let result =
                by_identity
                    .remove(&expected.result)
                    .ok_or(AnalysisExecutionError::Identity(
                        "structural frequency assembly is missing a requested result",
                    ))?;
            if result.request().entry().identity() != expected.entry
                || result.request().scope() != &self.scope
                || result.solution_space() != &self.solution_space
            {
                return Err(AnalysisExecutionError::Identity(
                    "structural frequency assembly result differs from its requested entry",
                ));
            }
        }
        if !by_identity.is_empty() {
            return Err(AnalysisExecutionError::Identity(
                "structural frequency assembly contains an unrelated result",
            ));
        }
        Ok(())
    }

    pub fn assemble(
        &self,
        results: &[StructuralFrequencyEntryResult],
    ) -> AnalysisExecutionResult<StructuralFrequencyAssemblyOutput> {
        self.validate_for(results)?;
        let by_identity = results
            .iter()
            .map(|result| (result.identity(), result))
            .collect::<BTreeMap<_, _>>();
        let ordered = self
            .entries
            .iter()
            .map(|entry| by_identity[&entry.result])
            .collect::<Vec<_>>();
        let result = StructuralFrequencySolverResult::assemble(
            self.scope.clone(),
            ordered.iter().map(|entry| entry.sample().clone()).collect(),
        )?;
        let receipt = StructuralFrequencyNumericalReceipt::new(
            &result,
            self.solution_space.clone(),
            ordered
                .iter()
                .map(|entry| entry.evidence().clone())
                .collect(),
        )?;
        Ok(StructuralFrequencyAssemblyOutput { result, receipt })
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate_identity()?;
        Ok(encode_contract(
            FREQUENCY_ASSEMBLY_PREFIX,
            "structural-frequency-result-assembly-request",
            self,
            FREQUENCY_ASSEMBLY_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            FREQUENCY_ASSEMBLY_PREFIX,
            "structural-frequency-result-assembly-request",
            bytes,
            FREQUENCY_ASSEMBLY_LIMITS,
        )?)
    }

    fn validate_identity(&self) -> AnalysisExecutionResult<()> {
        self.scope.validate()?;
        self.solution_space.validate(&self.scope)?;
        let expected_entries = self
            .scope
            .plan()
            .entries()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        if self.schema_version != 1
            || self.entries.len() != expected_entries.len()
            || self
                .entries
                .iter()
                .zip(expected_entries)
                .any(|(actual, expected)| actual.entry != expected.identity())
            || self.identity != self.derive_identity()?
        {
            return Err(AnalysisExecutionError::Identity(
                "structural frequency result assembly request identity does not converge",
            ));
        }
        Ok(())
    }

    fn derive_identity(
        &self,
    ) -> AnalysisExecutionResult<StructuralFrequencyResultAssemblyRequestIdentity> {
        StructuralFrequencyResultAssemblyRequestIdentity::derive(
            "structural-frequency-result-assembly-request",
            &(
                self.schema_version,
                &self.scope,
                &self.solution_space,
                &self.entries,
            ),
        )
    }
}
