use std::collections::BTreeSet;

use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use crate::frequency::{FrequencyResponsePlanEntry, FrequencyResponsePlanEntryIdentity};

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, SolverResultIdentity,
    StructuralFrequencyPhysicalSample, StructuralFrequencyPhysicalSampleIdentity,
    StructuralFrequencyScope,
};

const FREQUENCY_RESULT_PREFIX: &[u8] = b"runmat-structural-frequency-result/v1\0";
const FREQUENCY_RESULT_LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 32));

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralFrequencyResultEntry {
    entry: FrequencyResponsePlanEntry,
    sample: StructuralFrequencyPhysicalSampleIdentity,
}

impl StructuralFrequencyResultEntry {
    pub const fn plan_entry(self) -> FrequencyResponsePlanEntry {
        self.entry
    }

    pub const fn sample(self) -> StructuralFrequencyPhysicalSampleIdentity {
        self.sample
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    try_from = "StructuralFrequencySolverResultManifestWire"
)]
pub struct StructuralFrequencySolverResultManifest {
    schema_version: u16,
    identity: SolverResultIdentity,
    scope: StructuralFrequencyScope,
    samples: Vec<StructuralFrequencyResultEntry>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralFrequencySolverResultManifestWire {
    schema_version: u16,
    identity: SolverResultIdentity,
    scope: StructuralFrequencyScope,
    samples: Vec<StructuralFrequencyResultEntry>,
}

impl TryFrom<StructuralFrequencySolverResultManifestWire>
    for StructuralFrequencySolverResultManifest
{
    type Error = AnalysisExecutionError;

    fn try_from(wire: StructuralFrequencySolverResultManifestWire) -> Result<Self, Self::Error> {
        let manifest = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            scope: wire.scope,
            samples: wire.samples,
        };
        manifest.validate()?;
        Ok(manifest)
    }
}

impl StructuralFrequencySolverResultManifest {
    pub const fn identity(&self) -> SolverResultIdentity {
        self.identity
    }

    pub fn scope(&self) -> &StructuralFrequencyScope {
        &self.scope
    }

    pub fn samples(&self) -> &[StructuralFrequencyResultEntry] {
        &self.samples
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        self.scope.validate()?;
        let expected = self
            .scope
            .plan()
            .entries()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        let unique_samples = self
            .samples
            .iter()
            .map(|entry| entry.sample)
            .collect::<BTreeSet<_>>();
        if self.schema_version != 1
            || self.samples.len() != expected.len()
            || unique_samples.len() != self.samples.len()
            || self
                .samples
                .iter()
                .zip(expected)
                .any(|(actual, expected)| actual.entry != expected)
            || self.identity != result_identity(&self.scope, &self.samples)?
        {
            return Err(AnalysisExecutionError::Identity(
                "structural frequency result manifest does not exactly cover its plan",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            FREQUENCY_RESULT_PREFIX,
            "structural-frequency-solver-result-manifest",
            self,
            FREQUENCY_RESULT_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            FREQUENCY_RESULT_PREFIX,
            "structural-frequency-solver-result-manifest",
            bytes,
            FREQUENCY_RESULT_LIMITS,
        )?)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct StructuralFrequencySolverResult {
    manifest: StructuralFrequencySolverResultManifest,
    samples: Vec<StructuralFrequencyPhysicalSample>,
}

impl StructuralFrequencySolverResult {
    pub fn assemble(
        scope: StructuralFrequencyScope,
        samples: Vec<StructuralFrequencyPhysicalSample>,
    ) -> AnalysisExecutionResult<Self> {
        scope.validate()?;
        let entries = samples
            .iter()
            .map(|sample| StructuralFrequencyResultEntry {
                entry: sample.entry(),
                sample: sample.identity(),
            })
            .collect::<Vec<_>>();
        let manifest = StructuralFrequencySolverResultManifest {
            schema_version: 1,
            identity: result_identity(&scope, &entries)?,
            scope,
            samples: entries,
        };
        Self::from_manifest_and_samples(manifest, samples)
    }

    pub fn from_manifest_and_samples(
        manifest: StructuralFrequencySolverResultManifest,
        samples: Vec<StructuralFrequencyPhysicalSample>,
    ) -> AnalysisExecutionResult<Self> {
        let result = Self { manifest, samples };
        result.validate()?;
        Ok(result)
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        self.manifest.validate()?;
        if self.samples.len() != self.manifest.samples.len() {
            return Err(AnalysisExecutionError::Identity(
                "structural frequency result payload count differs from its manifest",
            ));
        }
        for (manifest_entry, sample) in self.manifest.samples.iter().zip(&self.samples) {
            sample.validate_against(&self.manifest.scope)?;
            if manifest_entry.entry != sample.entry() || manifest_entry.sample != sample.identity()
            {
                return Err(AnalysisExecutionError::Identity(
                    "structural frequency result sample order or identity is invalid",
                ));
            }
        }
        Ok(())
    }

    pub fn manifest(&self) -> &StructuralFrequencySolverResultManifest {
        &self.manifest
    }

    pub fn samples(&self) -> &[StructuralFrequencyPhysicalSample] {
        &self.samples
    }

    pub fn sample(
        &self,
        entry: FrequencyResponsePlanEntryIdentity,
    ) -> Option<&StructuralFrequencyPhysicalSample> {
        self.samples
            .iter()
            .find(|sample| sample.entry().identity() == entry)
    }
}

fn result_identity(
    scope: &StructuralFrequencyScope,
    entries: &[StructuralFrequencyResultEntry],
) -> AnalysisExecutionResult<SolverResultIdentity> {
    let content = encode_contract(
        FREQUENCY_RESULT_PREFIX,
        "structural-frequency-solver-result-content",
        &(scope, entries),
        FREQUENCY_RESULT_LIMITS,
    )?;
    SolverResultIdentity::from_structural_frequency_response(
        scope.identity(),
        Digest::sha256(content),
    )
}
