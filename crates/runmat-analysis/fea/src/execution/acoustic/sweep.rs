//! Plan-ordered fan-out/fan-in over canonical acoustic solves. This does not publish
//! a multi-frequency response as one single-frequency lifecycle state.
use std::collections::BTreeMap;

use super::*;
use crate::frequency::{FrequencyResponsePlan, FrequencyResponsePlanEntry};
use runmat_analysis_core::IndependentDomain;
use runmat_meshing_core::StableDigest;

const WORKLOAD_PREFIX: &[u8] = b"runmat-acoustic-frequency-workload/v1\0";
const RESULT_PREFIX: &[u8] = b"runmat-acoustic-frequency-result/v1\0";
const RESULT_LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));

mod publication;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "AcousticFrequencyWorkloadWire")]
pub struct AcousticFrequencyWorkload {
    template: AcousticHarmonicWorkload,
    plan: FrequencyResponsePlan,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AcousticFrequencyWorkloadWire {
    template: AcousticHarmonicWorkload,
    plan: FrequencyResponsePlan,
}

impl TryFrom<AcousticFrequencyWorkloadWire> for AcousticFrequencyWorkload {
    type Error = AnalysisExecutionError;
    fn try_from(wire: AcousticFrequencyWorkloadWire) -> AnalysisExecutionResult<Self> {
        let workload = Self {
            template: wire.template,
            plan: wire.plan,
        };
        workload.validate()?;
        Ok(workload)
    }
}

impl AcousticFrequencyWorkload {
    pub fn new(template: AcousticHarmonicWorkload) -> AnalysisExecutionResult<Self> {
        let plan = Self::derive_plan(&template)?;
        let workload = Self { template, plan };
        workload.validate()?;
        Ok(workload)
    }

    fn derive_plan(
        template: &AcousticHarmonicWorkload,
    ) -> AnalysisExecutionResult<FrequencyResponsePlan> {
        template.validate()?;
        if template.model.sequence.steps.len() != 1 {
            return Err(invalid(
                "acoustic frequency workload requires one standalone step",
            ));
        }
        let IndependentDomain::FrequencyHertz(frequencies) = &template.step()?.independent_domain
        else {
            return Err(invalid(
                "acoustic frequency workload requires an authored frequency domain",
            ));
        };
        FrequencyResponsePlan::new(template.frequency.canonical_value(), frequencies, &[])
            .map_err(invalid)
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        if self.plan != Self::derive_plan(&self.template)? {
            return Err(invalid(
                "acoustic frequency plan differs from its authored domain",
            ));
        }
        Ok(())
    }

    pub fn plan(&self) -> &FrequencyResponsePlan {
        &self.plan
    }

    /// Produces an independently executable, fully source-bound workload, without
    /// assembling an operator or invoking a numerical solve.
    pub fn entry(
        &self,
        entry: FrequencyResponsePlanEntry,
    ) -> AnalysisExecutionResult<AcousticHarmonicWorkload> {
        self.validate()?;
        self.plan.validate_entry(entry).map_err(invalid)?;
        let mut workload = self.template.clone();
        workload.frequency = Frequency::new(entry.frequency_hz());
        workload.validate()?;
        Ok(workload)
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            WORKLOAD_PREFIX,
            "acoustic-frequency-workload",
            self,
            LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            WORKLOAD_PREFIX,
            "acoustic-frequency-workload",
            bytes,
            LIMITS,
        )?)
    }

    pub fn identity(&self) -> AnalysisExecutionResult<AnalysisWorkloadIdentity> {
        AnalysisWorkloadIdentity::derive(
            "acoustic-frequency-workload",
            &Digest::sha256(self.canonical_bytes()?),
        )
    }
}

/// Complete response in exact shared-plan order, retaining every canonical entry's
/// fields and numerical closure evidence rather than selecting a representative frequency.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields, try_from = "AcousticFrequencyResultWire")]
pub struct AcousticFrequencyResult {
    workload: AcousticFrequencyWorkload,
    mesh_digest: StableDigest,
    results: Vec<AcousticHarmonicSolverResult>,
}

// The retained result and pre-assembly borrowed projection use one wire authority.
#[derive(Serialize)]
struct ResultSerialization<'a> {
    workload: &'a AcousticFrequencyWorkload,
    mesh_digest: StableDigest,
    results: &'a [AcousticHarmonicSolverResult],
}

impl Serialize for AcousticFrequencyResult {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        ResultSerialization {
            workload: &self.workload,
            mesh_digest: self.mesh_digest,
            results: &self.results,
        }
        .serialize(serializer)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AcousticFrequencyResultWire {
    workload: AcousticFrequencyWorkload,
    mesh_digest: StableDigest,
    results: Vec<AcousticHarmonicSolverResult>,
}

impl TryFrom<AcousticFrequencyResultWire> for AcousticFrequencyResult {
    type Error = AnalysisExecutionError;
    fn try_from(wire: AcousticFrequencyResultWire) -> AnalysisExecutionResult<Self> {
        let result = Self {
            workload: wire.workload,
            mesh_digest: wire.mesh_digest,
            results: wire.results,
        };
        result.validate()?;
        Ok(result)
    }
}

impl AcousticFrequencyResult {
    /// Admission sizing only: borrows originals without constructing a completed aggregate.
    /// Plan-order validation remains mandatory at the assembly boundary.
    pub fn assembly_encoding_profile(
        workload: &AcousticFrequencyWorkload,
        mesh: &SolverMeshArtifact,
        results: &[AcousticHarmonicSolverResult],
    ) -> AnalysisExecutionResult<runmat_canonical_codec::SerializedCanonicalProfile> {
        Ok(runmat_canonical_codec::profile_serialized(
            &ResultSerialization {
                workload,
                mesh_digest: mesh.canonical_digest,
                results,
            },
            RESULT_LIMITS,
        )?)
    }

    /// Pure fan-in: no operator assembly, solve, interpolation or field recovery.
    pub fn assemble(
        workload: AcousticFrequencyWorkload,
        mesh: &SolverMeshArtifact,
        results: Vec<AcousticHarmonicSolverResult>,
    ) -> AnalysisExecutionResult<Self> {
        workload.validate()?;
        let mut by_frequency = BTreeMap::new();
        for result in results {
            result.validate()?;
            if by_frequency
                .insert(result.frequency().canonical_value().to_bits(), result)
                .is_some()
            {
                return Err(invalid(
                    "acoustic frequency assembly contains duplicate entries",
                ));
            }
        }
        let results = workload
            .plan
            .entries()
            .map_err(invalid)?
            .into_iter()
            .map(|entry| {
                let result = by_frequency
                    .remove(&entry.frequency_hz().to_bits())
                    .ok_or_else(|| {
                        invalid("acoustic frequency assembly is missing a plan entry")
                    })?;
                result.validate_for(&workload.entry(entry)?, mesh)?;
                Ok(result)
            })
            .collect::<AnalysisExecutionResult<Vec<_>>>()?;
        if !by_frequency.is_empty() {
            return Err(invalid(
                "acoustic frequency assembly contains unrelated entries",
            ));
        }
        let result = Self {
            workload,
            mesh_digest: mesh.canonical_digest,
            results,
        };
        result.validate_for(&result.workload, mesh)?;
        Ok(result)
    }

    fn validate(&self) -> AnalysisExecutionResult<()> {
        self.workload.validate()?;
        let entries = self.workload.plan.entries().map_err(invalid)?;
        if self.results.len() != entries.len() {
            return Err(invalid("acoustic frequency result has an incomplete plan"));
        }
        for (entry, result) in entries.into_iter().zip(&self.results) {
            result.validate()?;
            let expected = self.workload.entry(entry)?;
            if result.frequency() != expected.frequency
                || result.source_workload() != expected.identity()?
                || result.producer() != expected.producer_identity()?
                || result.publication_declaration() != expected.publication_declaration()?
                || result.mesh_digest() != self.mesh_digest
                || result.ordered_node_ids() != self.results[0].ordered_node_ids()
                || result.ordered_element_ids() != self.results[0].ordered_element_ids()
                || result.formulation() != self.results[0].formulation()
            {
                return Err(invalid(
                    "acoustic frequency result differs from its exact plan/source/mesh",
                ));
            }
        }
        Ok(())
    }

    pub fn validate_for(
        &self,
        workload: &AcousticFrequencyWorkload,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        if &self.workload != workload || self.mesh_digest != mesh.canonical_digest {
            return Err(invalid(
                "acoustic frequency result belongs to a different workload or mesh",
            ));
        }
        for (entry, result) in workload
            .plan
            .entries()
            .map_err(invalid)?
            .into_iter()
            .zip(&self.results)
        {
            result.validate_for(&workload.entry(entry)?, mesh)?;
        }
        Ok(())
    }

    pub fn results(&self) -> &[AcousticHarmonicSolverResult] {
        &self.results
    }
    pub fn workload(&self) -> &AcousticFrequencyWorkload {
        &self.workload
    }
    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            RESULT_PREFIX,
            "acoustic-frequency-result",
            self,
            RESULT_LIMITS,
        )?)
    }
    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            RESULT_PREFIX,
            "acoustic-frequency-result",
            bytes,
            RESULT_LIMITS,
        )?)
    }
    pub fn source_result_identity(
        &self,
    ) -> AnalysisExecutionResult<super::super::SolverResultIdentity> {
        super::super::SolverResultIdentity::derive(
            "acoustic-frequency-result",
            &Digest::sha256(self.canonical_bytes()?),
        )
    }
}

#[cfg(test)]
mod tests;
