use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::{
    validate_finite, validate_token, AlgorithmVersionSet, CanonicalMeshingContract,
    GeometryRevisionRef, MeshingContractError, MeshingResourceBudgetV2, MeshingStageV2,
    PersistentEntityId, StableDigest,
};

pub const MESHING_EVIDENCE_SCHEMA_VERSION: u16 = 2;
const MAX_STAGE_COUNTERS: usize = 256;
const MAX_STAGE_INVARIANTS: usize = 256;
const MAX_SIZING_EVIDENCE: usize = 65_536;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ErrorDistributionV2 {
    pub sample_count: u64,
    pub minimum: f64,
    pub mean: f64,
    pub percentile_95: f64,
    pub percentile_99: f64,
    pub maximum: f64,
    pub unit: String,
}

impl ErrorDistributionV2 {
    fn validate(&self) -> Result<(), MeshingContractError> {
        if self.sample_count == 0 {
            return Err(MeshingContractError::invalid(
                "error distribution",
                "sample count must be non-zero",
            ));
        }
        for value in [
            self.minimum,
            self.mean,
            self.percentile_95,
            self.percentile_99,
            self.maximum,
        ] {
            validate_finite("error distribution", value)?;
        }
        if self.minimum > self.mean
            || self.mean > self.maximum
            || self.minimum > self.percentile_95
            || self.percentile_95 > self.percentile_99
            || self.percentile_99 > self.maximum
        {
            return Err(MeshingContractError::invalid(
                "error distribution",
                "summary statistics are inconsistent",
            ));
        }
        validate_token("error distribution unit", &self.unit, 64)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InvariantEvidenceV2 {
    pub invariant_id: String,
    pub passed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measured: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub required: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
}

impl InvariantEvidenceV2 {
    fn validate(&self) -> Result<(), MeshingContractError> {
        validate_token("invariant id", &self.invariant_id, 256)?;
        if !self.passed {
            return Err(MeshingContractError::invalid(
                "stage invariant",
                "successful meshing evidence cannot contain a failed invariant",
            ));
        }
        if let Some(value) = self.measured {
            validate_finite("invariant measured value", value)?;
        }
        if let Some(value) = self.required {
            validate_finite("invariant required value", value)?;
        }
        if let Some(unit) = &self.unit {
            validate_token("invariant unit", unit, 64)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageEvidenceV2 {
    pub stage: MeshingStageV2,
    pub stage_result_digest: StableDigest,
    pub entity_counts: BTreeMap<String, u64>,
    pub invariants: Vec<InvariantEvidenceV2>,
    pub achieved_error_distributions: BTreeMap<String, ErrorDistributionV2>,
    pub completed_work: u64,
    pub estimated_work: u64,
    pub peak_memory_bytes: u64,
    pub elapsed_time_ms: u64,
    pub cancellation_checkpoints: u64,
}

impl StageEvidenceV2 {
    fn validate(&self) -> Result<(), MeshingContractError> {
        self.stage_result_digest
            .validate_nonzero("stage evidence result digest")?;
        if self.entity_counts.len() > MAX_STAGE_COUNTERS
            || self.invariants.is_empty()
            || self.invariants.len() > MAX_STAGE_INVARIANTS
            || self.achieved_error_distributions.len() > MAX_STAGE_COUNTERS
            || self.completed_work > self.estimated_work
        {
            return Err(MeshingContractError::invalid(
                "stage evidence",
                "counters, invariants, distributions, or work bounds are invalid",
            ));
        }
        validate_map_keys("stage entity count", &self.entity_counts)?;
        validate_map_keys(
            "stage error distribution",
            &self.achieved_error_distributions,
        )?;
        for invariant in &self.invariants {
            invariant.validate()?;
        }
        if !strictly_increasing_by(&self.invariants, |value| &value.invariant_id) {
            return Err(MeshingContractError::invalid(
                "stage invariants",
                "must be unique and canonically ordered",
            ));
        }
        for distribution in self.achieved_error_distributions.values() {
            distribution.validate()?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SizingResolutionEvidenceV2 {
    pub scope: PersistentEntityId,
    pub requested_size_m: f64,
    pub resolved_size_m: f64,
    pub achieved_maximum_size_m: f64,
    pub clipped_contribution_count: u32,
    pub rejected_contribution_count: u32,
}

impl SizingResolutionEvidenceV2 {
    fn validate(&self) -> Result<(), MeshingContractError> {
        self.scope.validate()?;
        for value in [
            self.requested_size_m,
            self.resolved_size_m,
            self.achieved_maximum_size_m,
        ] {
            validate_finite("sizing resolution evidence", value)?;
            if value <= 0.0 {
                return Err(MeshingContractError::invalid(
                    "sizing resolution evidence",
                    "sizes must be positive",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshingResourceUsageV2 {
    pub generated_nodes: u64,
    pub generated_elements: u64,
    pub peak_memory_bytes: u64,
    pub peak_scratch_bytes: u64,
    pub wall_time_ms: u64,
    pub artifact_bytes: u64,
    pub search_work: u64,
    pub maximum_recursion_depth: u32,
    pub iterations: u64,
}

impl MeshingResourceUsageV2 {
    fn validate_against(
        &self,
        budget: &MeshingResourceBudgetV2,
    ) -> Result<(), MeshingContractError> {
        if self.generated_nodes > budget.maximum_nodes
            || self.generated_elements > budget.maximum_elements
            || self.peak_memory_bytes > budget.maximum_memory_bytes
            || self.peak_scratch_bytes > budget.maximum_scratch_bytes
            || self.wall_time_ms > budget.maximum_wall_time_ms
            || self.artifact_bytes > budget.maximum_artifact_bytes
            || self.search_work > budget.maximum_search_work
            || self.maximum_recursion_depth > budget.maximum_recursion_depth
            || self.iterations > budget.maximum_iterations
        {
            return Err(MeshingContractError::invalid(
                "meshing resource usage",
                "successful evidence exceeds a resolved hard budget",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlatformBuildIdentityV2 {
    pub capability_cohort: String,
    pub target_triple: String,
    pub build_digest: StableDigest,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exact_kernel_abi: Option<String>,
}

impl PlatformBuildIdentityV2 {
    fn validate(&self) -> Result<(), MeshingContractError> {
        validate_token("capability cohort", &self.capability_cohort, 128)?;
        validate_token("target triple", &self.target_triple, 128)?;
        self.build_digest
            .validate_nonzero("platform build digest")?;
        if let Some(abi) = &self.exact_kernel_abi {
            validate_token("exact kernel ABI", abi, 128)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheAdmissionDecisionV2 {
    Admitted,
    RejectedNonportableCapability,
    RejectedPolicy,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshingEvidenceV2 {
    pub schema_version: u16,
    pub geometry: GeometryRevisionRef,
    pub resolved_request_digest: StableDigest,
    pub artifact_digest: StableDigest,
    pub algorithms: AlgorithmVersionSet,
    pub deterministic_seed: u64,
    pub platform: PlatformBuildIdentityV2,
    pub stages: Vec<StageEvidenceV2>,
    pub sizing: Vec<SizingResolutionEvidenceV2>,
    pub resources: MeshingResourceUsageV2,
    pub cache_admission: CacheAdmissionDecisionV2,
}

impl MeshingEvidenceV2 {
    pub fn validate(
        &self,
        artifact: &super::AnalysisMeshArtifactV2,
    ) -> Result<(), MeshingContractError> {
        self.validate_standalone()?;
        artifact.validate()?;
        if self.geometry != artifact.geometry
            || self.resolved_request_digest != artifact.resolved_request.canonical_digest()?
            || self.artifact_digest != artifact.canonical_digest
            || self.algorithms != artifact.resolved_request.algorithms
            || self.deterministic_seed != artifact.resolved_request.deterministic_seed
        {
            return Err(MeshingContractError::invalid(
                "meshing evidence",
                "geometry, artifact, algorithm, or seed identity does not match the artifact",
            ));
        }
        if self.resources.generated_nodes != artifact.topology.nodes.len() as u64
            || self.resources.generated_elements != artifact.topology.volume_elements.len() as u64
        {
            return Err(MeshingContractError::invalid(
                "meshing resource usage",
                "generated entity counts must match the validated artifact",
            ));
        }
        self.resources
            .validate_against(&artifact.resolved_request.resources)
    }

    pub(super) fn validate_standalone(&self) -> Result<(), MeshingContractError> {
        if self.schema_version != MESHING_EVIDENCE_SCHEMA_VERSION {
            return Err(MeshingContractError::invalid(
                "meshing evidence schema version",
                format!("expected {MESHING_EVIDENCE_SCHEMA_VERSION}"),
            ));
        }
        self.geometry.validate()?;
        self.resolved_request_digest
            .validate_nonzero("evidence resolved request digest")?;
        self.artifact_digest
            .validate_nonzero("evidence artifact digest")?;
        self.algorithms.validate()?;
        self.platform.validate()?;
        if self.stages.len() != MeshingStageV2::ALL.len()
            || self
                .stages
                .iter()
                .map(|value| value.stage)
                .ne(MeshingStageV2::ALL)
            || self.sizing.len() > MAX_SIZING_EVIDENCE
            || !strictly_increasing_by(&self.sizing, |value| &value.scope)
        {
            return Err(MeshingContractError::invalid(
                "meshing evidence",
                "stage and sizing evidence must be bounded, unique, and canonically ordered",
            ));
        }
        for stage in &self.stages {
            stage.validate()?;
            if stage.peak_memory_bytes > self.resources.peak_memory_bytes
                || stage.elapsed_time_ms > self.resources.wall_time_ms
            {
                return Err(MeshingContractError::invalid(
                    "stage evidence",
                    "stage resource measurements exceed the aggregate resource evidence",
                ));
            }
        }
        for sizing in &self.sizing {
            sizing.validate()?;
        }
        Ok(())
    }
}

fn validate_map_keys<T>(
    field: &str,
    values: &BTreeMap<String, T>,
) -> Result<(), MeshingContractError> {
    for key in values.keys() {
        validate_token(field, key, 256)?;
    }
    Ok(())
}

fn strictly_increasing_by<T, K: Ord>(values: &[T], key: impl Fn(&T) -> &K) -> bool {
    values.windows(2).all(|pair| key(&pair[0]) < key(&pair[1]))
}
