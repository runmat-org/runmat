mod observation;
use runmat_canonical_codec::CanonicalCodecControl;
use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::{
    validate_finite, validate_token, AlgorithmVersionSet, CanonicalMeshingContract,
    GeometryRevisionRef, MeshingContractError, MeshingPartitionDescriptor, MeshingRequest,
    MeshingResourceBudget, MeshingStageKind, PersistentEntityId, StableDigest,
};

pub const MESHING_EVIDENCE_SCHEMA_VERSION: u16 = 5;
const MAX_STAGE_COUNTERS: usize = 256;
const MAX_STAGE_INVARIANTS: usize = 256;
const MAX_STAGE_EVIDENCE: usize = 65_536;
const MAX_SIZING_EVIDENCE: usize = 65_536;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ErrorDistribution {
    pub sample_count: u64,
    pub minimum: f64,
    pub mean: f64,
    pub percentile_95: f64,
    pub percentile_99: f64,
    pub maximum: f64,
    pub unit: String,
}

impl ErrorDistribution {
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
pub struct InvariantEvidence {
    pub invariant_id: String,
    pub passed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measured: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub required: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
}

impl InvariantEvidence {
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

/// Provenance of original producing-stage observations imported from accepted
/// artifacts. This is retained lineage, not an independently replayed proof.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ReusedMeshingStageOrigin {
    ImplementationCache {
        implementation_digest: StableDigest,
    },
    AcceptedInvocation {
        invocation_identity_digest: StableDigest,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReusedMeshingStageObservation {
    pub original_observation_digest: StableDigest,
    pub origin: ReusedMeshingStageOrigin,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshingStageEvidence {
    /// None denotes the original producer; cached observations retain original
    /// counters and timings and explicitly bind their original evidence identity.
    /// Omission preserves canonical bytes of existing fresh observations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reused_from: Option<ReusedMeshingStageObservation>,
    pub stage: MeshingStageKind,
    pub partition: MeshingPartitionDescriptor,
    pub stage_result_digest: StableDigest,
    pub entity_counts: BTreeMap<String, u64>,
    pub invariants: Vec<InvariantEvidence>,
    pub achieved_error_distributions: BTreeMap<String, ErrorDistribution>,
    pub completed_work: u64,
    pub estimated_work: u64,
    pub peak_memory_bytes: u64,
    pub peak_scratch_bytes: u64,
    pub search_work: u64,
    pub maximum_recursion_depth: u32,
    pub iterations: u64,
    pub elapsed_time_ms: u64,
    pub cancellation_checkpoints: u64,
}

impl MeshingStageEvidence {
    pub fn validate(&self) -> Result<(), MeshingContractError> {
        let control = super::MeshingCanonicalCodecControl::new(&super::NeverCancelled);
        self.validate_with_control(&control)
    }
    fn validate_with_control(
        &self,
        control: &dyn CanonicalCodecControl,
    ) -> Result<(), MeshingContractError> {
        observation::checkpoint(control)?;
        if let Some(origin) = &self.reused_from {
            origin
                .original_observation_digest
                .validate_nonzero("original observation digest")?;
            match &origin.origin {
                ReusedMeshingStageOrigin::ImplementationCache {
                    implementation_digest,
                } => {
                    implementation_digest.validate_nonzero("reused implementation digest")?;
                }
                ReusedMeshingStageOrigin::AcceptedInvocation {
                    invocation_identity_digest,
                } => {
                    invocation_identity_digest
                        .validate_nonzero("reused invocation identity digest")?;
                }
            }
        }
        self.partition.validate()?;
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
        if let Some(origin) = &self.reused_from {
            if observation::original_digest(self, control)? != origin.original_observation_digest {
                return Err(MeshingContractError::invalid(
                    "reused stage observation",
                    "retained values differ from the original observation identity",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SizingResolutionEvidence {
    pub scope: PersistentEntityId,
    pub requested_size_m: f64,
    pub resolved_size_m: f64,
    pub achieved_maximum_size_m: f64,
    pub clipped_contribution_count: u32,
    pub rejected_contribution_count: u32,
    /// Authored size control that requested this scope's size.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub control_id: Option<String>,
    /// Achieved corner edge lengths of the mesh entities on the scope.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub achieved: Option<AchievedEdgeLengths>,
}

/// Corner edge-length statistics of the mesh entities on one sizing scope.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AchievedEdgeLengths {
    pub edge_count: u64,
    pub minimum_m: f64,
    pub mean_m: f64,
    pub maximum_m: f64,
}

impl SizingResolutionEvidence {
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
pub struct MeshingResourceUsage {
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

impl MeshingResourceUsage {
    fn validate_against(&self, budget: &MeshingResourceBudget) -> Result<(), MeshingContractError> {
        let exceeded = [
            ("nodes", self.generated_nodes, budget.maximum_nodes),
            ("elements", self.generated_elements, budget.maximum_elements),
            (
                "memory_bytes",
                self.peak_memory_bytes,
                budget.maximum_memory_bytes,
            ),
            (
                "scratch_bytes",
                self.peak_scratch_bytes,
                budget.maximum_scratch_bytes,
            ),
            (
                "wall_time_ms",
                self.wall_time_ms,
                budget.maximum_wall_time_ms,
            ),
            (
                "artifact_bytes",
                self.artifact_bytes,
                budget.maximum_artifact_bytes,
            ),
            ("search_work", self.search_work, budget.maximum_search_work),
            (
                "recursion_depth",
                u64::from(self.maximum_recursion_depth),
                u64::from(budget.maximum_recursion_depth),
            ),
            ("iterations", self.iterations, budget.maximum_iterations),
        ]
        .into_iter()
        .filter(|(_, used, limit)| used > limit)
        .map(|(name, used, limit)| format!("{name} {used} > {limit}"))
        .collect::<Vec<_>>();
        if !exceeded.is_empty() {
            return Err(MeshingContractError::invalid(
                "meshing resource usage",
                format!(
                    "successful evidence exceeds a resolved hard budget: {}",
                    exceeded.join(", ")
                ),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlatformBuildIdentity {
    pub capability_cohort: String,
    pub target_triple: String,
    pub build_digest: StableDigest,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exact_kernel_abi: Option<String>,
}

impl PlatformBuildIdentity {
    pub fn validate(&self) -> Result<(), MeshingContractError> {
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
pub enum CacheAdmissionDecision {
    Admitted,
    RejectedNonportableCapability,
    RejectedPolicy,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshingEvidence {
    pub schema_version: u16,
    pub geometry: GeometryRevisionRef,
    pub resolved_request_digest: StableDigest,
    pub artifact_digest: StableDigest,
    pub algorithms: AlgorithmVersionSet,
    pub deterministic_seed: u64,
    pub platform: PlatformBuildIdentity,
    pub stages: Vec<MeshingStageEvidence>,
    pub sizing: Vec<SizingResolutionEvidence>,
    pub resources: MeshingResourceUsage,
    /// The execution ceilings the run was admitted under; with the request's size limits they
    /// form the hard budget the usage is checked against.
    pub ceilings: super::MeshingExecutionCeilings,
    pub cache_admission: CacheAdmissionDecision,
}

impl MeshingEvidence {
    pub fn validate(
        &self,
        artifact: &super::SolverMeshArtifact,
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
            || self.resources.generated_elements != artifact.generated_element_count()
            || self.resources.artifact_bytes != artifact.canonical_encode()?.len() as u64
        {
            return Err(MeshingContractError::invalid(
                "meshing resource usage",
                "generated entity counts and canonical bytes must match the validated artifact",
            ));
        }
        self.resources.validate_against(&MeshingResourceBudget::new(
            artifact.resolved_request.limits,
            self.ceilings,
        ))
    }

    /// Validates the terminal evidence against the identities and byte inventory sealed by the
    /// serialization stage. Serialization owns full artifact validation; publication verifies
    /// that the independently supplied evidence closes over that validated result without
    /// reconstructing the complete solver topology a second time.
    pub fn validate_publication_closure(
        &self,
        geometry: &GeometryRevisionRef,
        resolved_request: &MeshingRequest,
        artifact_digest: StableDigest,
        artifact_bytes: u64,
    ) -> Result<(), MeshingContractError> {
        self.validate_standalone()?;
        if self.geometry != *geometry
            || self.resolved_request_digest != resolved_request.canonical_digest()?
            || self.artifact_digest != artifact_digest
            || self.algorithms != resolved_request.algorithms
            || self.deterministic_seed != resolved_request.deterministic_seed
        {
            return Err(MeshingContractError::invalid(
                "meshing publication evidence",
                "geometry, request, artifact, algorithm, or seed identity does not close over the serialized artifact",
            ));
        }
        if self.resources.artifact_bytes != artifact_bytes {
            return Err(MeshingContractError::invalid(
                "meshing publication evidence",
                "artifact byte inventory does not match the serialized artifact",
            ));
        }
        self.resources.validate_against(&MeshingResourceBudget::new(
            resolved_request.limits,
            self.ceilings,
        ))
    }

    pub(super) fn validate_standalone(&self) -> Result<(), MeshingContractError> {
        let control = super::MeshingCanonicalCodecControl::new(&super::NeverCancelled);
        self.validate_standalone_with_control(&control)
    }
    fn validate_standalone_with_control(
        &self,
        control: &dyn CanonicalCodecControl,
    ) -> Result<(), MeshingContractError> {
        observation::checkpoint(control)?;
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
        if self.stages.is_empty()
            || self.stages.len() > MAX_STAGE_EVIDENCE
            || self.stages.last().map(|value| value.stage) != Some(MeshingStageKind::Serialization)
            || !self.stages.windows(2).all(|pair| {
                (
                    pair[0].stage,
                    &pair[0].partition,
                    pair[0].stage_result_digest,
                ) < (
                    pair[1].stage,
                    &pair[1].partition,
                    pair[1].stage_result_digest,
                )
            })
            || self.sizing.len() > MAX_SIZING_EVIDENCE
            || !strictly_increasing_by(&self.sizing, |value| &value.scope)
        {
            return Err(MeshingContractError::invalid(
                "meshing evidence",
                "stage and sizing evidence must be bounded, unique, and canonically ordered",
            ));
        }
        for stage in &self.stages {
            observation::checkpoint(control)?;
            stage.validate_with_control(control)?;
            if stage.peak_memory_bytes > self.resources.peak_memory_bytes
                || stage.peak_scratch_bytes > self.resources.peak_scratch_bytes
                || stage.search_work > self.resources.search_work
                || stage.maximum_recursion_depth > self.resources.maximum_recursion_depth
                || stage.iterations > self.resources.iterations
                // A reused observation retains the timing of the run that produced it,
                // which may exceed the wall time of the run that restored it.
                || (stage.reused_from.is_none()
                    && stage.elapsed_time_ms > self.resources.wall_time_ms)
            {
                return Err(MeshingContractError::invalid(
                    "stage evidence",
                    "stage resource measurements exceed the aggregate resource evidence",
                ));
            }
        }
        let expected_peak_memory = self
            .stages
            .iter()
            .map(|stage| stage.peak_memory_bytes)
            .max()
            .unwrap_or(0);
        let expected_peak_scratch = self
            .stages
            .iter()
            .map(|stage| stage.peak_scratch_bytes)
            .max()
            .unwrap_or(0);
        let expected_recursion = self
            .stages
            .iter()
            .map(|stage| stage.maximum_recursion_depth)
            .max()
            .unwrap_or(0);
        let expected_search_work = checked_stage_sum(&self.stages, |stage| stage.search_work)?;
        let expected_iterations = checked_stage_sum(&self.stages, |stage| stage.iterations)?;
        if self.resources.peak_memory_bytes != expected_peak_memory
            || self.resources.peak_scratch_bytes != expected_peak_scratch
            || self.resources.search_work != expected_search_work
            || self.resources.maximum_recursion_depth != expected_recursion
            || self.resources.iterations != expected_iterations
        {
            return Err(MeshingContractError::invalid(
                "meshing resource usage",
                "aggregate stage resources must equal their factual maxima and checked totals",
            ));
        }
        for sizing in &self.sizing {
            observation::checkpoint(control)?;
            sizing.validate()?;
        }
        Ok(())
    }
}

fn checked_stage_sum(
    stages: &[MeshingStageEvidence],
    value: impl Fn(&MeshingStageEvidence) -> u64,
) -> Result<u64, MeshingContractError> {
    stages.iter().try_fold(0_u64, |total, stage| {
        total.checked_add(value(stage)).ok_or_else(|| {
            MeshingContractError::invalid(
                "meshing resource usage",
                "aggregate stage resource counter overflowed",
            )
        })
    })
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

impl MeshingEvidence {
    /// Admit actual typed CBOR materialization and the largest original-observation
    /// codec workspace before validating the complete evidence. No stage clone.
    pub fn canonical_decode_with_control(
        bytes: &[u8],
        control: &dyn CanonicalCodecControl,
        admit: impl FnMut(u64) -> Result<(), MeshingContractError>,
    ) -> Result<Self, MeshingContractError> {
        super::codec::decode_validated(
            bytes,
            control,
            admit,
            |evidence: &Self| observation::validation_workspace(evidence, control),
            |evidence| evidence.validate_standalone_with_control(control),
        )
    }
}
