use std::collections::BTreeSet;

use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::{
    identity::{AttemptId, WorkerId},
    resource::{ResourceAccounting, ResourceAccountingSummary, ResourceUsage},
    Digest,
};
use serde::{Deserialize, Serialize};

use super::{
    admission::allocation_matches_plan,
    contract::{ByteCount, FeaResourcePlan, FeaResourcePlanIdentity, ResourcePlanningError},
};

const RESOURCE_USAGE_PREFIX: &[u8] = b"runmat-fea-resource-usage/v1\0";
pub(super) const RESOURCE_USAGE_LIMITS: CanonicalLimits =
    CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 65_536, 64);

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// Scheduler allocation accounting, not sampled process or device utilization.
pub struct AnalysisResourceUsage {
    pub scheduler: ResourceUsage,
    pub peak_cpu_millicores: u32,
    pub peak_memory_bytes: ByteCount,
    pub peak_scratch_bytes: ByteCount,
    pub artifact_bytes: ByteCount,
}

impl From<ResourceAccountingSummary> for AnalysisResourceUsage {
    fn from(summary: ResourceAccountingSummary) -> Self {
        Self {
            scheduler: summary.usage,
            peak_cpu_millicores: summary.peak_cpu_millicores,
            peak_memory_bytes: ByteCount(summary.peak_memory_bytes),
            peak_scratch_bytes: ByteCount(summary.peak_scratch_bytes),
            artifact_bytes: ByteCount(summary.artifact_bytes),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceOverrun {
    CpuPeak {
        planned: u32,
        actual: u32,
    },
    CpuTime {
        planned_core_millis: u64,
        actual_core_millis: u64,
    },
    Memory {
        planned: u64,
        actual: u64,
    },
    MemoryTime {
        planned_byte_millis: u128,
        actual_byte_millis: u128,
    },
    Scratch {
        planned: u64,
        actual: u64,
    },
    Artifact {
        planned: u64,
        actual: u64,
    },
    Egress {
        planned: u64,
        actual: u64,
    },
    Relay {
        planned: u64,
        actual: u64,
    },
    WallTime {
        planned: u64,
        actual: u64,
    },
    RetainedTime {
        planned_byte_millis: u128,
        actual_byte_millis: u128,
    },
    AcceleratorTime {
        class: String,
        planned_device_millis: u64,
        actual_device_millis: u64,
    },
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ResourceUsageEvidenceIdentity(Digest);

impl ResourceUsageEvidenceIdentity {
    pub const fn digest(self) -> Digest {
        self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceUsageAttempt {
    pub attempt_id: AttemptId,
    pub worker_id: WorkerId,
    /// Exact, independently rederived plan that governed this scheduler attempt.
    pub plan: FeaResourcePlanIdentity,
    pub resources: ResourceAccounting,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceUsageAttemptOverrun {
    pub attempt_id: AttemptId,
    pub overrun: ResourceOverrun,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "ResourceUsageEvidenceWire")]
pub struct ResourceUsageEvidence {
    pub schema_version: u16,
    identity: ResourceUsageEvidenceIdentity,
    /// Root solve plan used to bind usage to the published numerical result.
    pub primary_plan: FeaResourcePlanIdentity,
    /// Canonically ordered, self-validating plans for every measured graph phase.
    pub plans: Vec<FeaResourcePlan>,
    pub workload_digest: Digest,
    pub result_digest: Digest,
    pub attempts: Vec<ResourceUsageAttempt>,
    /// Aggregated reserved allocations and observed transfer/artifact amounts; memory peaks are
    /// reservation peaks, not measured worker RSS.
    pub actual: AnalysisResourceUsage,
    pub overruns: Vec<ResourceUsageAttemptOverrun>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResourceUsageEvidenceWire {
    schema_version: u16,
    identity: ResourceUsageEvidenceIdentity,
    primary_plan: FeaResourcePlanIdentity,
    plans: Vec<FeaResourcePlan>,
    workload_digest: Digest,
    result_digest: Digest,
    attempts: Vec<ResourceUsageAttempt>,
    actual: AnalysisResourceUsage,
    overruns: Vec<ResourceUsageAttemptOverrun>,
}

impl TryFrom<ResourceUsageEvidenceWire> for ResourceUsageEvidence {
    type Error = ResourcePlanningError;

    fn try_from(wire: ResourceUsageEvidenceWire) -> Result<Self, Self::Error> {
        let actual = wire.actual;
        let expected = Self::build(
            wire.primary_plan,
            wire.plans,
            wire.workload_digest,
            wire.result_digest,
            wire.attempts,
            actual.scheduler.wall_millis,
        )?;
        if wire.schema_version != expected.schema_version
            || wire.identity != expected.identity
            || actual != expected.actual
            || wire.overruns != expected.overruns
        {
            return Err(ResourcePlanningError::Identity);
        }
        Ok(expected)
    }
}

impl ResourceUsageEvidence {
    /// One source-plan authority: the exact primary plan inside validated usage evidence.
    pub fn primary_plan(&self) -> Result<&FeaResourcePlan, ResourcePlanningError> {
        self.validate_identity()?;
        self.plans
            .binary_search_by_key(&self.primary_plan, |plan| plan.identity)
            .map(|index| &self.plans[index])
            .map_err(|_| ResourcePlanningError::Identity)
    }

    pub(crate) fn build(
        primary_plan: FeaResourcePlanIdentity,
        mut plans: Vec<FeaResourcePlan>,
        workload_digest: Digest,
        result_digest: Digest,
        mut attempts: Vec<ResourceUsageAttempt>,
        graph_wall_millis: u64,
    ) -> Result<Self, ResourcePlanningError> {
        if workload_digest == Digest::sha256([]) || result_digest == Digest::sha256([]) {
            return Err(ResourcePlanningError::Invalid(
                "resource usage must bind nonempty sources".into(),
            ));
        }
        plans.sort_by_key(|plan| plan.identity);
        if plans.is_empty()
            || plans
                .windows(2)
                .any(|pair| pair[0].identity == pair[1].identity)
        {
            return Err(ResourcePlanningError::Invalid(
                "resource usage requires distinct resource plans".into(),
            ));
        }
        for plan in &plans {
            plan.validate_identity()?;
        }
        if plans
            .binary_search_by_key(&primary_plan, |plan| plan.identity)
            .is_err()
        {
            return Err(ResourcePlanningError::Invalid(
                "resource usage primary plan is absent from its plan catalog".into(),
            ));
        }
        attempts.sort_by_key(|attempt| attempt.attempt_id);
        if attempts.is_empty()
            || attempts
                .windows(2)
                .any(|pair| pair[0].attempt_id == pair[1].attempt_id)
        {
            return Err(ResourcePlanningError::Invalid(
                "resource usage requires distinct measured attempts".into(),
            ));
        }
        let mut overruns = Vec::new();
        for attempt in &attempts {
            attempt
                .resources
                .validate()
                .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
            let plan = plans
                .binary_search_by_key(&attempt.plan, |plan| plan.identity)
                .ok()
                .map(|index| &plans[index])
                .ok_or_else(|| {
                    ResourcePlanningError::Invalid(
                        "resource usage attempt refers to an absent resource plan".into(),
                    )
                })?;
            if !allocation_matches_plan(&attempt.resources.allocation, &plan.request) {
                return Err(ResourcePlanningError::Invalid(
                    "resource usage attempt allocation differs from its plan".into(),
                ));
            }
            let attempt_summary = ResourceAccountingSummary::aggregate(
                std::slice::from_ref(&attempt.resources),
                attempt.resources.usage.wall_millis,
            )
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
            let attempt_actual = AnalysisResourceUsage::from(attempt_summary);
            overruns.extend(
                super::reconciliation::reconcile_resource_request(&plan.request, &attempt_actual)?
                    .into_iter()
                    .map(|overrun| ResourceUsageAttemptOverrun {
                        attempt_id: attempt.attempt_id,
                        overrun,
                    }),
            );
        }
        let referenced_plans = attempts
            .iter()
            .map(|attempt| attempt.plan)
            .collect::<BTreeSet<_>>();
        if referenced_plans.len() != plans.len()
            || plans
                .iter()
                .any(|plan| !referenced_plans.contains(&plan.identity))
        {
            return Err(ResourcePlanningError::Invalid(
                "resource usage plan catalog must exactly cover measured attempts".into(),
            ));
        }
        let intervals = attempts
            .iter()
            .map(|attempt| attempt.resources.clone())
            .collect::<Vec<_>>();
        let summary = ResourceAccountingSummary::aggregate(&intervals, graph_wall_millis)
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        let actual = AnalysisResourceUsage::from(summary);
        let identity = ResourceUsageEvidenceIdentity(Digest::sha256(encode_contract(
            RESOURCE_USAGE_PREFIX,
            "resource-usage-identity",
            &(
                primary_plan,
                &plans,
                workload_digest,
                result_digest,
                &attempts,
                &actual,
                &overruns,
            ),
            RESOURCE_USAGE_LIMITS,
        )?));
        Ok(Self {
            schema_version: 1,
            identity,
            primary_plan,
            plans,
            workload_digest,
            result_digest,
            attempts,
            actual,
            overruns,
        })
    }

    pub const fn identity(&self) -> ResourceUsageEvidenceIdentity {
        self.identity
    }

    pub fn validate_identity(&self) -> Result<(), ResourcePlanningError> {
        let expected = Self::build(
            self.primary_plan,
            self.plans.clone(),
            self.workload_digest,
            self.result_digest,
            self.attempts.clone(),
            self.actual.scheduler.wall_millis,
        )?;
        if self != &expected {
            return Err(ResourcePlanningError::Identity);
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ResourcePlanningError> {
        self.validate_identity()?;
        Ok(encode_contract(
            RESOURCE_USAGE_PREFIX,
            "resource-usage-evidence",
            self,
            RESOURCE_USAGE_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, ResourcePlanningError> {
        Ok(decode_contract(
            RESOURCE_USAGE_PREFIX,
            "resource-usage-evidence",
            bytes,
            RESOURCE_USAGE_LIMITS,
        )?)
    }
}
