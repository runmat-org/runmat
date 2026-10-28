use std::num::NonZeroU32;

use runmat_analysis_core::{ObservationCostEstimate, ObservationId};
use runmat_canonical_codec::{decode_contract, encode_contract};
use runmat_execution::Digest;
use runmat_meshing_core::StableDigest;
use serde::{Deserialize, Serialize};

use crate::results::field::RecoveredFieldEntityDomain;

use super::{
    contract::{OBSERVATION_LIMITS, OBSERVATION_PREFIX},
    ObservationError, ResolvedObservationPlan, ResolvedObservationTarget,
};

pub const DEFAULT_MAXIMUM_OBSERVATION_PARTITIONS: u32 = 64;

/// Portable execution policy for observation work.
///
/// The limit belongs to the numerical workload so resource admission, task fan-out, retries, and
/// replay all derive the same canonical partition plan. A value of one is the explicit serial
/// policy; callers never select partitioning through an unbound scheduler hint.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationExecutionPolicy {
    maximum_partitions_per_observation: NonZeroU32,
}

impl ObservationExecutionPolicy {
    pub const fn canonical(maximum_partitions_per_observation: NonZeroU32) -> Self {
        Self {
            maximum_partitions_per_observation,
        }
    }

    pub const fn serial() -> Self {
        Self::canonical(NonZeroU32::MIN)
    }

    pub const fn scalable() -> Self {
        let Some(maximum) = NonZeroU32::new(DEFAULT_MAXIMUM_OBSERVATION_PARTITIONS) else {
            unreachable!()
        };
        Self::canonical(maximum)
    }

    pub const fn maximum_partitions_per_observation(self) -> NonZeroU32 {
        self.maximum_partitions_per_observation
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ObservationPartitionPlanIdentity(Digest);

impl ObservationPartitionPlanIdentity {
    pub const fn digest(self) -> Digest {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ObservationPartitionIdentity(Digest);

impl ObservationPartitionIdentity {
    pub const fn digest(self) -> Digest {
        self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum ObservationPartitionTarget {
    Nodes {
        indices: Vec<usize>,
        identities: Vec<StableDigest>,
    },
    Elements {
        domain: RecoveredFieldEntityDomain,
        indices: Vec<usize>,
        identities: Vec<StableDigest>,
    },
    Points {
        sample_ordinals: Vec<usize>,
        identities: Vec<StableDigest>,
    },
    Volume {
        element_indices: Vec<usize>,
        element_identities: Vec<StableDigest>,
    },
    Boundary {
        face_indices: Vec<usize>,
        face_identities: Vec<StableDigest>,
    },
    Connectors {
        indices: Vec<usize>,
        identities: Vec<StableDigest>,
    },
    ConstraintGroups {
        indices: Vec<usize>,
        identities: Vec<StableDigest>,
    },
}

impl ObservationPartitionTarget {
    pub fn entity_count(&self) -> usize {
        match self {
            Self::Nodes { indices, .. } | Self::Elements { indices, .. } => indices.len(),
            Self::Points {
                sample_ordinals, ..
            } => sample_ordinals.len(),
            Self::Volume {
                element_indices, ..
            } => element_indices.len(),
            Self::Boundary { face_indices, .. } => face_indices.len(),
            Self::Connectors { indices, .. } => indices.len(),
            Self::ConstraintGroups { indices, .. } => indices.len(),
        }
    }

    pub fn identities(&self) -> &[StableDigest] {
        match self {
            Self::Nodes { identities, .. }
            | Self::Elements { identities, .. }
            | Self::Points { identities, .. } => identities,
            Self::Volume {
                element_identities, ..
            } => element_identities,
            Self::Boundary {
                face_identities, ..
            } => face_identities,
            Self::Connectors { identities, .. } => identities,
            Self::ConstraintGroups { identities, .. } => identities,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationPartition {
    pub identity: ObservationPartitionIdentity,
    pub ordinal: u32,
    pub target: ObservationPartitionTarget,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "ObservationPartitionPlanWire")]
pub struct ObservationPartitionPlan {
    pub schema_version: u16,
    pub identity: ObservationPartitionPlanIdentity,
    pub observation_plan: super::ResolvedObservationPlanIdentity,
    pub topology_digest: Digest,
    pub observation_id: ObservationId,
    pub partitions: Vec<ObservationPartition>,
    pub resource_cost: ObservationCostEstimate,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservationPartitionPlanWire {
    schema_version: u16,
    identity: ObservationPartitionPlanIdentity,
    observation_plan: super::ResolvedObservationPlanIdentity,
    topology_digest: Digest,
    observation_id: ObservationId,
    partitions: Vec<ObservationPartition>,
    resource_cost: ObservationCostEstimate,
}

impl TryFrom<ObservationPartitionPlanWire> for ObservationPartitionPlan {
    type Error = ObservationError;

    fn try_from(wire: ObservationPartitionPlanWire) -> Result<Self, Self::Error> {
        let plan = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            observation_plan: wire.observation_plan,
            topology_digest: wire.topology_digest,
            observation_id: wire.observation_id,
            partitions: wire.partitions,
            resource_cost: wire.resource_cost,
        };
        plan.validate_structure()?;
        Ok(plan)
    }
}

impl ObservationPartitionPlan {
    pub fn derive(
        plan: &ResolvedObservationPlan,
        observation_id: &ObservationId,
        policy: ObservationExecutionPolicy,
    ) -> Result<Self, ObservationError> {
        plan.validate_identity()?;
        Self::derive_for_validated_plan(plan, observation_id, policy)
    }

    /// As `derive`, for a resolved plan whose identity the caller validated in the same
    /// operation (for example through `ObservationEvaluationRequest::validate`).
    pub(crate) fn derive_for_validated_plan(
        plan: &ResolvedObservationPlan,
        observation_id: &ObservationId,
        policy: ObservationExecutionPolicy,
    ) -> Result<Self, ObservationError> {
        let observation = plan
            .observations
            .iter()
            .find(|candidate| &candidate.spec.observation_id == observation_id)
            .ok_or_else(|| {
                ObservationError::Invalid(
                    "observation partition plan names an absent observation".into(),
                )
            })?;
        let entity_count = observation.target.selected_entity_count();
        let partition_count = usize::min(
            entity_count,
            policy.maximum_partitions_per_observation().get() as usize,
        );
        let mut partitions = Vec::with_capacity(partition_count);
        for ordinal in 0..partition_count {
            let start = ordinal * entity_count / partition_count;
            let end = (ordinal + 1) * entity_count / partition_count;
            let target = partition_target(&observation.target, start, end);
            let ordinal = u32::try_from(ordinal)
                .map_err(|_| ObservationError::Limit("observation partition ordinal"))?;
            let identity = derive_partition_identity(
                plan.identity,
                plan.topology_digest,
                observation_id,
                ordinal,
                &target,
            )?;
            partitions.push(ObservationPartition {
                identity,
                ordinal,
                target,
            });
        }
        let mut partition_plan = Self {
            schema_version: 1,
            identity: ObservationPartitionPlanIdentity(Digest::sha256([])),
            observation_plan: plan.identity,
            topology_digest: plan.topology_digest,
            observation_id: observation_id.clone(),
            partitions,
            resource_cost: partitioned_resource_cost(observation, partition_count, policy)?,
        };
        partition_plan.identity = partition_plan.derive_identity()?;
        // Identities were just derived from this content; check only the shape.
        partition_plan.validate_shape()?;
        Ok(partition_plan)
    }

    /// The canonical derivation validates the resolved plan identity once and owns every
    /// partition target, identity and cost; equality with it is the whole check, so neither
    /// the stored identities nor the resolved plan are hashed a second time.
    pub fn validate_for(&self, plan: &ResolvedObservationPlan) -> Result<(), ObservationError> {
        if self.observation_plan != plan.identity || self.topology_digest != plan.topology_digest {
            return Err(ObservationError::Identity(
                "observation partition plan names a different resolved plan",
            ));
        }
        let partitions = NonZeroU32::new(
            u32::try_from(self.partitions.len())
                .map_err(|_| ObservationError::Limit("observation partition count"))?,
        )
        .ok_or_else(|| ObservationError::Invalid("observation partition plan is empty".into()))?;
        let expected = ObservationPartitionPlan::derive(
            plan,
            &self.observation_id,
            ObservationExecutionPolicy::canonical(partitions),
        )?;
        if *self != expected {
            return Err(ObservationError::Identity(
                "observation partition ownership does not cover the canonical target exactly",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ObservationError> {
        self.validate_structure()?;
        Ok(encode_contract(
            OBSERVATION_PREFIX,
            "observation-partition-plan",
            self,
            OBSERVATION_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, ObservationError> {
        Ok(decode_contract(
            OBSERVATION_PREFIX,
            "observation-partition-plan",
            bytes,
            OBSERVATION_LIMITS,
        )?)
    }

    /// Metadata and per-partition shape, without hashing any identity.
    fn validate_shape(&self) -> Result<(), ObservationError> {
        if self.schema_version != 1
            || self.observation_id.0.trim().is_empty()
            || self.partitions.is_empty()
            || self.resource_cost.operation_count == 0
            || self.resource_cost.peak_resident_bytes == 0
            || self.resource_cost.stored_bytes == 0
            || self.resource_cost.transfer_bytes == 0
        {
            return Err(ObservationError::Invalid(
                "observation partition plan metadata or identity is invalid".into(),
            ));
        }
        for (ordinal, partition) in self.partitions.iter().enumerate() {
            if partition.ordinal as usize != ordinal
                || partition.target.entity_count() == 0
                || partition.target.entity_count() != partition.target.identities().len()
            {
                return Err(ObservationError::Invalid(
                    "observation partition metadata or identity is invalid".into(),
                ));
            }
        }
        Ok(())
    }

    fn validate_structure(&self) -> Result<(), ObservationError> {
        self.validate_shape()?;
        if self.identity != self.derive_identity()? {
            return Err(ObservationError::Invalid(
                "observation partition plan metadata or identity is invalid".into(),
            ));
        }
        for partition in &self.partitions {
            if partition.identity
                    != derive_partition_identity(
                        self.observation_plan,
                        self.topology_digest,
                        &self.observation_id,
                        partition.ordinal,
                        &partition.target,
                    )?
            {
                return Err(ObservationError::Invalid(
                    "observation partition metadata or identity is invalid".into(),
                ));
            }
        }
        Ok(())
    }

    fn derive_identity(&self) -> Result<ObservationPartitionPlanIdentity, ObservationError> {
        let bytes = encode_contract(
            OBSERVATION_PREFIX,
            "observation-partition-plan-identity",
            &(
                self.schema_version,
                self.observation_plan,
                self.topology_digest,
                &self.observation_id,
                &self.partitions,
                self.resource_cost,
            ),
            OBSERVATION_LIMITS,
        )?;
        Ok(ObservationPartitionPlanIdentity(Digest::sha256(bytes)))
    }
}

fn partitioned_resource_cost(
    observation: &super::ResolvedObservation,
    partition_count: usize,
    policy: ObservationExecutionPolicy,
) -> Result<ObservationCostEstimate, ObservationError> {
    let partitions = u64::try_from(partition_count)
        .map_err(|_| ObservationError::Limit("observation partition count"))?;
    let extra_partitions = partitions.saturating_sub(1);
    let produced_samples = u64::try_from(observation.sample_bounds.maximum_produced_samples)
        .map_err(|_| ObservationError::Limit("observation produced samples"))?;
    let stored_bytes = observation
        .resolved_cost
        .stored_bytes
        .checked_mul(partitions)
        .ok_or(ObservationError::Limit(
            "observation partition retained bytes",
        ))?;
    let transfer_bytes = observation
        .resolved_cost
        .peak_resident_bytes
        .checked_mul(extra_partitions)
        .and_then(|extra| observation.resolved_cost.transfer_bytes.checked_add(extra))
        .ok_or(ObservationError::Limit(
            "observation partition transfer bytes",
        ))?;
    let spatial_collectives = if observation
        .spec
        .reduction
        .as_ref()
        .is_some_and(|reduction| reduction.spatial != runmat_analysis_core::SpatialReduction::None)
    {
        extra_partitions
            .checked_mul(produced_samples)
            .ok_or(ObservationError::Limit(
                "observation partition collective count",
            ))?
    } else {
        0
    };
    let series_collectives = if observation
        .spec
        .reduction
        .as_ref()
        .is_some_and(|reduction| reduction.series != runmat_analysis_core::SeriesReduction::None)
    {
        produced_samples
            .min(u64::from(policy.maximum_partitions_per_observation().get()))
            .saturating_sub(1)
    } else {
        0
    };
    let collective_count =
        spatial_collectives
            .checked_add(series_collectives)
            .ok_or(ObservationError::Limit(
                "observation partition collective count",
            ))?;
    Ok(ObservationCostEstimate {
        operation_count: observation.resolved_cost.operation_count,
        peak_resident_bytes: observation.resolved_cost.peak_resident_bytes,
        stored_bytes,
        transfer_bytes,
        collective_count,
    })
}

fn partition_target(
    target: &ResolvedObservationTarget,
    start: usize,
    end: usize,
) -> ObservationPartitionTarget {
    match target {
        ResolvedObservationTarget::Nodes {
            indices,
            identities,
        } => ObservationPartitionTarget::Nodes {
            indices: indices[start..end].to_vec(),
            identities: identities[start..end].to_vec(),
        },
        ResolvedObservationTarget::Elements {
            domain,
            indices,
            identities,
        } => ObservationPartitionTarget::Elements {
            domain: *domain,
            indices: indices[start..end].to_vec(),
            identities: identities[start..end].to_vec(),
        },
        ResolvedObservationTarget::Points { samples } => ObservationPartitionTarget::Points {
            sample_ordinals: (start..end).collect(),
            identities: samples[start..end]
                .iter()
                .map(|sample| sample.identity)
                .collect(),
        },
        ResolvedObservationTarget::Volume {
            element_indices,
            element_identities,
        } => ObservationPartitionTarget::Volume {
            element_indices: element_indices[start..end].to_vec(),
            element_identities: element_identities[start..end].to_vec(),
        },
        ResolvedObservationTarget::Boundary {
            face_indices,
            face_identities,
        } => ObservationPartitionTarget::Boundary {
            face_indices: face_indices[start..end].to_vec(),
            face_identities: face_identities[start..end].to_vec(),
        },
        ResolvedObservationTarget::ConstraintGroups {
            indices,
            identities,
        } => ObservationPartitionTarget::ConstraintGroups {
            indices: indices[start..end].to_vec(),
            identities: identities[start..end].to_vec(),
        },
        ResolvedObservationTarget::Connectors {
            indices,
            identities,
        } => ObservationPartitionTarget::Connectors {
            indices: indices[start..end].to_vec(),
            identities: identities[start..end].to_vec(),
        },
    }
}

fn derive_partition_identity(
    plan: super::ResolvedObservationPlanIdentity,
    topology_digest: Digest,
    observation_id: &ObservationId,
    ordinal: u32,
    target: &ObservationPartitionTarget,
) -> Result<ObservationPartitionIdentity, ObservationError> {
    let bytes = encode_contract(
        OBSERVATION_PREFIX,
        "observation-partition-identity",
        &(plan, topology_digest, observation_id, ordinal, target),
        OBSERVATION_LIMITS,
    )?;
    Ok(ObservationPartitionIdentity(Digest::sha256(bytes)))
}
