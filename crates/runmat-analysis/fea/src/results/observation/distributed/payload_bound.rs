//! Canonical size ceiling of the contribution an observation partition writes, derived before
//! the partition is evaluated.
//!
//! The ceiling encodes real contributions of this partition with one and two widest payload
//! items and extends the per-item increment to the admitted item count. Identity metadata
//! (plan, partition, source field, source partitions, sample, event attribution) is encoded
//! exactly as the worker writes it. The partition's inputs (mesh, workload, field values) never
//! reach the contribution and take no part in the ceiling.
use super::*;

/// Most a contribution payload can hold, by payload kind.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ObservationPartitionPayloadCapacity {
    /// Selected values or extrema candidates: at most `entity_count` entities of the partition's
    /// target kind, each with at most `values_per_entity` values.
    Entities {
        extrema: bool,
        entity_count: usize,
        values_per_entity: usize,
    },
    /// A compensated weighted reduction with sums of at most `sum_length` values, or a paired
    /// boundary reduction (four scalars).
    Weighted { sum_length: usize },
}

const WIDEST_STABLE_DIGEST: runmat_meshing_core::StableDigest =
    runmat_meshing_core::StableDigest::from_bytes([u8::MAX; 32]);

// Header growth of one canonical array from its one-byte form to its widest form.
const ARRAY_HEADER_GROWTH_BYTES: u64 = 8;

impl ObservationPartitionContribution {
    pub fn encoded_bytes_upper_bound(
        partition_plan: &ObservationPartitionPlan,
        partition: &ObservationPartition,
        field: &RecoveredFieldPartitionSet,
        event_attribution: Option<ObservationEventAttribution>,
        capacity: ObservationPartitionPayloadCapacity,
    ) -> Result<u64, ObservationError> {
        let encoded = |payload| -> Result<u64, ObservationError> {
            let contribution = Self::new(
                partition_plan,
                partition,
                field,
                event_attribution,
                None,
                payload,
            )?;
            u64::try_from(contribution.canonical_bytes()?.len())
                .map_err(|_| ObservationError::Limit("observation contribution bytes"))
        };
        let overflow = || ObservationError::Limit("observation contribution ceiling");
        match capacity {
            ObservationPartitionPayloadCapacity::Entities {
                extrema,
                entity_count,
                values_per_entity,
            } => {
                if entity_count == 0 || values_per_entity == 0 {
                    return Err(ObservationError::Invalid(
                        "observation contribution capacity must be nonzero".into(),
                    ));
                }
                let entity = widest_entity(&partition.target)?;
                let payload = |items: usize| {
                    let entities = vec![entity.clone(); items];
                    let values = vec![f64::MAX; items * values_per_entity];
                    if extrema {
                        ObservationPartitionPayload::ExtremaCandidates { entities, values }
                    } else {
                        ObservationPartitionPayload::SelectedValues { entities, values }
                    }
                };
                let one = encoded(payload(1))?;
                let two = encoded(payload(2))?;
                let increment = two.checked_sub(one).ok_or_else(overflow)?;
                u64::try_from(entity_count - 1)
                    .ok()
                    .and_then(|items| items.checked_mul(increment))
                    .and_then(|bytes| bytes.checked_add(one))
                    .and_then(|bytes| bytes.checked_add(2 * ARRAY_HEADER_GROWTH_BYTES))
                    .ok_or_else(overflow)
            }
            ObservationPartitionPayloadCapacity::Weighted { sum_length } => {
                if sum_length == 0 {
                    return Err(ObservationError::Invalid(
                        "observation contribution capacity must be nonzero".into(),
                    ));
                }
                let weighted = encoded(ObservationPartitionPayload::WeightedReduction {
                    linear_sum: vec![f64::MAX; sum_length],
                    square_sum: vec![f64::MAX; sum_length],
                    measure: f64::MAX,
                })?;
                let paired = encoded(ObservationPartitionPayload::PairedWeightedReduction {
                    first_sum: f64::MAX,
                    first_measure: f64::MAX,
                    second_sum: f64::MAX,
                    second_measure: f64::MAX,
                })?;
                Ok(weighted.max(paired))
            }
        }
    }
}

/// The partition's entity kind with every variable-width scalar at its widest encoding.
fn widest_entity(
    target: &ObservationPartitionTarget,
) -> Result<ObservationEntity, ObservationError> {
    if target.identities().is_empty() {
        return Err(ObservationError::Identity(
            "observation partition owns no entities",
        ));
    }
    // A stable digest encodes as an array of 32 byte integers; a byte of 24 or more takes two
    // bytes, so the all-ones digest is the widest identity any entity can carry.
    let identity = WIDEST_STABLE_DIGEST;
    Ok(match target {
        ObservationPartitionTarget::Nodes { .. } => ObservationEntity::Node(identity),
        ObservationPartitionTarget::Elements { domain, .. } => {
            super::super::contract::observation_element_entity(*domain, identity)
        }
        ObservationPartitionTarget::Points { .. } => ObservationEntity::Point {
            identity,
            ordinal: runmat_types::wire_integer::MAX_WIRE_INTEGER,
            position_m: [f64::MAX; 3],
            source_element: identity,
        },
        ObservationPartitionTarget::Connectors { .. } => ObservationEntity::Connector(identity),
        ObservationPartitionTarget::Volume { .. }
        | ObservationPartitionTarget::Boundary { .. }
        | ObservationPartitionTarget::BoundaryPair { .. }
        | ObservationPartitionTarget::ConstraintGroups { .. } => {
            return Err(ObservationError::Identity(
                "integral partitions do not own selected entities",
            ))
        }
    })
}
