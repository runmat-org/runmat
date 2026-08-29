use runmat_types::{
    DistributedOwner, DistributedValueId, DistributionScheme, LabCount, LabRank, ParallelRegionId,
    ValueFact,
};
use serde::{Deserialize, Serialize};

use crate::value::ValueRef;
use crate::{
    CompositeId, ContractError, DistributedObjectId, ExecutionScopeId, GangHandle, PoolHandle,
};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DistributedValueHandle {
    pub id: DistributedObjectId,
    /// Stable compiler identity for the value contract that created this
    /// runtime object.
    pub contract: DistributedValueId,
    pub owner: DistributedOwner,
    pub scope_id: ExecutionScopeId,
    pub generation: u64,
    pub pool: PoolHandle,
    pub partition_count: LabCount,
    /// Immutable semantic metadata. Array payloads and partition references
    /// remain owned by the execution service.
    pub value: ValueFact,
    pub global_shape: Vec<u64>,
    pub scheme: DistributionScheme,
    pub materializable: bool,
}

impl DistributedValueHandle {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.generation == 0
            || self.partition_count.0 == 0
            || self.scope_id != self.pool.scope_id
            || self.owner.function() != self.contract.function
        {
            return Err(ContractError::invalid(
                "distributed value handle",
                "generation and partition count must be non-zero and the handle must match the pool scope",
            ));
        }
        match &self.scheme {
            DistributionScheme::Block { dimension }
            | DistributionScheme::Cyclic { dimension }
            | DistributionScheme::OneDimensional { dimension, .. } => {
                dimension_extent(*dimension, &self.global_shape)?;
            }
            DistributionScheme::TwoDimensionalBlockCyclic {
                worker_grid,
                block_size,
                ..
            } => {
                if self.global_shape.len() != 2
                    || worker_grid.contains(&0)
                    || u64::from(worker_grid[0]) * u64::from(worker_grid[1])
                        != u64::from(self.partition_count.0)
                    || *block_size == 0
                {
                    return Err(ContractError::invalid(
                        "distributed value handle",
                        "2-D block-cyclic distribution requires a matrix, a positive worker grid matching the partition count, and a positive block size",
                    ));
                }
            }
            DistributionScheme::Custom { partitioner }
                if partitioner.trim().is_empty() || partitioner.contains('\0') =>
            {
                return Err(ContractError::invalid(
                    "distributed value handle",
                    "custom partitioner identity must be non-empty and contain no NUL",
                ));
            }
            DistributionScheme::Replicated | DistributionScheme::Custom { .. } => {}
        }
        if let DistributionScheme::OneDimensional { partition, .. } = &self.scheme {
            let extent = dimension_extent(
                match &self.scheme {
                    DistributionScheme::OneDimensional { dimension, .. } => *dimension,
                    _ => unreachable!(),
                },
                &self.global_shape,
            )?;
            let partition_extent = partition.iter().try_fold(0_u64, |sum, length| {
                sum.checked_add(*length).ok_or_else(|| {
                    ContractError::invalid(
                        "distributed value handle",
                        "1-D partition lengths overflow the global extent",
                    )
                })
            })?;
            if partition.len() != self.partition_count.0 as usize || partition_extent != extent {
                return Err(ContractError::invalid(
                    "distributed value handle",
                    "1-D partition lengths must match the worker count and distribution extent",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompositeHandle {
    pub id: CompositeId,
    pub owner_region: ParallelRegionId,
    pub scope_id: ExecutionScopeId,
    pub generation: u64,
    pub gang: GangHandle,
    /// Join of the values retained by the gang. Individual entries remain
    /// execution-service-owned and may be more precise.
    pub value: ValueFact,
}

impl CompositeHandle {
    pub fn validate(&self) -> Result<(), ContractError> {
        self.gang.validate()?;
        if self.generation == 0 || self.scope_id != self.gang.scope_id {
            return Err(ContractError::invalid(
                "composite handle",
                "handle generation must be non-zero and match the gang scope",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartitionRange {
    /// One-based array dimension described by this range.
    pub dimension: u32,
    /// Zero-based inclusive element offset along the dimension.
    pub start: u64,
    /// Zero-based exclusive element offset along the dimension.
    pub end: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "selection", deny_unknown_fields)]
pub enum PartitionSelection {
    Range(PartitionRange),
    Strided {
        dimension: u32,
        start: u64,
        step: u64,
        count: u64,
    },
    Indices {
        dimension: u32,
        indices: Vec<u64>,
    },
}

impl PartitionSelection {
    pub fn dimension(&self) -> u32 {
        match self {
            Self::Range(range) => range.dimension,
            Self::Strided { dimension, .. } | Self::Indices { dimension, .. } => *dimension,
        }
    }

    pub fn element_count(&self) -> u64 {
        match self {
            Self::Range(range) => range.end - range.start,
            Self::Strided { count, .. } => *count,
            Self::Indices { indices, .. } => indices.len() as u64,
        }
    }

    pub fn validate(&self, global_shape: &[u64]) -> Result<(), ContractError> {
        match self {
            Self::Range(range) => range.validate(global_shape),
            Self::Strided {
                dimension,
                start,
                step,
                count,
            } => {
                let extent = dimension_extent(*dimension, global_shape)?;
                if *step == 0 {
                    return Err(ContractError::invalid(
                        "distributed strided selection",
                        "step must be non-zero",
                    ));
                }
                if *count == 0 {
                    // Canonical cyclic layouts retain one rank-derived start
                    // per lab. When there are more labs than elements, an
                    // empty tail rank can therefore begin beyond the extent;
                    // no element is selected or dereferenced.
                    return Ok(());
                }
                let last = start
                    .checked_add(step.checked_mul(count - 1).ok_or_else(|| {
                        ContractError::invalid(
                            "distributed strided selection",
                            "selection extent overflows",
                        )
                    })?)
                    .ok_or_else(|| {
                        ContractError::invalid(
                            "distributed strided selection",
                            "selection extent overflows",
                        )
                    })?;
                if last >= extent {
                    return Err(ContractError::invalid(
                        "distributed strided selection",
                        "selection lies outside the dimension",
                    ));
                }
                Ok(())
            }
            Self::Indices { dimension, indices } => {
                let extent = dimension_extent(*dimension, global_shape)?;
                if indices.windows(2).any(|pair| pair[0] >= pair[1])
                    || indices.iter().any(|index| *index >= extent)
                {
                    return Err(ContractError::invalid(
                        "distributed indexed selection",
                        "indices must be unique, ascending, and lie within the dimension",
                    ));
                }
                Ok(())
            }
        }
    }
}

fn dimension_extent(dimension: u32, global_shape: &[u64]) -> Result<u64, ContractError> {
    usize::try_from(dimension)
        .ok()
        .and_then(|dimension| dimension.checked_sub(1))
        .and_then(|dimension| global_shape.get(dimension).copied())
        .ok_or_else(|| {
            ContractError::invalid(
                "distributed partition selection",
                "dimension must be one-based and lie within the global shape",
            )
        })
}

impl PartitionRange {
    pub fn validate(&self, global_shape: &[u64]) -> Result<(), ContractError> {
        let dimension = usize::try_from(self.dimension)
            .ok()
            .and_then(|dimension| dimension.checked_sub(1))
            .ok_or_else(|| {
                ContractError::invalid("distributed partition range", "dimension must be one-based")
            })?;
        if self.start > self.end
            || global_shape
                .get(dimension)
                .is_none_or(|extent| self.end > *extent)
        {
            return Err(ContractError::invalid(
                "distributed partition range",
                "range must be ordered and lie within the global shape",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DistributedPartitionLayout {
    pub rank: LabRank,
    /// One selection per global dimension, in canonical dimension order.
    pub selections: Vec<PartitionSelection>,
    pub local_shape: Vec<u64>,
}

impl DistributedPartitionLayout {
    pub fn validate(&self, global_shape: &[u64]) -> Result<(), ContractError> {
        validate_partition_layout(self, global_shape)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DistributedPartition {
    #[serde(flatten)]
    pub layout: DistributedPartitionLayout,
    pub value: ValueRef,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DistributedValueSnapshot {
    pub handle: DistributedValueHandle,
    pub partitions: Vec<DistributedPartition>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DistributedOwnedPartition {
    pub layout: DistributedPartitionLayout,
    pub value: crate::value::ValuePayload,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DistributedShardSnapshot {
    pub handle: DistributedValueHandle,
    pub layouts: Vec<DistributedPartitionLayout>,
    pub owned: DistributedOwnedPartition,
}

impl DistributedShardSnapshot {
    pub fn validate(&self) -> Result<(), ContractError> {
        self.handle.validate()?;
        validate_partition_layouts(&self.handle, &self.layouts)?;
        self.owned
            .value
            .validate(crate::value::ValueLimits::default())?;
        if matches!(
            self.owned.value,
            crate::value::ValuePayload::Distributed(_) | crate::value::ValuePayload::Composite(_)
        ) || self
            .layouts
            .iter()
            .find(|layout| layout.rank == self.owned.layout.rank)
            != Some(&self.owned.layout)
        {
            return Err(ContractError::invalid(
                "distributed shard snapshot",
                "owned payload must match one authoritative non-recursive partition layout",
            ));
        }
        Ok(())
    }
}

impl DistributedValueSnapshot {
    pub fn validate(&self) -> Result<(), ContractError> {
        self.handle.validate()?;
        let layouts = self
            .partitions
            .iter()
            .map(|partition| partition.layout.clone())
            .collect::<Vec<_>>();
        validate_partition_layouts(&self.handle, &layouts)?;
        for partition in &self.partitions {
            crate::value::ValuePayload::Object(Box::new(partition.value.clone()))
                .validate(crate::value::ValueLimits::default())?;
        }
        Ok(())
    }
}

pub fn validate_partition_layouts(
    handle: &DistributedValueHandle,
    partitions: &[DistributedPartitionLayout],
) -> Result<(), ContractError> {
    handle.validate()?;
    if partitions.len() != handle.partition_count.0 as usize {
        return Err(ContractError::invalid(
            "distributed value snapshot",
            "partition count must match the handle contract",
        ));
    }
    let expected_ranks = (1..=handle.partition_count.0)
        .map(LabRank)
        .collect::<Vec<_>>();
    let actual_ranks = partitions
        .iter()
        .map(|partition| partition.rank)
        .collect::<Vec<_>>();
    if actual_ranks != expected_ranks {
        return Err(ContractError::invalid(
            "distributed value snapshot",
            "partitions must have complete, one-based, unique, canonically ordered ranks",
        ));
    }
    for partition in partitions {
        partition.validate(&handle.global_shape)?;
    }
    validate_scheme(handle, partitions)?;
    Ok(())
}

fn validate_partition_layout(
    partition: &DistributedPartitionLayout,
    global_shape: &[u64],
) -> Result<(), ContractError> {
    if partition.selections.len() != global_shape.len()
        || partition.local_shape.len() != global_shape.len()
    {
        return Err(ContractError::invalid(
            "distributed partition",
            "selection and local-shape ranks must match the global shape",
        ));
    }
    for (axis, selection) in partition.selections.iter().enumerate() {
        if selection.dimension() != axis as u32 + 1 {
            return Err(ContractError::invalid(
                "distributed partition",
                "selections must cover every dimension in canonical order",
            ));
        }
        selection.validate(global_shape)?;
        if selection.element_count() != partition.local_shape[axis] {
            return Err(ContractError::invalid(
                "distributed partition",
                "local shape must match each partition selection",
            ));
        }
    }
    Ok(())
}

fn validate_scheme(
    handle: &DistributedValueHandle,
    partitions: &[DistributedPartitionLayout],
) -> Result<(), ContractError> {
    match &handle.scheme {
        DistributionScheme::Replicated => {
            for partition in partitions {
                let complete = partition
                    .selections
                    .iter()
                    .enumerate()
                    .all(|(axis, value)| {
                        matches!(
                            value,
                            PartitionSelection::Range(PartitionRange {
                                start: 0,
                                end,
                                ..
                            }) if *end == handle.global_shape[axis]
                        )
                    });
                if !complete {
                    return Err(ContractError::invalid(
                        "replicated distributed value",
                        "every partition must select the complete global value",
                    ));
                }
            }
        }
        DistributionScheme::Block { dimension }
        | DistributionScheme::OneDimensional { dimension, .. } => {
            let axis = partition_axis(*dimension, &handle.global_shape)?;
            let mut cursor = 0_u64;
            for (index, partition) in partitions.iter().enumerate() {
                match &partition.selections[axis] {
                    PartitionSelection::Range(range) if range.start == cursor => {
                        if let DistributionScheme::OneDimensional {
                            partition: lengths, ..
                        } = &handle.scheme
                        {
                            if range.end - range.start != lengths[index] {
                                return Err(ContractError::invalid(
                                    "1-D distributed value",
                                    "partition selection lengths must match the codistributor",
                                ));
                            }
                        }
                        cursor = range.end;
                    }
                    _ => {
                        return Err(ContractError::invalid(
                            "block distributed value",
                            "block selections must be contiguous and canonically ranked",
                        ));
                    }
                }
            }
            if cursor != handle.global_shape[axis] {
                return Err(ContractError::invalid(
                    "block distributed value",
                    "block selections must cover the partitioned dimension exactly",
                ));
            }
        }
        DistributionScheme::Cyclic { dimension } => {
            let axis = partition_axis(*dimension, &handle.global_shape)?;
            let labs = u64::from(handle.partition_count.0);
            for partition in partitions {
                let expected_start = u64::from(partition.rank.0 - 1);
                match &partition.selections[axis] {
                    PartitionSelection::Strided { start, step, .. }
                        if *start == expected_start && *step == labs => {}
                    _ => {
                        return Err(ContractError::invalid(
                            "cyclic distributed value",
                            "cyclic selections must use rank-derived starts and the partition count as stride",
                        ));
                    }
                }
            }
        }
        DistributionScheme::TwoDimensionalBlockCyclic {
            worker_grid,
            block_size,
            orientation,
        } => {
            for partition in partitions {
                let zero_based = partition.rank.0 - 1;
                let (grid_row, grid_column) = match orientation {
                    runmat_types::WorkerGridOrientation::Row => {
                        (zero_based / worker_grid[1], zero_based % worker_grid[1])
                    }
                    runmat_types::WorkerGridOrientation::Column => {
                        (zero_based % worker_grid[0], zero_based / worker_grid[0])
                    }
                };
                for (axis, (grid_extent, grid_position)) in
                    [(worker_grid[0], grid_row), (worker_grid[1], grid_column)]
                        .into_iter()
                        .enumerate()
                {
                    let expected = (0..handle.global_shape[axis])
                        .filter(|index| {
                            ((*index / *block_size) % u64::from(grid_extent))
                                == u64::from(grid_position)
                        })
                        .collect::<Vec<_>>();
                    if partition.selections[axis]
                        != (PartitionSelection::Indices {
                            dimension: axis as u32 + 1,
                            indices: expected,
                        })
                    {
                        return Err(ContractError::invalid(
                            "2-D block-cyclic distributed value",
                            "partition selections do not match the worker grid and block size",
                        ));
                    }
                }
            }
        }
        DistributionScheme::Custom { .. } => {}
    }
    Ok(())
}

fn partition_axis(dimension: u32, global_shape: &[u64]) -> Result<usize, ContractError> {
    dimension_extent(dimension, global_shape)?;
    Ok(dimension as usize - 1)
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompositeSnapshot {
    pub handle: CompositeHandle,
    /// One entry per rank. `None` represents a variable that was not defined
    /// by that rank; the vector itself is always complete and ordered.
    pub entries: Vec<Option<ValueRef>>,
}

impl CompositeSnapshot {
    pub fn validate(&self) -> Result<(), ContractError> {
        self.handle.validate()?;
        if self.entries.len() != self.handle.gang.labs.0 as usize {
            return Err(ContractError::invalid(
                "composite snapshot",
                "entry count must match the gang's lab count",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::ValueId;
    use crate::schema::VALUE_PAYLOAD_SCHEMA_V1;
    use crate::value::{ResidentFence, ValueRefKind};
    use crate::{Digest, PoolId};
    use runmat_types::{
        NumericClass, NumericDomain, NumericFact, ProgramFunctionId, RegionId, ValueKindFact,
    };

    fn value_fact() -> ValueFact {
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }))
    }

    fn handle(scheme: DistributionScheme) -> DistributedValueHandle {
        let function = ProgramFunctionId(4);
        let owner_region = ParallelRegionId(RegionId {
            function,
            ordinal: 2,
        });
        let scope_id = ExecutionScopeId::derive(&[b"distributed-test"]);
        DistributedValueHandle {
            id: DistributedObjectId::derive(&[b"array"]),
            contract: DistributedValueId {
                function,
                ordinal: 3,
            },
            owner: DistributedOwner::Region(owner_region),
            scope_id,
            generation: 1,
            pool: PoolHandle {
                id: PoolId::derive(&[b"pool"]),
                scope_id,
                generation: 1,
            },
            partition_count: LabCount(2),
            value: value_fact(),
            global_shape: vec![8, 2],
            scheme,
            materializable: true,
        }
    }

    fn value_ref(rank: u32) -> ValueRef {
        let rank_bytes = rank.to_be_bytes();
        ValueRef {
            schema_version: VALUE_PAYLOAD_SCHEMA_V1,
            id: ValueId::derive(&[b"partition".as_slice(), rank_bytes.as_slice()]),
            logical_digest: Digest::sha256(rank.to_be_bytes()),
            encoded_length: 64,
            media_type: "application/vnd.runmat.value".into(),
            value_schema: "runmat-value-v1".into(),
            encryption_context: Digest::sha256(b"test-encryption-context"),
            kind: ValueRefKind::SlicedObject,
            authorization_scope: "test".into(),
            resident_fence: None::<ResidentFence>,
        }
    }

    fn complete_axis(dimension: u32, end: u64) -> PartitionSelection {
        PartitionSelection::Range(PartitionRange {
            dimension,
            start: 0,
            end,
        })
    }

    #[test]
    fn block_snapshot_requires_exact_canonical_coverage() {
        let snapshot = DistributedValueSnapshot {
            handle: handle(DistributionScheme::Block { dimension: 1 }),
            partitions: vec![
                DistributedPartition {
                    layout: DistributedPartitionLayout {
                        rank: LabRank(1),
                        selections: vec![
                            PartitionSelection::Range(PartitionRange {
                                dimension: 1,
                                start: 0,
                                end: 4,
                            }),
                            complete_axis(2, 2),
                        ],
                        local_shape: vec![4, 2],
                    },
                    value: value_ref(1),
                },
                DistributedPartition {
                    layout: DistributedPartitionLayout {
                        rank: LabRank(2),
                        selections: vec![
                            PartitionSelection::Range(PartitionRange {
                                dimension: 1,
                                start: 4,
                                end: 8,
                            }),
                            complete_axis(2, 2),
                        ],
                        local_shape: vec![4, 2],
                    },
                    value: value_ref(2),
                },
            ],
        };
        snapshot.validate().expect("complete block partition");

        let mut gap = snapshot.clone();
        let PartitionSelection::Range(range) = &mut gap.partitions[1].layout.selections[0] else {
            unreachable!()
        };
        range.start = 5;
        assert!(gap.validate().is_err());
    }

    #[test]
    fn cyclic_snapshot_uses_rank_starts_and_gang_stride() {
        let snapshot = DistributedValueSnapshot {
            handle: handle(DistributionScheme::Cyclic { dimension: 1 }),
            partitions: (1..=2)
                .map(|rank| DistributedPartition {
                    layout: DistributedPartitionLayout {
                        rank: LabRank(rank),
                        selections: vec![
                            PartitionSelection::Strided {
                                dimension: 1,
                                start: u64::from(rank - 1),
                                step: 2,
                                count: 4,
                            },
                            complete_axis(2, 2),
                        ],
                        local_shape: vec![4, 2],
                    },
                    value: value_ref(rank),
                })
                .collect(),
        };
        snapshot.validate().expect("complete cyclic partition");
    }
}
