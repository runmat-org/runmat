use std::collections::BTreeMap;

use runmat_execution::value::{ValueLimits, ValuePayload};
use runmat_execution::{
    validate_partition_layouts, CompositeHandle, CompositeId, DistributedObjectId,
    DistributedOwnedPartition, DistributedPartitionLayout, DistributedShardSnapshot,
    DistributedValueHandle, PoolHandle,
};
use runmat_types::LabRank;

use crate::{RunnerError, RunnerResult};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OwnedPartition {
    pub layout: DistributedPartitionLayout,
    pub value: ValuePayload,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RetiredDistributedObjects {
    pub distributed: usize,
    pub composites: usize,
}

#[derive(Clone, Debug)]
struct DistributedRecord {
    handle: DistributedValueHandle,
    layouts: Vec<DistributedPartitionLayout>,
    partitions: BTreeMap<LabRank, OwnedPartition>,
}

#[derive(Clone, Debug)]
struct CompositeRecord {
    handle: CompositeHandle,
    entries: Vec<Option<ValuePayload>>,
}

/// Execution-owned distributed payload storage. Public runtime values carry
/// only fenced handles; partition payloads stay behind this authority.
#[derive(Default)]
pub struct DistributedStore {
    distributed: BTreeMap<DistributedObjectId, DistributedRecord>,
    composites: BTreeMap<CompositeId, CompositeRecord>,
}

impl DistributedStore {
    pub fn insert(
        &mut self,
        handle: DistributedValueHandle,
        partitions: Vec<OwnedPartition>,
    ) -> RunnerResult<()> {
        handle.validate().map_err(invalid)?;
        let layouts = partitions
            .iter()
            .map(|partition| partition.layout.clone())
            .collect::<Vec<_>>();
        validate_partition_layouts(&handle, &layouts).map_err(invalid)?;
        for partition in &partitions {
            partition
                .value
                .validate(ValueLimits::default())
                .map_err(invalid)?;
            if matches!(
                partition.value,
                ValuePayload::Distributed(_) | ValuePayload::Composite(_)
            ) {
                return Err(RunnerError::Invalid(
                    "distributed partitions cannot recursively contain execution handles".into(),
                ));
            }
        }
        match self.distributed.get(&handle.id) {
            Some(record) if record.handle != handle => {
                return Err(RunnerError::Invalid(
                    "distributed object identity is already bound to another generation".into(),
                ));
            }
            Some(_) => {
                return Err(RunnerError::Invalid(
                    "distributed object was inserted more than once".into(),
                ));
            }
            None => {}
        }
        let layouts = partitions
            .iter()
            .map(|partition| partition.layout.clone())
            .collect();
        let partitions = partitions
            .into_iter()
            .map(|partition| (partition.layout.rank, partition))
            .collect();
        self.distributed.insert(
            handle.id,
            DistributedRecord {
                handle,
                layouts,
                partitions,
            },
        );
        Ok(())
    }

    pub fn insert_local_coordinated(
        &mut self,
        handle: DistributedValueHandle,
        layouts: Vec<DistributedPartitionLayout>,
        partition: OwnedPartition,
    ) -> RunnerResult<()> {
        handle.validate().map_err(invalid)?;
        validate_partition_layouts(&handle, &layouts).map_err(invalid)?;
        partition
            .value
            .validate(ValueLimits::default())
            .map_err(invalid)?;
        if matches!(
            partition.value,
            ValuePayload::Distributed(_) | ValuePayload::Composite(_)
        ) || layouts
            .iter()
            .find(|layout| layout.rank == partition.layout.rank)
            != Some(&partition.layout)
        {
            return Err(RunnerError::Invalid(
                "owned distributed partition disagrees with its authoritative layout".into(),
            ));
        }
        let record = self
            .distributed
            .entry(handle.id)
            .or_insert_with(|| DistributedRecord {
                handle: handle.clone(),
                layouts: layouts.clone(),
                partitions: BTreeMap::new(),
            });
        if record.handle != handle || record.layouts != layouts {
            return Err(RunnerError::Invalid(
                "coordinated distributed object identity disagrees with its installed layout directory"
                    .into(),
            ));
        }
        match record.partitions.get(&partition.layout.rank) {
            Some(existing) if existing == &partition => Ok(()),
            Some(_) => Err(RunnerError::Invalid(
                "distributed partition rank is already bound to another payload".into(),
            )),
            None => {
                record.partitions.insert(partition.layout.rank, partition);
                Ok(())
            }
        }
    }

    pub fn handle(&self, handle: &DistributedValueHandle) -> RunnerResult<&DistributedValueHandle> {
        Ok(&self.record(handle)?.handle)
    }

    pub fn local_part(
        &self,
        handle: &DistributedValueHandle,
        rank: LabRank,
    ) -> RunnerResult<&OwnedPartition> {
        let record = self.record(handle)?;
        if rank.0 == 0 || rank.0 > handle.partition_count.0 {
            return Err(RunnerError::Invalid(
                "distributed partition rank is outside the admitted lab range".into(),
            ));
        }
        record
            .partitions
            .get(&rank)
            .ok_or_else(|| RunnerError::Invalid("distributed partition is unavailable".into()))
    }

    pub fn layouts(
        &self,
        handle: &DistributedValueHandle,
    ) -> RunnerResult<Vec<DistributedPartitionLayout>> {
        Ok(self.record(handle)?.layouts.clone())
    }

    pub fn cloned_parts(
        &self,
        handle: &DistributedValueHandle,
    ) -> RunnerResult<Vec<OwnedPartition>> {
        let record = self.record(handle)?;
        (1..=handle.partition_count.0)
            .map(|rank| {
                record
                    .partitions
                    .get(&LabRank(rank))
                    .cloned()
                    .ok_or_else(|| {
                        RunnerError::Invalid(
                            "distributed operation requires a partition owned by another execution node"
                                .into(),
                        )
                    })
            })
            .collect()
    }

    pub fn export_local(
        &self,
        handle: &DistributedValueHandle,
        rank: LabRank,
    ) -> RunnerResult<DistributedShardSnapshot> {
        let record = self.record(handle)?;
        let partition = record.partitions.get(&rank).ok_or_else(|| {
            RunnerError::Invalid("distributed partition is not owned by this execution node".into())
        })?;
        let snapshot = DistributedShardSnapshot {
            handle: handle.clone(),
            layouts: record.layouts.clone(),
            owned: DistributedOwnedPartition {
                layout: partition.layout.clone(),
                value: partition.value.clone(),
            },
        };
        snapshot.validate().map_err(invalid)?;
        Ok(snapshot)
    }

    pub fn import_shard(&mut self, snapshot: DistributedShardSnapshot) -> RunnerResult<()> {
        snapshot.validate().map_err(invalid)?;
        self.insert_local_coordinated(
            snapshot.handle,
            snapshot.layouts,
            OwnedPartition {
                layout: snapshot.owned.layout,
                value: snapshot.owned.value,
            },
        )
    }

    pub fn retire(&mut self, handle: &DistributedValueHandle) -> RunnerResult<()> {
        self.record(handle)?;
        self.distributed.remove(&handle.id);
        Ok(())
    }

    pub fn insert_composite(
        &mut self,
        handle: CompositeHandle,
        entries: Vec<Option<ValuePayload>>,
    ) -> RunnerResult<()> {
        handle.validate().map_err(invalid)?;
        if entries.len() != handle.gang.labs.0 as usize {
            return Err(RunnerError::Invalid(
                "composite must contain one ordered entry per lab".into(),
            ));
        }
        for value in entries.iter().flatten() {
            value.validate(ValueLimits::default()).map_err(invalid)?;
            if matches!(value, ValuePayload::Composite(_)) {
                return Err(RunnerError::Invalid(
                    "composite entries cannot recursively contain a composite handle".into(),
                ));
            }
        }
        if self.composites.contains_key(&handle.id) {
            return Err(RunnerError::Invalid(
                "composite object was inserted more than once".into(),
            ));
        }
        self.composites
            .insert(handle.id, CompositeRecord { handle, entries });
        Ok(())
    }

    pub fn composite_entry(
        &self,
        handle: &CompositeHandle,
        rank: LabRank,
    ) -> RunnerResult<Option<&ValuePayload>> {
        let record = self
            .composites
            .get(&handle.id)
            .filter(|record| record.handle == *handle)
            .ok_or_else(|| {
                RunnerError::Invalid("composite handle is stale or unavailable".into())
            })?;
        if rank.0 == 0 || rank.0 > handle.gang.labs.0 {
            return Err(RunnerError::Invalid(
                "composite rank is outside the admitted lab range".into(),
            ));
        }
        Ok(record.entries[(rank.0 - 1) as usize].as_ref())
    }

    pub fn retire_pool(&mut self, pool: &PoolHandle) -> RetiredDistributedObjects {
        let distributed_before = self.distributed.len();
        let composites_before = self.composites.len();
        self.distributed.retain(|_, record| {
            record.handle.pool.id != pool.id
                || record.handle.pool.generation != pool.generation
                || record.handle.pool.scope_id != pool.scope_id
        });
        self.composites.retain(|_, record| {
            record.handle.gang.pool.id != pool.id
                || record.handle.gang.pool.generation != pool.generation
                || record.handle.gang.pool.scope_id != pool.scope_id
        });
        RetiredDistributedObjects {
            distributed: distributed_before - self.distributed.len(),
            composites: composites_before - self.composites.len(),
        }
    }

    fn record(&self, handle: &DistributedValueHandle) -> RunnerResult<&DistributedRecord> {
        handle.validate().map_err(invalid)?;
        self.distributed
            .get(&handle.id)
            .filter(|record| record.handle == *handle)
            .ok_or_else(|| {
                RunnerError::Invalid("distributed handle is stale or unavailable".into())
            })
    }
}

fn invalid(error: impl std::fmt::Display) -> RunnerError {
    RunnerError::Invalid(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::{
        ExecutionScopeId, GangHandle, GangId, PartitionRange, PartitionSelection, PoolId,
    };
    use runmat_types::{
        DistributedValueId, DistributionScheme, LabCount, NumericClass, NumericDomain, NumericFact,
        ParallelRegionId, ProgramFunctionId, RegionId, ValueFact, ValueKindFact,
    };

    fn fixture() -> (DistributedValueHandle, Vec<OwnedPartition>) {
        let scope_id = ExecutionScopeId::derive(&[b"store"]);
        let function = ProgramFunctionId(1);
        let owner_region = ParallelRegionId(RegionId {
            function,
            ordinal: 1,
        });
        let handle = DistributedValueHandle {
            id: DistributedObjectId::derive(&[b"value"]),
            contract: DistributedValueId {
                function,
                ordinal: 1,
            },
            owner: runmat_types::DistributedOwner::Region(owner_region),
            scope_id,
            generation: 1,
            pool: PoolHandle {
                id: PoolId::derive(&[b"pool"]),
                scope_id,
                generation: 1,
            },
            partition_count: LabCount(2),
            value: ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })),
            global_shape: vec![4, 1],
            scheme: DistributionScheme::Block { dimension: 1 },
            materializable: true,
        };
        let partitions = (0..2)
            .map(|index| OwnedPartition {
                layout: DistributedPartitionLayout {
                    rank: LabRank(index + 1),
                    selections: vec![
                        PartitionSelection::Range(PartitionRange {
                            dimension: 1,
                            start: u64::from(index * 2),
                            end: u64::from(index * 2 + 2),
                        }),
                        PartitionSelection::Range(PartitionRange {
                            dimension: 2,
                            start: 0,
                            end: 1,
                        }),
                    ],
                    local_shape: vec![2, 1],
                },
                value: ValuePayload::Inline(Box::new(
                    runmat_execution::value::InlineValue::F64Bits(f64::from(index).to_bits()),
                )),
            })
            .collect();
        (handle, partitions)
    }

    #[test]
    fn store_requires_exact_handle_fences_and_ranked_layouts() {
        let (handle, partitions) = fixture();
        let mut store = DistributedStore::default();
        store.insert(handle.clone(), partitions).unwrap();
        assert_eq!(
            store.local_part(&handle, LabRank(2)).unwrap().layout.rank,
            LabRank(2)
        );
        let mut stale = handle.clone();
        stale.generation += 1;
        assert!(store.local_part(&stale, LabRank(1)).is_err());
    }

    #[test]
    fn shard_export_and_import_preserve_local_ownership_until_every_rank_arrives() {
        let (handle, partitions) = fixture();
        let layouts = partitions
            .iter()
            .map(|partition| partition.layout.clone())
            .collect::<Vec<_>>();
        let mut worker_one = DistributedStore::default();
        worker_one
            .insert_local_coordinated(handle.clone(), layouts.clone(), partitions[0].clone())
            .unwrap();
        assert!(worker_one.cloned_parts(&handle).is_err());

        let first = worker_one.export_local(&handle, LabRank(1)).unwrap();
        let mut driver = DistributedStore::default();
        driver.import_shard(first).unwrap();
        assert_eq!(
            driver.local_part(&handle, LabRank(1)).unwrap(),
            &partitions[0]
        );
        assert!(driver.local_part(&handle, LabRank(2)).is_err());

        let mut worker_two = DistributedStore::default();
        worker_two
            .insert_local_coordinated(handle.clone(), layouts, partitions[1].clone())
            .unwrap();
        driver
            .import_shard(worker_two.export_local(&handle, LabRank(2)).unwrap())
            .unwrap();
        assert_eq!(driver.cloned_parts(&handle).unwrap(), partitions);
    }

    #[test]
    fn composite_entries_preserve_absent_per_rank_values() {
        let (distributed, _) = fixture();
        let gang = GangHandle {
            id: GangId::derive(&[b"gang"]),
            scope_id: distributed.scope_id,
            generation: 1,
            pool: distributed.pool,
            labs: LabCount(2),
        };
        let handle = CompositeHandle {
            id: CompositeId::derive(&[b"composite"]),
            owner_region: match distributed.owner {
                runmat_types::DistributedOwner::Region(region) => region,
                runmat_types::DistributedOwner::Client(_) => {
                    panic!("test fixture uses a region-owned distributed value")
                }
            },
            scope_id: distributed.scope_id,
            generation: 1,
            gang,
            value: distributed.value,
        };
        let mut store = DistributedStore::default();
        store
            .insert_composite(
                handle.clone(),
                vec![
                    Some(ValuePayload::Inline(Box::new(
                        runmat_execution::value::InlineValue::U64(9),
                    ))),
                    None,
                ],
            )
            .unwrap();
        assert!(store
            .composite_entry(&handle, LabRank(1))
            .unwrap()
            .is_some());
        assert!(store
            .composite_entry(&handle, LabRank(2))
            .unwrap()
            .is_none());
    }

    #[test]
    fn retiring_a_pool_fences_distributed_and_composite_generations() {
        let (distributed, partitions) = fixture();
        let gang = GangHandle {
            id: GangId::derive(&[b"retired-gang"]),
            scope_id: distributed.scope_id,
            generation: 1,
            pool: distributed.pool.clone(),
            labs: LabCount(2),
        };
        let composite = CompositeHandle {
            id: CompositeId::derive(&[b"retired-composite"]),
            owner_region: match distributed.owner {
                runmat_types::DistributedOwner::Region(region) => region,
                runmat_types::DistributedOwner::Client(_) => unreachable!(),
            },
            scope_id: distributed.scope_id,
            generation: 1,
            gang,
            value: distributed.value.clone(),
        };
        let mut store = DistributedStore::default();
        store.insert(distributed.clone(), partitions).unwrap();
        store
            .insert_composite(composite.clone(), vec![None, None])
            .unwrap();
        assert_eq!(
            store.retire_pool(&distributed.pool),
            RetiredDistributedObjects {
                distributed: 1,
                composites: 1,
            }
        );
        assert!(store.local_part(&distributed, LabRank(1)).is_err());
        assert!(store.composite_entry(&composite, LabRank(1)).is_err());
    }

    #[test]
    fn pool_retirement_is_exact_and_idempotent() {
        let (retired, retired_parts) = fixture();
        let mut current = retired.clone();
        current.id = DistributedObjectId::derive(&[b"current-generation"]);
        current.generation += 1;
        current.pool.generation += 1;
        let current_parts = retired_parts.clone();

        let mut unrelated = retired.clone();
        unrelated.id = DistributedObjectId::derive(&[b"unrelated-pool"]);
        unrelated.pool.id = PoolId::derive(&[b"other-pool"]);
        let unrelated_parts = retired_parts.clone();

        let mut store = DistributedStore::default();
        store.insert(retired.clone(), retired_parts).unwrap();
        store.insert(current.clone(), current_parts).unwrap();
        store.insert(unrelated.clone(), unrelated_parts).unwrap();

        assert_eq!(
            store.retire_pool(&retired.pool),
            RetiredDistributedObjects {
                distributed: 1,
                composites: 0,
            }
        );
        assert!(store.local_part(&retired, LabRank(1)).is_err());
        assert!(store.local_part(&current, LabRank(1)).is_ok());
        assert!(store.local_part(&unrelated, LabRank(1)).is_ok());
        assert_eq!(
            store.retire_pool(&retired.pool),
            RetiredDistributedObjects::default()
        );
    }
}
