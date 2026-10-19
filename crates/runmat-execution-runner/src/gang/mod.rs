use std::collections::BTreeMap;

use runmat_execution::{GangHandle, GangId, GangRequest, GangSnapshot};
use runmat_types::{LabCount, LabRank, SpmdLabRequirement};

use crate::{RunnerError, RunnerResult};

/// Deterministic gang admission and fencing. Worker selection remains the
/// scheduler's responsibility; this authority fixes the lab count and stable
/// one-based ranks for one admitted generation.
#[derive(Default)]
pub struct GangCoordinator {
    generations: BTreeMap<GangId, u64>,
    active: BTreeMap<GangId, GangSnapshot>,
}

impl GangCoordinator {
    pub fn admit(
        &mut self,
        request: GangRequest,
        available_labs: LabCount,
    ) -> RunnerResult<GangSnapshot> {
        if request.pool.generation == 0 {
            return Err(RunnerError::Invalid(
                "SPMD admission requires a live pool generation".into(),
            ));
        }
        let labs = select_lab_count(request.labs, available_labs)?;
        let generation = self
            .generations
            .entry(GangId::derive(&[
                request.pool.id.bytes(),
                request.pool.scope_id.bytes(),
            ]))
            .and_modify(|generation| *generation += 1)
            .or_insert(1);
        let handle = GangHandle {
            id: GangId::derive(&[
                request.pool.id.bytes(),
                request.pool.scope_id.bytes(),
                &generation.to_be_bytes(),
            ]),
            scope_id: request.pool.scope_id,
            generation: *generation,
            pool: request.pool,
            labs,
        };
        let snapshot = GangSnapshot {
            handle: handle.clone(),
            ranks: (1..=labs.0).map(LabRank).collect(),
        };
        snapshot.validate().map_err(invalid)?;
        self.active.insert(handle.id, snapshot.clone());
        Ok(snapshot)
    }

    pub fn snapshot(&self, handle: &GangHandle) -> RunnerResult<&GangSnapshot> {
        handle.validate().map_err(invalid)?;
        self.active
            .get(&handle.id)
            .filter(|snapshot| snapshot.handle == *handle)
            .ok_or_else(|| RunnerError::Invalid("SPMD gang handle is stale or inactive".into()))
    }

    pub fn retire(&mut self, handle: &GangHandle) -> RunnerResult<GangSnapshot> {
        self.snapshot(handle)?;
        self.active
            .remove(&handle.id)
            .ok_or_else(|| RunnerError::Invalid("SPMD gang is not active".into()))
    }
}

fn select_lab_count(
    requirement: SpmdLabRequirement,
    available: LabCount,
) -> RunnerResult<LabCount> {
    if available.0 == 0 {
        return Err(RunnerError::Invalid(
            "SPMD admission requires at least one available lab".into(),
        ));
    }
    match requirement {
        SpmdLabRequirement::Default => Ok(available),
        SpmdLabRequirement::Exact { labs } if labs.0 > 0 && labs.0 <= available.0 => Ok(labs),
        SpmdLabRequirement::Range { minimum, maximum }
            if minimum.0 > 0 && minimum.0 <= maximum.0 && minimum.0 <= available.0 =>
        {
            Ok(LabCount(maximum.0.min(available.0)))
        }
        SpmdLabRequirement::Exact { .. } | SpmdLabRequirement::Range { .. } => Err(
            RunnerError::Invalid("available workers do not satisfy the SPMD lab request".into()),
        ),
    }
}

fn invalid(error: impl std::fmt::Display) -> RunnerError {
    RunnerError::Invalid(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::{ExecutionScopeId, PoolHandle, PoolId};

    fn pool() -> PoolHandle {
        let scope_id = ExecutionScopeId::derive(&[b"gang-test"]);
        PoolHandle {
            id: PoolId::derive(&[b"pool"]),
            scope_id,
            generation: 1,
        }
    }

    #[test]
    fn range_admission_chooses_largest_available_gang_with_stable_ranks() {
        let mut coordinator = GangCoordinator::default();
        let snapshot = coordinator
            .admit(
                GangRequest {
                    pool: pool(),
                    labs: SpmdLabRequirement::Range {
                        minimum: LabCount(2),
                        maximum: LabCount(5),
                    },
                },
                LabCount(3),
            )
            .unwrap();
        assert_eq!(snapshot.handle.labs, LabCount(3));
        assert_eq!(snapshot.ranks, vec![LabRank(1), LabRank(2), LabRank(3)]);
        assert_eq!(coordinator.snapshot(&snapshot.handle).unwrap(), &snapshot);
    }

    #[test]
    fn retired_and_replaced_gangs_are_fenced_by_exact_handle() {
        let mut coordinator = GangCoordinator::default();
        let request = GangRequest {
            pool: pool(),
            labs: SpmdLabRequirement::Exact { labs: LabCount(2) },
        };
        let first = coordinator.admit(request.clone(), LabCount(2)).unwrap();
        coordinator.retire(&first.handle).unwrap();
        assert!(coordinator.snapshot(&first.handle).is_err());
        let second = coordinator.admit(request, LabCount(2)).unwrap();
        assert_ne!(first.handle, second.handle);
    }
}
