use runmat_types::ParallelRegionId;
use serde::{Deserialize, Serialize};

use crate::ContractError;

/// Versioned state for one deterministic random stream assigned to one
/// logical parallel-loop iteration. The state is execution metadata, not a
/// program argument, and therefore cannot be observed or mistaken for a
/// captured workspace value.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "algorithm", rename_all = "snake_case", deny_unknown_fields)]
pub enum ParallelRandomStream {
    RunMatLcgV1 {
        state: u64,
    },
    /// Compact RunMat-owned seed initialization. A worker derives the MT
    /// state only when its logical iteration begins.
    Mt19937InitialSeedV1 {
        seed: u64,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "policy", rename_all = "snake_case", deny_unknown_fields)]
pub enum ParallelRandomnessContext {
    Inherit,
    Deterministic {
        streams: Vec<ParallelRandomStream>,
    },
    /// Compact range for newly issued legacy-LCG tasks. The historical
    /// `Deterministic` vector remains accepted for replay.
    RunMatLcgStateRangeV1 {
        root: u64,
        start: usize,
        len: usize,
    },
    /// One MT initialization root for a contiguous logical-iteration span.
    /// Chunking only changes `start` and `len`, never the derived stream.
    Mt19937InitialSeedRangeV1 {
        root: u64,
        start: usize,
        len: usize,
    },
    Nondeterministic,
}

impl ParallelRandomnessContext {
    pub fn validate_for_iterations(&self, iterations: usize) -> Result<(), ContractError> {
        let valid = match self {
            Self::Deterministic { streams } => streams.len() == iterations,
            Self::RunMatLcgStateRangeV1 { start, len, .. }
            | Self::Mt19937InitialSeedRangeV1 { start, len, .. } => {
                *len == iterations
                    && start
                        .checked_add(*len)
                        .is_some_and(|end| u64::try_from(end).is_ok())
            }
            Self::Inherit | Self::Nondeterministic => true,
        };
        if !valid {
            return Err(ContractError::invalid(
                "parallel randomness context",
                "deterministic stream extent must match the logical iteration count",
            ));
        }
        Ok(())
    }

    pub fn deterministic_stream(&self, index: usize) -> Option<ParallelRandomStream> {
        match self {
            Self::Deterministic { streams } => streams.get(index).cloned(),
            Self::RunMatLcgStateRangeV1 { root, start, len } if index < *len => {
                let ordinal = u64::try_from(start.checked_add(index)?).ok()?;
                Some(ParallelRandomStream::RunMatLcgV1 {
                    state: deterministic_iteration_seed(*root, ordinal),
                })
            }
            Self::Mt19937InitialSeedRangeV1 { root, start, len } if index < *len => {
                let ordinal = u64::try_from(start.checked_add(index)?).ok()?;
                Some(ParallelRandomStream::Mt19937InitialSeedV1 {
                    seed: deterministic_iteration_seed(*root, ordinal),
                })
            }
            _ => None,
        }
    }

    pub fn is_deterministic(&self) -> bool {
        matches!(
            self,
            Self::Deterministic { .. }
                | Self::RunMatLcgStateRangeV1 { .. }
                | Self::Mt19937InitialSeedRangeV1 { .. }
        )
    }
}

/// Stable RunMat stream derivation, independent of worker or chunk ordering.
pub const RUNMAT_MT_DEFAULT_SEED: u64 = 5489;

pub fn deterministic_iteration_seed(root: u64, ordinal: u64) -> u64 {
    let seed = root ^ ordinal.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    if seed == 0 {
        return RUNMAT_MT_DEFAULT_SEED;
    }
    let mut mixed = seed.wrapping_add(0x9e3779b97f4a7c15);
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d049bb133111eb);
    let mixed = mixed ^ (mixed >> 31);
    if mixed == 0 {
        RUNMAT_MT_DEFAULT_SEED
    } else {
        mixed
    }
}

/// Dynamic execution identity for one scheduler task in a parallel region.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParallelTaskContext {
    pub region: ParallelRegionId,
    pub chunk: ParallelChunk,
    pub randomness: ParallelRandomnessContext,
}

impl ParallelTaskContext {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.chunk.len == 0 {
            return Err(ContractError::invalid(
                "parallel task context",
                "chunk length must be nonzero",
            ));
        }
        self.randomness.validate_for_iterations(self.chunk.len)?;
        Ok(())
    }
}

/// Invocation metadata interpreted by an exact program worker.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ProgramInvocationContext {
    Direct,
    ParallelTask { task: ParallelTaskContext },
    SpmdTask { task: crate::SpmdTaskContext },
}

impl Default for ProgramInvocationContext {
    fn default() -> Self {
        Self::Direct
    }
}

impl ProgramInvocationContext {
    pub fn validate_for(&self, callable: &crate::ProgramCallable) -> Result<(), ContractError> {
        match (self, callable) {
            (Self::Direct, crate::ProgramCallable::ParallelRegion { .. }) => {
                Err(ContractError::invalid(
                    "program invocation context",
                    "parallel-region callables require a parallel task context",
                ))
            }
            (Self::Direct, crate::ProgramCallable::SpmdRegion { .. }) => {
                Err(ContractError::invalid(
                    "program invocation context",
                    "SPMD-region callables require an SPMD task context",
                ))
            }
            (Self::ParallelTask { task }, crate::ProgramCallable::ParallelRegion { region }) => {
                task.validate()?;
                if task.region != *region {
                    return Err(ContractError::invalid(
                        "program invocation context",
                        "parallel task context identifies a different region",
                    ));
                }
                Ok(())
            }
            (Self::ParallelTask { .. }, _) => Err(ContractError::invalid(
                "program invocation context",
                "only parallel-region callables accept a parallel task context",
            )),
            (Self::SpmdTask { task }, crate::ProgramCallable::SpmdRegion { region }) => {
                task.validate()?;
                if task.region != *region {
                    return Err(ContractError::invalid(
                        "program invocation context",
                        "SPMD task context identifies a different region",
                    ));
                }
                Ok(())
            }
            (Self::SpmdTask { .. }, _) => Err(ContractError::invalid(
                "program invocation context",
                "only SPMD-region callables accept an SPMD task context",
            )),
            (Self::Direct, _) => Ok(()),
        }
    }
}

/// Deterministic scheduler-neutral work graph for one parallel region.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParallelTaskGraph {
    pub region: ParallelRegionId,
    pub iteration_count: usize,
    pub worker_limit: u32,
    pub chunks: Vec<ParallelChunk>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParallelChunk {
    pub ordinal: u32,
    pub start: usize,
    pub len: usize,
}

impl ParallelTaskGraph {
    /// Partitions work into several bounded tasks per admitted worker. More
    /// tasks than workers let the execution backend adapt placement to actual
    /// completion rates without making task identity nondeterministic.
    pub fn balanced(
        region: ParallelRegionId,
        iteration_count: usize,
        worker_limit: u32,
    ) -> Result<Self, ContractError> {
        if worker_limit == 0 {
            return Err(ContractError::invalid(
                "parallel task graph",
                "worker limit must be positive",
            ));
        }
        let worker_limit_usize = usize::try_from(worker_limit).map_err(|_| {
            ContractError::invalid(
                "parallel task graph",
                "worker limit exceeds this execution target",
            )
        })?;
        let target_chunks = worker_limit_usize
            .saturating_mul(4)
            .min(iteration_count)
            .min(u32::MAX as usize);
        let chunks = if target_chunks == 0 {
            Vec::new()
        } else {
            let base = iteration_count / target_chunks;
            let remainder = iteration_count % target_chunks;
            let mut start = 0usize;
            (0..target_chunks)
                .map(|ordinal| {
                    let len = base + usize::from(ordinal < remainder);
                    let chunk = ParallelChunk {
                        ordinal: ordinal as u32,
                        start,
                        len,
                    };
                    start += len;
                    chunk
                })
                .collect()
        };
        let graph = Self {
            region,
            iteration_count,
            worker_limit,
            chunks,
        };
        graph.validate()?;
        Ok(graph)
    }

    pub fn validate(&self) -> Result<(), ContractError> {
        if self.worker_limit == 0 {
            return Err(ContractError::invalid(
                "parallel task graph",
                "worker limit must be positive",
            ));
        }
        let mut cursor = 0usize;
        for (ordinal, chunk) in self.chunks.iter().enumerate() {
            let expected_ordinal = u32::try_from(ordinal).map_err(|_| {
                ContractError::invalid(
                    "parallel task graph",
                    "chunk count exceeds its portable identity range",
                )
            })?;
            if chunk.ordinal != expected_ordinal || chunk.start != cursor || chunk.len == 0 {
                return Err(ContractError::invalid(
                    "parallel task graph",
                    "chunks must be nonempty, ordered, contiguous, and canonically numbered",
                ));
            }
            cursor = cursor.checked_add(chunk.len).ok_or_else(|| {
                ContractError::invalid("parallel task graph", "chunk extent overflows")
            })?;
        }
        if cursor != self.iteration_count {
            return Err(ContractError::invalid(
                "parallel task graph",
                "chunks do not cover the exact iteration space",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use runmat_types::{ParallelRegionId, ProgramFunctionId, RegionId};

    use super::*;

    #[test]
    fn balanced_graph_is_stable_contiguous_and_bounded() {
        let region = ParallelRegionId(RegionId {
            function: ProgramFunctionId(7),
            ordinal: 2,
        });
        let graph = ParallelTaskGraph::balanced(region, 17, 3).unwrap();
        assert_eq!(graph.chunks.len(), 12);
        assert_eq!(graph.chunks.first().unwrap().start, 0);
        assert_eq!(
            graph.chunks.last().unwrap().start + graph.chunks.last().unwrap().len,
            17
        );
        graph.validate().unwrap();
    }

    #[test]
    fn parallel_task_context_binds_region_chunk_and_stream_count() {
        let region = ParallelRegionId(RegionId {
            function: ProgramFunctionId(7),
            ordinal: 2,
        });
        let task = ParallelTaskContext {
            region,
            chunk: ParallelChunk {
                ordinal: 0,
                start: 0,
                len: 2,
            },
            randomness: ParallelRandomnessContext::Deterministic {
                streams: vec![
                    ParallelRandomStream::RunMatLcgV1 { state: 1 },
                    ParallelRandomStream::RunMatLcgV1 { state: 2 },
                ],
            },
        };
        ProgramInvocationContext::ParallelTask { task }
            .validate_for(&crate::ProgramCallable::parallel_region(region))
            .unwrap();
    }

    #[test]
    fn compact_mt_initial_seed_wire_roundtrips() {
        let stream = ParallelRandomStream::Mt19937InitialSeedV1 { seed: u64::MAX };
        let encoded = serde_json::to_string(&stream).unwrap();
        assert!(encoded.len() < 96);
        assert_eq!(
            serde_json::from_str::<ParallelRandomStream>(&encoded).unwrap(),
            stream
        );
    }

    #[test]
    fn mt_seed_range_wire_and_chunking_preserve_logical_streams() {
        let whole = ParallelRandomnessContext::Mt19937InitialSeedRangeV1 {
            root: 0x1234_5678,
            start: 0,
            len: 1_000_000,
        };
        let encoded = serde_json::to_string(&whole).unwrap();
        assert!(
            encoded.len() < 128,
            "wire size must not grow with iterations"
        );
        assert_eq!(
            serde_json::from_str::<ParallelRandomnessContext>(&encoded).unwrap(),
            whole
        );
        let chunk = ParallelRandomnessContext::Mt19937InitialSeedRangeV1 {
            root: 0x1234_5678,
            start: 713,
            len: 4,
        };
        chunk.validate_for_iterations(4).unwrap();
        for index in 0..4 {
            assert_eq!(
                chunk.deterministic_stream(index),
                whole.deterministic_stream(713 + index)
            );
        }
        assert!(chunk.deterministic_stream(4).is_none());
        let overflowing = ParallelRandomnessContext::Mt19937InitialSeedRangeV1 {
            root: 1,
            start: usize::MAX,
            len: 1,
        };
        assert!(overflowing.validate_for_iterations(1).is_err());

        let legacy = ParallelRandomnessContext::RunMatLcgStateRangeV1 {
            root: 0x1234_5678,
            start: 713,
            len: 4,
        };
        legacy.validate_for_iterations(4).unwrap();
        assert!(serde_json::to_string(&legacy).unwrap().len() < 128);
        for index in 0..4 {
            assert_eq!(
                legacy.deterministic_stream(index),
                Some(ParallelRandomStream::RunMatLcgV1 {
                    state: deterministic_iteration_seed(0x1234_5678, (713 + index) as u64),
                })
            );
        }
    }
}
