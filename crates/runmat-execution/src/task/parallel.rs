use runmat_types::ParallelRegionId;
use serde::{Deserialize, Serialize};

use crate::ContractError;

/// Versioned state for one deterministic random stream assigned to one
/// logical parallel-loop iteration. The state is execution metadata, not a
/// program argument, and therefore cannot be observed or mistaken for a
/// captured workspace value.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "algorithm", rename_all = "snake_case", deny_unknown_fields)]
pub enum ParallelRandomStream {
    RunMatLcgV1 { state: u64 },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "policy", rename_all = "snake_case", deny_unknown_fields)]
pub enum ParallelRandomnessContext {
    Inherit,
    Deterministic { streams: Vec<ParallelRandomStream> },
    Nondeterministic,
}

impl ParallelRandomnessContext {
    pub fn validate_for_iterations(&self, iterations: usize) -> Result<(), ContractError> {
        if let Self::Deterministic { streams } = self {
            if streams.len() != iterations {
                return Err(ContractError::invalid(
                    "parallel randomness context",
                    "deterministic stream count must match the logical iteration count",
                ));
            }
        }
        Ok(())
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
}
