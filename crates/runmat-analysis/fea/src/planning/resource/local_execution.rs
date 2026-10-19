use serde::{Deserialize, Serialize};

use super::ByteCount;

/// Maximum local thread count admitted by the FEA shared-memory contract.
///
/// This is a contract bound, not a host-discovery result. Callers must derive the
/// requested count from an admitted resource allocation before constructing a plan.
pub const MAX_LOCAL_EXECUTION_THREADS: u16 =
    runmat_analysis_core::MAX_CPU_THREADS_PER_WORKER as u16;

/// CPU budget represented by one fully admitted local execution thread.
pub const CPU_MILLICORES_PER_LOCAL_THREAD: u32 = 1_000;

#[derive(Clone, Copy, Debug, thiserror::Error, Eq, PartialEq)]
pub enum LocalExecutionPlanError {
    #[error("invalid local execution plan: {0}")]
    Invalid(&'static str),
    #[error("local execution plan arithmetic overflow: {0}")]
    Overflow(&'static str),
}

/// A positive, contract-bounded local thread count.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(try_from = "u16", into = "u16")]
pub struct LocalThreadCount(u16);

impl LocalThreadCount {
    pub fn new(value: u16) -> Result<Self, LocalExecutionPlanError> {
        if value == 0 {
            return Err(LocalExecutionPlanError::Invalid(
                "local thread count must be positive",
            ));
        }
        if value > MAX_LOCAL_EXECUTION_THREADS {
            return Err(LocalExecutionPlanError::Invalid(
                "local thread count exceeds the contract maximum",
            ));
        }
        Ok(Self(value))
    }

    /// Derives local concurrency from an exact whole-core admitted CPU budget.
    ///
    /// Fractional-core allocations are rejected because silently rounding them would make
    /// the numerical execution authority differ from the scheduled resource authority.
    pub fn from_cpu_millicores(cpu_millicores: u32) -> Result<Self, LocalExecutionPlanError> {
        if cpu_millicores == 0 {
            return Err(LocalExecutionPlanError::Invalid(
                "CPU millicores must be positive",
            ));
        }
        if !cpu_millicores.is_multiple_of(CPU_MILLICORES_PER_LOCAL_THREAD) {
            return Err(LocalExecutionPlanError::Invalid(
                "CPU millicores must encode an exact whole-thread allocation",
            ));
        }
        let thread_count = cpu_millicores / CPU_MILLICORES_PER_LOCAL_THREAD;
        let thread_count = u16::try_from(thread_count).map_err(|_| {
            LocalExecutionPlanError::Invalid("CPU millicores exceed the local thread-count bound")
        })?;
        Self::new(thread_count)
    }

    pub const fn get(self) -> u16 {
        self.0
    }

    /// Returns the total CPU allocation required for this many threads at the
    /// supplied positive per-thread rate.
    pub fn checked_cpu_millicores(
        self,
        millicores_per_thread: u32,
    ) -> Result<u32, LocalExecutionPlanError> {
        if millicores_per_thread == 0 {
            return Err(LocalExecutionPlanError::Invalid(
                "CPU millicores per thread must be positive",
            ));
        }
        u32::from(self.0)
            .checked_mul(millicores_per_thread)
            .ok_or(LocalExecutionPlanError::Overflow("local CPU millicores"))
    }

    /// Returns the total fixed scratch reservation for this many threads.
    pub fn checked_fixed_scratch_bytes(
        self,
        bytes_per_thread: ByteCount,
    ) -> Result<ByteCount, LocalExecutionPlanError> {
        if bytes_per_thread.0 == 0 {
            return Err(LocalExecutionPlanError::Invalid(
                "fixed scratch bytes per thread must be positive",
            ));
        }
        bytes_per_thread
            .0
            .checked_mul(u64::from(self.0))
            .map(ByteCount)
            .ok_or(LocalExecutionPlanError::Overflow(
                "local fixed scratch bytes",
            ))
    }
}

impl TryFrom<u16> for LocalThreadCount {
    type Error = LocalExecutionPlanError;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<LocalThreadCount> for u16 {
    fn from(value: LocalThreadCount) -> Self {
        value.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum LocalExecutionMode {
    Serial,
    Fixed { thread_count: LocalThreadCount },
}

/// Immutable local shared-memory execution authority.
///
/// The plan deliberately contains no host-discovery or global-pool behavior. A one-thread
/// plan has exactly one canonical encoding (`serial`); fixed plans always contain at least
/// two threads.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(try_from = "LocalExecutionMode", into = "LocalExecutionMode")]
pub struct LocalExecutionPlan {
    mode: LocalExecutionMode,
}

impl LocalExecutionPlan {
    pub const fn serial() -> Self {
        Self {
            mode: LocalExecutionMode::Serial,
        }
    }

    pub fn fixed(thread_count: LocalThreadCount) -> Result<Self, LocalExecutionPlanError> {
        if thread_count.get() == 1 {
            return Err(LocalExecutionPlanError::Invalid(
                "one local thread must use the serial plan",
            ));
        }
        Ok(Self {
            mode: LocalExecutionMode::Fixed { thread_count },
        })
    }

    pub fn from_thread_count(
        thread_count: LocalThreadCount,
    ) -> Result<Self, LocalExecutionPlanError> {
        if thread_count.get() == 1 {
            Ok(Self::serial())
        } else {
            Self::fixed(thread_count)
        }
    }

    pub fn from_cpu_millicores(cpu_millicores: u32) -> Result<Self, LocalExecutionPlanError> {
        Self::from_thread_count(LocalThreadCount::from_cpu_millicores(cpu_millicores)?)
    }

    pub const fn thread_count(self) -> LocalThreadCount {
        match self.mode {
            LocalExecutionMode::Serial => LocalThreadCount(1),
            LocalExecutionMode::Fixed { thread_count } => thread_count,
        }
    }

    pub const fn is_serial(self) -> bool {
        matches!(self.mode, LocalExecutionMode::Serial)
    }

    pub fn checked_cpu_millicores(self) -> Result<u32, LocalExecutionPlanError> {
        self.thread_count()
            .checked_cpu_millicores(CPU_MILLICORES_PER_LOCAL_THREAD)
    }

    pub fn checked_fixed_scratch_bytes(
        self,
        bytes_per_thread: ByteCount,
    ) -> Result<ByteCount, LocalExecutionPlanError> {
        self.thread_count()
            .checked_fixed_scratch_bytes(bytes_per_thread)
    }

    /// Derives deterministic, nonempty, contiguous chunks covering `item_count`.
    ///
    /// Extra requested threads remain inactive when there are fewer items than threads.
    /// Remainder items are assigned to the lowest chunk ordinals, yielding one canonical
    /// partition for a given plan and item count.
    pub fn chunks(self, item_count: u64) -> Result<Vec<LocalWorkChunk>, LocalExecutionPlanError> {
        if item_count == 0 {
            return Err(LocalExecutionPlanError::Invalid(
                "local work item count must be positive",
            ));
        }

        let active_count_u64 = u64::from(self.thread_count().get()).min(item_count);
        let active_count = u16::try_from(active_count_u64)
            .map_err(|_| LocalExecutionPlanError::Overflow("active local execution chunk count"))?;
        let base_len = item_count / active_count_u64;
        let remainder = item_count % active_count_u64;
        let mut chunks = Vec::with_capacity(usize::from(active_count));
        let mut start = 0_u64;

        for ordinal in 0..active_count {
            let extra = u64::from(ordinal) < remainder;
            let len = base_len
                .checked_add(u64::from(extra))
                .ok_or(LocalExecutionPlanError::Overflow("local chunk length"))?;
            let end_exclusive = start
                .checked_add(len)
                .ok_or(LocalExecutionPlanError::Overflow("local chunk end"))?;
            chunks.push(LocalWorkChunk::new(ordinal, start, end_exclusive)?);
            start = end_exclusive;
        }

        if start != item_count {
            return Err(LocalExecutionPlanError::Invalid(
                "local chunks do not cover the requested item count",
            ));
        }
        Ok(chunks)
    }
}

impl TryFrom<LocalExecutionMode> for LocalExecutionPlan {
    type Error = LocalExecutionPlanError;

    fn try_from(mode: LocalExecutionMode) -> Result<Self, Self::Error> {
        match mode {
            LocalExecutionMode::Serial => Ok(Self::serial()),
            LocalExecutionMode::Fixed { thread_count } => Self::fixed(thread_count),
        }
    }
}

impl From<LocalExecutionPlan> for LocalExecutionMode {
    fn from(value: LocalExecutionPlan) -> Self {
        value.mode
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(try_from = "LocalWorkChunkData", into = "LocalWorkChunkData")]
pub struct LocalWorkChunk {
    ordinal: u16,
    start: u64,
    end_exclusive: u64,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LocalWorkChunkData {
    ordinal: u16,
    start: u64,
    end_exclusive: u64,
}

impl LocalWorkChunk {
    fn new(ordinal: u16, start: u64, end_exclusive: u64) -> Result<Self, LocalExecutionPlanError> {
        if ordinal >= MAX_LOCAL_EXECUTION_THREADS {
            return Err(LocalExecutionPlanError::Invalid(
                "local chunk ordinal exceeds the contract maximum",
            ));
        }
        if start >= end_exclusive {
            return Err(LocalExecutionPlanError::Invalid(
                "local chunk must contain at least one item",
            ));
        }
        Ok(Self {
            ordinal,
            start,
            end_exclusive,
        })
    }

    pub const fn ordinal(self) -> u16 {
        self.ordinal
    }

    pub const fn start(self) -> u64 {
        self.start
    }

    pub const fn end_exclusive(self) -> u64 {
        self.end_exclusive
    }

    pub const fn len(self) -> u64 {
        self.end_exclusive - self.start
    }

    pub const fn is_empty(self) -> bool {
        false
    }
}

impl TryFrom<LocalWorkChunkData> for LocalWorkChunk {
    type Error = LocalExecutionPlanError;

    fn try_from(value: LocalWorkChunkData) -> Result<Self, Self::Error> {
        Self::new(value.ordinal, value.start, value.end_exclusive)
    }
}

impl From<LocalWorkChunk> for LocalWorkChunkData {
    fn from(value: LocalWorkChunk) -> Self {
        Self {
            ordinal: value.ordinal,
            start: value.start,
            end_exclusive: value.end_exclusive,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};

    const TEST_LIMITS: CanonicalLimits = CanonicalLimits::new(4_096, 64, 512, 16);

    #[test]
    fn thread_counts_are_positive_bounded_and_canonical() {
        assert_eq!(
            LocalThreadCount::new(0),
            Err(LocalExecutionPlanError::Invalid(
                "local thread count must be positive"
            ))
        );
        assert_eq!(
            LocalThreadCount::new(MAX_LOCAL_EXECUTION_THREADS + 1),
            Err(LocalExecutionPlanError::Invalid(
                "local thread count exceeds the contract maximum"
            ))
        );

        let one = LocalThreadCount::new(1).unwrap();
        assert_eq!(
            LocalExecutionPlan::from_thread_count(one).unwrap(),
            LocalExecutionPlan::serial()
        );
        assert_eq!(
            LocalExecutionPlan::fixed(one),
            Err(LocalExecutionPlanError::Invalid(
                "one local thread must use the serial plan"
            ))
        );

        let encoded = serde_json::to_value(LocalExecutionPlan::serial()).unwrap();
        assert_eq!(encoded, serde_json::json!("serial"));
        let noncanonical = serde_json::json!({ "fixed": { "thread_count": 1 } });
        assert!(serde_json::from_value::<LocalExecutionPlan>(noncanonical).is_err());
    }

    #[test]
    fn cpu_budget_derivation_is_exact_and_bounded() {
        assert!(LocalExecutionPlan::from_cpu_millicores(0).is_err());
        assert!(LocalExecutionPlan::from_cpu_millicores(999).is_err());
        assert!(LocalExecutionPlan::from_cpu_millicores(1_999).is_err());
        assert_eq!(
            LocalExecutionPlan::from_cpu_millicores(1_000).unwrap(),
            LocalExecutionPlan::serial()
        );
        assert_eq!(
            LocalExecutionPlan::from_cpu_millicores(2_000)
                .unwrap()
                .thread_count()
                .get(),
            2
        );
        assert!(LocalExecutionPlan::from_cpu_millicores(
            u32::from(MAX_LOCAL_EXECUTION_THREADS + 1) * CPU_MILLICORES_PER_LOCAL_THREAD
        )
        .is_err());
    }

    #[test]
    fn deterministic_chunks_are_contiguous_balanced_and_complete() {
        let plan = LocalExecutionPlan::fixed(LocalThreadCount::new(3).unwrap()).unwrap();
        let expected = vec![(0, 0, 4), (1, 4, 7), (2, 7, 10)];

        for _ in 0..8 {
            let chunks = plan.chunks(10).unwrap();
            let actual = chunks
                .iter()
                .map(|chunk| (chunk.ordinal(), chunk.start(), chunk.end_exclusive()))
                .collect::<Vec<_>>();
            assert_eq!(actual, expected);
            assert_eq!(chunks.iter().map(|chunk| chunk.len()).sum::<u64>(), 10);
            assert!(chunks.iter().all(|chunk| !chunk.is_empty()));
        }
    }

    #[test]
    fn surplus_threads_do_not_create_empty_chunks() {
        let plan = LocalExecutionPlan::fixed(LocalThreadCount::new(8).unwrap()).unwrap();
        let chunks = plan.chunks(3).unwrap();
        let actual = chunks
            .iter()
            .map(|chunk| (chunk.ordinal(), chunk.start(), chunk.end_exclusive()))
            .collect::<Vec<_>>();
        assert_eq!(actual, vec![(0, 0, 1), (1, 1, 2), (2, 2, 3)]);
        assert_eq!(plan.thread_count().get(), 8);
    }

    #[test]
    fn zero_work_and_invalid_serialized_chunks_fail_closed() {
        assert!(LocalExecutionPlan::serial().chunks(0).is_err());
        assert!(serde_json::from_value::<LocalWorkChunk>(serde_json::json!({
            "ordinal": 0,
            "start": 2,
            "end_exclusive": 2
        }))
        .is_err());
        assert!(serde_json::from_value::<LocalWorkChunk>(serde_json::json!({
            "ordinal": MAX_LOCAL_EXECUTION_THREADS,
            "start": 0,
            "end_exclusive": 1
        }))
        .is_err());
    }

    #[test]
    fn cpu_and_fixed_scratch_accounting_are_checked() {
        let two = LocalThreadCount::new(2).unwrap();
        assert_eq!(two.checked_cpu_millicores(1_000).unwrap(), 2_000);
        assert_eq!(
            two.checked_fixed_scratch_bytes(ByteCount(4_096)).unwrap(),
            ByteCount(8_192)
        );
        assert!(two.checked_cpu_millicores(0).is_err());
        assert!(two.checked_cpu_millicores(u32::MAX).is_err());
        assert!(two.checked_fixed_scratch_bytes(ByteCount(0)).is_err());
        assert!(two
            .checked_fixed_scratch_bytes(ByteCount(u64::MAX))
            .is_err());
    }

    #[test]
    fn fixed_plan_and_chunks_have_stable_serde_shapes() {
        let plan = LocalExecutionPlan::fixed(LocalThreadCount::new(3).unwrap()).unwrap();
        let encoded = serde_json::to_value(plan).unwrap();
        assert_eq!(
            encoded,
            serde_json::json!({ "fixed": { "thread_count": 3 } })
        );
        assert_eq!(
            serde_json::from_value::<LocalExecutionPlan>(encoded).unwrap(),
            plan
        );

        let chunk = plan.chunks(5).unwrap()[0];
        let encoded = serde_json::to_value(chunk).unwrap();
        assert_eq!(
            encoded,
            serde_json::json!({ "ordinal": 0, "start": 0, "end_exclusive": 2 })
        );
        assert_eq!(
            serde_json::from_value::<LocalWorkChunk>(encoded).unwrap(),
            chunk
        );
    }

    #[test]
    fn equivalent_construction_has_identical_canonical_bytes() {
        let from_threads = LocalExecutionPlan::fixed(LocalThreadCount::new(3).unwrap()).unwrap();
        let from_cpu = LocalExecutionPlan::from_cpu_millicores(3_000).unwrap();
        assert_eq!(from_threads, from_cpu);

        let first = encode_contract(
            b"runmat-fea-local-execution-test/v1\0",
            "plan",
            &from_threads,
            TEST_LIMITS,
        )
        .unwrap();
        let second = encode_contract(
            b"runmat-fea-local-execution-test/v1\0",
            "plan",
            &from_cpu,
            TEST_LIMITS,
        )
        .unwrap();
        assert_eq!(first, second);

        let decoded: LocalExecutionPlan = decode_contract(
            b"runmat-fea-local-execution-test/v1\0",
            "plan",
            &first,
            TEST_LIMITS,
        )
        .unwrap();
        assert_eq!(decoded, from_threads);
    }
}
