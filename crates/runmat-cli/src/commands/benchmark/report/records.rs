use super::{identity, profile, provider, Phase};
use runmat_core::{abi::RuntimeFlow, RunMatSession};
use serde::Serialize;

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Record {
    Begin,
    Identity,
    Sample,
    Terminal,
}
#[derive(Serialize)]
pub(super) struct Begin {
    pub(super) record: Record,
    pub(super) schema_version: u32,
}
#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Boundary {
    CommandReturnIncludingCleanup,
}
#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Validation {
    WorkloadResponsibility,
}
#[derive(Serialize)]
pub(super) struct Header {
    pub(super) record: Record,
    pub(super) source: identity::Identity,
    pub(super) executable: identity::Identity,
    pub(super) jit_allowed: bool,
    pub(super) requested_jit_threshold: u32,
    pub(super) warmup: u32,
    pub(super) iterations: u32,
    pub(super) requested_acceleration_enabled: bool,
    pub(super) requested_provider: runmat_config::runtime::AccelerateProviderPreference,
    pub(super) requested_inprocess_fallback_allowed: bool,
    pub(super) boundary: Boundary,
    pub(super) host_result_validation: Validation,
}
#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Status {
    ReturnedWithoutErrors,
    LanguageError,
    FrontendError,
}
#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(super) enum ResultBoundary<'a> {
    NoValue,
    SingleHostNumeric {
        dtype: runmat_value::NumericDType,
        shape: &'a [usize],
    },
    SingleResident,
    SingleOther,
    OutputList {
        len: usize,
    },
    DynamicList,
}
pub(super) fn result(flow: &RuntimeFlow) -> ResultBoundary<'_> {
    match flow {
        RuntimeFlow::NoValue => ResultBoundary::NoValue,
        RuntimeFlow::Single(runmat_value::Value::Tensor(t)) => ResultBoundary::SingleHostNumeric {
            dtype: t.numeric_dtype(),
            shape: t.shape(),
        },
        RuntimeFlow::Single(runmat_value::Value::ComplexTensor(t)) => {
            ResultBoundary::SingleHostNumeric {
                dtype: t.numeric_dtype(),
                shape: t.shape(),
            }
        }
        RuntimeFlow::Single(runmat_value::Value::GpuTensor(_)) => ResultBoundary::SingleResident,
        RuntimeFlow::Single(_) => ResultBoundary::SingleOther,
        RuntimeFlow::OutputList(values) => ResultBoundary::OutputList { len: values.len() },
        RuntimeFlow::DynamicList(_) => ResultBoundary::DynamicList,
    }
}
#[derive(Serialize)]
pub(super) struct Sample<'a> {
    pub(super) record: Record,
    pub(super) ordinal: u64,
    pub(super) phase: Phase,
    pub(super) elapsed_ns: String,
    pub(super) status: Status,
    pub(super) used_jit: Option<bool>,
    pub(super) result: Option<ResultBoundary<'a>>,
    pub(super) feedback: profile::Delta<'a>,
    pub(super) stats: StatsDelta,
    pub(super) transfers: provider::Delta,
    pub(super) command: Option<Command>,
    pub(super) profiling: Option<Summary>,
    pub(super) dependency_lock: Option<identity::Identity>,
}
#[derive(Clone, Copy)]
pub(super) struct Stats {
    pub(super) total: usize,
    pub(super) jit: usize,
    pub(super) interpreter: usize,
    pub(super) osr: usize,
    pub(super) vectorized: u64,
}
impl Stats {
    pub(super) fn capture(engine: &RunMatSession) -> Self {
        let s = engine.stats();
        Self {
            total: s.total_executions,
            jit: s.jit_compiled,
            interpreter: s.interpreter_fallback,
            osr: s.native_osr_transfers,
            vectorized: s.vectorized_native_regions,
        }
    }
    pub(super) fn delta(self, before: Self) -> StatsDelta {
        StatsDelta {
            total: self.total.checked_sub(before.total),
            jit: self.jit.checked_sub(before.jit),
            interpreter: self.interpreter.checked_sub(before.interpreter),
            osr: self.osr.checked_sub(before.osr),
            vectorized: self.vectorized.checked_sub(before.vectorized),
        }
    }
}
#[derive(Serialize)]
pub(super) struct StatsDelta {
    pub(super) total: Option<usize>,
    pub(super) jit: Option<usize>,
    pub(super) interpreter: Option<usize>,
    pub(super) osr: Option<usize>,
    pub(super) vectorized: Option<u64>,
}
#[derive(Serialize)]
pub(super) struct PhysicalCounts {
    pub(super) pending: u64,
    pub(super) completed: u64,
    pub(super) failed: u64,
    pub(super) missing: u64,
    pub(super) incomplete: u64,
}
#[derive(Serialize)]
pub(super) struct DeferredCounts {
    pub(super) pending: u64,
    pub(super) completed: u64,
    pub(super) failed: u64,
    pub(super) missing: u64,
    pub(super) admitted: u64,
}
#[derive(Serialize)]
pub(super) struct Summary {
    pub(super) cpu_ms: Option<u64>,
    pub(super) gpu_ms: Option<u64>,
    pub(super) kernel_count: Option<u32>,
}
#[derive(Serialize)]
pub(super) struct Command {
    pub(super) physical: Option<PhysicalCounts>,
    pub(super) deferred: Option<DeferredCounts>,
    pub(super) coverage: Coverage,
    pub(super) canonical_device_route_admitted: bool,
    pub(super) unproven_device_route: bool,
    pub(super) unproven_device_time: bool,
}
#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Coverage {
    Incomplete,
}
impl From<runmat_core::CommandExecutionEvidence> for Command {
    fn from(e: runmat_core::CommandExecutionEvidence) -> Self {
        let coverage = match e.route_coverage {
            runmat_execution::profiling::CommandRouteCoverage::Incomplete => Coverage::Incomplete,
        };
        Self {
            physical: e.physical_submissions.map(|c| PhysicalCounts {
                pending: c.pending,
                completed: c.completed,
                failed: c.failed,
                missing: c.missing,
                incomplete: c.incomplete,
            }),
            deferred: e.deferred_work.map(|c| DeferredCounts {
                pending: c.pending,
                completed: c.completed,
                failed: c.failed,
                missing: c.missing,
                admitted: c.admitted,
            }),
            coverage,
            canonical_device_route_admitted: e.canonical_device_route_admitted,
            unproven_device_route: e.unproven_device_route,
            unproven_device_time: e.unproven_device_time,
        }
    }
}
#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum TerminalStatus {
    Completed,
    FailedOrAbandoned,
}
#[derive(Serialize)]
pub(super) struct Terminal {
    pub(super) record: Record,
    pub(super) status: TerminalStatus,
    pub(super) samples: u64,
}
