//! Local streamed measurements of the existing command-return/cleanup boundary.
//! Neither this report nor provider-wide counters grant execution authority.
mod identity;
mod profile;
mod provider;
mod records;
use records::*;
#[cfg(test)]
mod tests;

use anyhow::{Context, Result};
use runmat_accelerate::placement::PlacementProfileSnapshot;
use runmat_config::runtime::RunMatRuntimeConfig;
use runmat_core::{
    abi::{ExecutionResponse, SourceInput},
    RunMatSession,
};
use serde::Serialize;
use std::{
    fs::{File, OpenOptions},
    io::Write,
    path::Path,
    time::Instant,
};

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Phase {
    ColdFirst,
    Warmup,
    Warm,
}

pub(super) struct Report {
    file: File,
    finished: bool,
    samples: u64,
}

impl Report {
    pub(super) fn create(path: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .with_context(|| format!("create new benchmark report {}", path.display()))?;
        let mut report = Self {
            file,
            finished: false,
            samples: 0,
        };
        report.write(&Begin {
            record: Record::Begin,
            schema_version: 1,
        })?;
        Ok(report)
    }
    pub(super) fn header(
        &mut self,
        source: &SourceInput,
        jit: bool,
        warmup: u32,
        iterations: u32,
        config: &RunMatRuntimeConfig,
    ) -> Result<()> {
        self.write(&Header {
            record: Record::Identity,
            source: identity::source(source),
            executable: identity::executable(),
            jit_allowed: jit,
            requested_jit_threshold: config.jit.threshold,
            warmup,
            iterations,
            requested_acceleration_enabled: config.accelerate.enabled,
            requested_provider: config.accelerate.provider,
            requested_inprocess_fallback_allowed: config.accelerate.allow_inprocess_fallback,
            boundary: Boundary::CommandReturnIncludingCleanup,
            host_result_validation: Validation::WorkloadResponsibility,
        })
    }
    fn write(&mut self, value: &impl Serialize) -> Result<()> {
        serde_json::to_writer(&mut self.file, value).context("write benchmark JSONL record")?;
        self.file.write_all(b"\n")?;
        self.file.flush()?;
        Ok(())
    }
    pub(super) fn sample(
        &mut self,
        start: SampleStart,
        engine: &RunMatSession,
        response: &ExecutionResponse,
    ) -> Result<()> {
        // Stop first: post-call snapshots, validation and writes are untimed.
        let elapsed_ns = start.started.elapsed().as_nanos().to_string();
        let after = engine.placement_profile_snapshot();
        let outcome = response.result.as_ref().ok().map(|root| root.value());
        let status = match outcome {
            Some(outcome) if !outcome.has_errors() => Status::ReturnedWithoutErrors,
            Some(_) => Status::LanguageError,
            None => Status::FrontendError,
        };
        let command = outcome
            .and_then(|outcome| outcome.profiling.as_ref())
            .and_then(|p| p.command_evidence);
        self.write(&Sample {
            record: Record::Sample,
            ordinal: self.samples,
            phase: start.phase,
            elapsed_ns,
            status,
            used_jit: outcome.map(|o| o.used_jit),
            result: outcome.map(|o| result(&o.flow)),
            feedback: profile::delta(&start.profile, &after),
            stats: Stats::capture(engine).delta(start.stats),
            transfers: provider::delta(start.provider, provider::snapshot()),
            command: command.map(Command::from),
            profiling: outcome.and_then(|o| o.profiling.as_ref()).map(|p| Summary {
                cpu_ms: p.cpu_ms,
                gpu_ms: p.gpu_ms,
                kernel_count: p.kernel_count,
            }),
            dependency_lock: response.dependency_lock.as_deref().map(identity::content),
        })?;
        self.samples = self
            .samples
            .checked_add(1)
            .context("benchmark sample ordinal overflow")?;
        Ok(())
    }
    pub(super) fn finish(&mut self) -> Result<()> {
        self.write(&Terminal {
            record: Record::Terminal,
            status: TerminalStatus::Completed,
            samples: self.samples,
        })?;
        self.finished = true;
        Ok(())
    }
}

impl Drop for Report {
    fn drop(&mut self) {
        if !self.finished {
            // Preserve partial rows on error/cancellation. Failed I/O or abrupt
            // process death may prevent a terminal row; absence is not success.
            let _ = self.write(&Terminal {
                record: Record::Terminal,
                status: TerminalStatus::FailedOrAbandoned,
                samples: self.samples,
            });
        }
    }
}

pub(super) struct SampleStart {
    phase: Phase,
    profile: PlacementProfileSnapshot,
    provider: Option<provider::Snapshot>,
    stats: Stats,
    started: Instant,
}
impl SampleStart {
    pub(super) fn capture(engine: &RunMatSession, phase: Phase) -> Self {
        let profile = engine.placement_profile_snapshot();
        let provider = provider::snapshot();
        let stats = Stats::capture(engine);
        Self {
            phase,
            profile,
            provider,
            stats,
            started: Instant::now(),
        }
    }
}
