//! The usage-metering seam for constraint solves and regenerations.
//!
//! RunMat meters cluster work through one authority, `runmat-simulation-usage`. This crate
//! only reports what it did: each sketch solve, each assembly mate solve, each feature
//! regeneration and each part regeneration becomes a [`UsageEvent`] handed to a
//! [`UsageMeter`]. The real meter plugs in through an adapter; this crate ships no meter of
//! its own beyond [`NoUsageMeter`], the default, which records nothing.
//!
//! Durations are reported as provided: this crate never reads a clock by itself (the design
//! layer runs in WebAssembly, where `std::time::Instant` is unavailable, and its results must
//! not depend on time). A meter that wants durations supplies a [`MeterClock`]; call sites read
//! it before and after the work and report the difference. With no clock, every duration is
//! `None` and no time is read.
//!
//! Events carry counts and sizes (entities, constraints, bodies, iterations, features) so a
//! meter can price work without access to the model. Events are side-channel only: nothing
//! the crate computes depends on what a meter does with them.

use crate::ids::{FeatureId, FilePath};
use serde::{Deserialize, Serialize};

/// A monotonic clock a meter supplies. Nanoseconds from an arbitrary origin.
pub trait MeterClock {
    fn now_nanos(&self) -> u64;
}

/// Receives usage events. Implemented by the adapter to `runmat-simulation-usage`.
pub trait UsageMeter {
    fn record(&self, event: &UsageEvent);
    /// The clock durations are measured with; `None` (the default) reports no durations.
    fn clock(&self) -> Option<&dyn MeterClock> {
        None
    }
}

/// Records nothing and reads no clock. The default wherever no meter is configured.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoUsageMeter;

impl UsageMeter for NoUsageMeter {
    fn record(&self, _event: &UsageEvent) {}
}

/// A started measurement: the clock reading at the start, if the meter has a clock.
#[derive(Clone, Copy, Debug)]
pub struct Stopwatch(Option<u64>);

impl Stopwatch {
    pub fn start(meter: &dyn UsageMeter) -> Self {
        Stopwatch(meter.clock().map(|c| c.now_nanos()))
    }
    /// Nanoseconds since [`Stopwatch::start`], or `None` without a clock.
    pub fn elapsed(&self, meter: &dyn UsageMeter) -> Option<u64> {
        let start = self.0?;
        let now = meter.clock()?.now_nanos();
        Some(now.saturating_sub(start))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SolveOutcome {
    /// Solved and fully determined.
    WellDefined,
    /// Solved, with remaining freedom.
    UnderDefined,
    /// Solved, with redundant or conflicting constraints reported.
    OverDefined,
    /// Coordinates taken as written; no numeric solve ran.
    Explicit,
    /// The solve did not converge or could not be built.
    Failed,
}

/// How one feature's regeneration ended, as metered.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureWork {
    /// The kernel (or the solver, for a sketch) ran.
    Computed,
    /// The result came from the regeneration cache; the kernel did not run.
    CacheHit,
    /// Frozen geometry was reused.
    Frozen,
    /// Compiled or ran and failed.
    Failed,
    /// Not run: suppressed, rolled back, waiting or blocked.
    Skipped,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "event")]
pub enum UsageEvent {
    /// One sketch solve during regeneration.
    SketchSolve {
        file: FilePath,
        feature: FeatureId,
        entities: u32,
        constraints: u32,
        /// Remaining DOF, where the solver reports it.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        dof: Option<u32>,
        outcome: SolveOutcome,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        duration_nanos: Option<u64>,
    },
    /// One mate-solve pass of an assembly scope (an assembly level plus its flexible members).
    MateSolve {
        /// The assembly file that owns the scope.
        assembly: FilePath,
        /// The scope root's display path (empty for the top).
        scope: String,
        /// 1 for the first pass, 2 for the pass over mates touching pattern instances.
        pass: u32,
        bodies: u32,
        grounded: u32,
        mates: u32,
        variables: u32,
        iterations: u32,
        converged: bool,
        dof: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        duration_nanos: Option<u64>,
    },
    /// One feature of one part regeneration.
    FeatureRegeneration {
        file: FilePath,
        feature: FeatureId,
        kind: String,
        work: FeatureWork,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        duration_nanos: Option<u64>,
    },
    /// One whole part regeneration.
    PartRegeneration {
        file: FilePath,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        configuration: Option<String>,
        features: u32,
        computed: u32,
        cache_hits: u32,
        failed: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        duration_nanos: Option<u64>,
    },
}
