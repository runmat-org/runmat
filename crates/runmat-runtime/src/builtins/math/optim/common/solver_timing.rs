//! User-observable solver timing belongs to the original admitted clock,
//! not the independent physical clock used for host profiling/telemetry.
use crate::context::{RuntimeContext, RuntimeTimingService};
use crate::{BuiltinResult, RuntimeError};
use std::{rc::Rc, time::Duration};

#[derive(Clone)]
pub(crate) struct SolverTiming {
    service: Rc<dyn RuntimeTimingService>,
    started: Duration,
    prior_elapsed: Duration,
}

impl SolverTiming {
    pub(crate) fn start(runtime: Option<&RuntimeContext>) -> BuiltinResult<Self> {
        let runtime = runtime.ok_or_else(Self::unavailable)?;
        let service = Rc::clone(runtime.service_ports().timing().map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })?);
        let started = service.monotonic_elapsed();
        Ok(Self {
            service,
            started,
            prior_elapsed: Duration::ZERO,
        })
    }

    pub(crate) fn started(&self) -> Duration {
        self.started
    }

    pub(crate) fn elapsed(&self) -> Duration {
        self.prior_elapsed.saturating_add(self.current_elapsed())
    }

    /// Continue one saved elapsed interval on this invocation's original
    /// timing service. No service, start point or cooperation budget is reset.
    pub(crate) fn with_prior_elapsed(mut self, elapsed: Duration) -> Self {
        self.prior_elapsed = elapsed;
        self
    }

    pub(crate) fn current_elapsed(&self) -> Duration {
        self.service
            .monotonic_elapsed()
            .saturating_sub(self.started)
    }

    /// Checkpoint consumers reject an unrepresentable accumulated duration
    /// before further source effects; saturation is not a new time grant.
    pub(crate) fn checked_elapsed(&self) -> BuiltinResult<Duration> {
        self.prior_elapsed
            .checked_add(self.current_elapsed())
            .ok_or_else(|| {
                crate::build_runtime_error("accumulated solver elapsed time is not representable")
                    .with_identifier("RunMat:Clock:ElapsedOverflow")
                    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                    .build()
            })
    }

    /// Derive an auxiliary duration origin from one read of this SAME port.
    /// The original epoch is unchanged; the returned offset is relative to it.
    pub(crate) fn subsearch_view(&self) -> (Duration, Self) {
        let observed = self.service.monotonic_elapsed();
        let offset = self
            .prior_elapsed
            .saturating_add(observed.saturating_sub(self.started));
        let view = Self {
            service: Rc::clone(&self.service),
            started: observed.max(self.started),
            prior_elapsed: Duration::ZERO,
        };
        (offset, view)
    }

    pub(crate) fn unavailable() -> RuntimeError {
        crate::build_runtime_error(
            "Clock is unavailable without an admitted invocation timing service",
        )
        .with_identifier("RunMat:Clock:ServiceUnavailable")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
    }
}
