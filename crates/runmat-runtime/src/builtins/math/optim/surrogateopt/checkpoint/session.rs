//! One captured file provider/path and one invocation Clock/cadence. A saved
//! cut owns all accepted results; queued but unsubmitted proposals stay live.
use super::super::{
    controller::Controller,
    initial::{BorrowedInitial, Initial},
    notice::Frame,
    observers::Observers,
};
use super::{
    error, geometry, options::SavedOptions, record, trials, with_source, ArchivedCallbacks,
};
use crate::{
    builtins::{
        common::random,
        io::mat::records::{Policy, RecordPath, ReservedRecord},
        math::optim::common::ScalarCooperation,
        stats::random::rng::PortableSnapshot,
    },
    call::descriptor::RepeatableCallable,
    BuiltinResult,
};
use std::{path::PathBuf, time::Duration};

pub(in super::super) struct Session {
    path: RecordPath,
    reserved: Option<ReservedRecord>,
    pub(super) options: SavedOptions,
    pub(super) initial: Option<Initial>,
    pub(super) initial_rng: PortableSnapshot,
    pub(super) notices: Vec<Frame>,
    last_saved: Duration,
}

pub(super) fn path(path: PathBuf, c: &ScalarCooperation) -> BuiltinResult<RecordPath> {
    RecordPath::capture(
        path,
        c.runtime()
            .ok_or_else(|| error("checkpoint files require an original runtime"))?,
        Policy {
            identity: &super::super::IDENTITY,
            error: &runmat_builtins::SURROGATEOPT_ERROR_CHECKPOINT,
            record: "RunMatSurrogateCheckpoint",
        },
    )
}

pub(super) async fn reserve(
    path: &RecordPath,
    c: &mut ScalarCooperation,
) -> BuiltinResult<ReservedRecord> {
    let (runtime, budget) = c.runtime_budget();
    match (runtime, budget) {
        (Some(runtime), Some(budget)) => path.reserve(runtime, budget).await,
        _ => Err(error(
            "checkpoint publication lacks original cooperation authority",
        )),
    }
}

impl Session {
    pub(super) fn restored(
        path: RecordPath,
        reserved: Option<ReservedRecord>,
        options: SavedOptions,
        initial: Option<Initial>,
        initial_rng: PortableSnapshot,
        notices: Vec<Frame>,
        elapsed: Duration,
    ) -> Self {
        Self {
            path,
            reserved,
            options,
            initial,
            initial_rng,
            notices,
            last_saved: elapsed,
        }
    }
    pub(in super::super) fn enabled(&self) -> bool {
        !self.options.checkpoint().is_empty()
    }
    pub(in super::super) fn notices(&self) -> &[Frame] {
        &self.notices
    }
    pub(super) fn new(
        path: RecordPath,
        reserved: ReservedRecord,
        options: SavedOptions,
        initial: Option<Initial>,
        initial_rng: PortableSnapshot,
    ) -> Self {
        Self {
            path,
            reserved: Some(reserved),
            options,
            initial,
            initial_rng,
            notices: Vec::new(),
            last_saved: Duration::ZERO,
        }
    }
    pub(in super::super) fn record_notice(&mut self, frame: Frame) -> BuiltinResult<()> {
        if !self.enabled() {
            return Ok(());
        }
        self.notices.try_reserve(1).map_err(|source| {
            with_source("checkpoint notice metadata allocation failed", source)
        })?;
        self.notices.push(frame);
        Ok(())
    }
    pub(in super::super) fn due(&self, controller: &Controller<'_>) -> BuiltinResult<bool> {
        Ok(self.enabled()
            && controller
                .clock
                .checked_elapsed()?
                .saturating_sub(self.last_saved)
                >= Duration::from_secs(10))
    }
    pub(in super::super) async fn save(
        &mut self,
        controller: &Controller<'_>,
        objective: &RepeatableCallable,
        observers: &Observers,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        c.completed(0).await?;
        if controller.counts().0 != controller.counts().1 {
            return Err(error(
                "checkpoint publication requires a quiesced accepted-work owner",
            ));
        }
        let runtime = c
            .runtime()
            .cloned()
            .ok_or_else(|| error("checkpoint publication requires an original runtime"))?;
        let environment = runtime
            .program_revision()
            .ok_or_else(|| error("checkpoint publication lacks its original program environment"))?
            .environment();
        let frame = Frame::capture(controller)?;
        let current_rng = {
            let mut turn =
                random::acquire_original_turn(Some(&runtime), super::super::NAME).await?;
            let snapshot = random::snapshot_in_turn(&mut turn)?;
            drop(turn);
            PortableSnapshot::capture(&snapshot)
        };
        let callbacks =
            ArchivedCallbacks::capture(objective, observers.outputs(), observers.plots(), c)?;
        let record = record::Borrowed {
            version: record::VERSION,
            environment,
            options: &self.options,
            geometry: geometry::Borrowed(controller.original_domain()),
            trials: trials::Borrowed(controller.campaign.history()),
            initial: self.initial.as_ref().map(BorrowedInitial),
            initial_rng: &self.initial_rng,
            current_rng,
            frame,
            campaign: controller.campaign.checkpoint_state(),
            notices: &self.notices,
            callbacks,
        };
        let bytes = {
            let (runtime, budget) = c.runtime_budget();
            let (Some(runtime), Some(budget)) = (runtime, budget) else {
                return Err(error(
                    "checkpoint encoding lacks original cooperation authority",
                ));
            };
            let mut cooperation = super::json::Cooperation::new(runtime, budget)?;
            record
                .encode(&mut cooperation)
                .await
                .map_err(super::codec_error)?
        };
        let reserved = match self.reserved.take() {
            Some(stage) => stage,
            None => reserve(&self.path, c).await?,
        };
        let (runtime, budget) = c.runtime_budget();
        match (runtime, budget) {
            (Some(runtime), Some(budget)) => {
                self.path
                    .write_reserved(reserved, bytes, runtime, budget)
                    .await?
            }
            _ => {
                return Err(error(
                    "checkpoint publication lacks original cooperation authority",
                ))
            }
        }
        self.last_saved = frame.elapsed;
        Ok(())
    }
}
