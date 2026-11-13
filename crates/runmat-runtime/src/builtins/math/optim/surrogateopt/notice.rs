//! Borrowed observer projections. Replay carries only actual small notice
//! metadata; every x/F/constraint value remains in one measured History.

use super::{
    controller::Controller,
    history::History,
    incumbent::{State, Tracker},
    invalid,
};
use crate::BuiltinResult;
use std::time::Duration;

#[derive(Clone, Copy, serde::Serialize, serde::Deserialize, Eq, PartialEq, Ord, PartialOrd)]
pub(super) struct PointEvaluations(pub usize);
#[derive(Clone, Copy, serde::Serialize, serde::Deserialize, Eq, PartialEq, Ord, PartialOrd)]
pub(super) struct UpdateOrdinal(pub usize);

#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Frame {
    pub rows: usize,
    pub evaluations: PointEvaluations,
    pub updates: UpdateOrdinal,
    pub elapsed: Duration,
    pub resets: u64,
    pub reset: bool,
    pub resume: bool,
    pub tracker: State,
}

impl Frame {
    pub(super) fn capture(controller: &Controller<'_>) -> BuiltinResult<Self> {
        Ok(Self {
            rows: controller.campaign.history().len(),
            evaluations: PointEvaluations(controller.counts().1),
            updates: UpdateOrdinal(controller.iteration),
            elapsed: controller.clock.checked_elapsed()?,
            resets: controller.campaign.phase().0,
            reset: controller.reset_since_notice(),
            resume: false,
            tracker: controller.campaign.tracker().state(),
        })
    }

    /// Constant-work carrier checks. Restored chronology and derived ranks
    /// are validated once by incumbent::Validation before notice replay.
    pub(super) fn validate_shape(&self, history: &History) -> BuiltinResult<()> {
        let committed = self
            .tracker
            .current
            .map_or(0, |index| index.0.saturating_add(1));
        if committed > self.rows || self.rows > history.len() {
            return Err(invalid(
                "saved observer prefix exceeds its measured history",
            ));
        }
        Tracker::validate_state_shape(self.tracker, history, committed)?;
        if self.evaluations.0 > self.rows {
            return Err(invalid(
                "saved point-evaluation count exceeds its actual trial prefix",
            ));
        }
        Ok(())
    }
}

pub(super) struct Notice<'a> {
    pub history: &'a History,
    pub frame: Frame,
    pub tolerance: f64,
    pub replay: bool,
    pub resume: bool,
}
impl<'a> Notice<'a> {
    pub(super) fn live(controller: &'a Controller<'_>) -> BuiltinResult<Self> {
        Ok(Self {
            history: controller.campaign.history(),
            frame: Frame::capture(controller)?,
            tolerance: controller.constraint_tolerance(),
            replay: false,
            resume: false,
        })
    }
    pub(super) fn replay(
        history: &'a History,
        frame: Frame,
        tolerance: f64,
    ) -> BuiltinResult<Self> {
        frame.validate_shape(history)?;
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(invalid("invalid replay constraint tolerance"));
        }
        Ok(Self {
            history,
            frame,
            tolerance,
            replay: true,
            resume: frame.resume,
        })
    }
}

pub(super) struct Observed {
    pub stop: bool,
    pub frame: Frame,
}
