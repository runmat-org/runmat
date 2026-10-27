//! Linear-time validation of committed chronological checkpoint prefixes.
//! Reuse the live ranking transitions; saved indices never nominate a best.
use super::*;

#[derive(Default)]
pub(in super::super) struct Validation {
    tracker: Tracker,
    rows: usize,
    phase: u64,
}
impl Validation {
    pub(in super::super) async fn advance(
        &mut self,
        expected: State,
        history: &History,
        rows: usize,
        phase: u64,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        Tracker::validate_state_shape(expected, history, rows)?;
        if rows < self.rows
            || phase < self.phase
            || phase == self.phase && expected.incumbent_start != self.tracker.incumbent_start
            || phase > self.phase && expected.incumbent_start < self.rows
        {
            return Err(invalid("saved incumbent reset chronology is inconsistent"));
        }
        while self.rows < rows {
            if self.rows == expected.incumbent_start {
                self.tracker.reset(self.rows);
            }
            self.tracker.record(history, TrialIndex(self.rows))?;
            self.rows += 1;
            cooperation.completed(1).await?;
        }
        if rows == expected.incumbent_start {
            self.tracker.reset(rows);
        }
        self.phase = phase;
        if self.tracker.state() != expected {
            return Err(invalid(
                "saved observer summary differs from its measured prefix",
            ));
        }
        Ok(())
    }
    pub(in super::super) fn into_tracker(self) -> Tracker {
        self.tracker
    }
}
