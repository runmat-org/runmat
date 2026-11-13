//! Current and best are original rooted generations, not replayed payloads.
use crate::execution::RootedValue;
pub(super) struct State {
    pub current: RootedValue,
    pub current_f: f64,
    pub best: RootedValue,
    pub best_f: f64,
    pub temperature: Vec<f64>,
    pub k: Vec<f64>,
    pub iteration: usize,
    pub accepted: usize,
    pub stall: Stall,
}
pub(super) struct Stall {
    changes: std::collections::VecDeque<f64>,
    average: f64,
    width: usize,
}
impl Stall {
    pub fn new(width: usize) -> Self {
        Self {
            changes: std::collections::VecDeque::new(),
            average: 0.0,
            width,
        }
    }
    pub fn reset(&mut self, width: usize) {
        self.changes.clear();
        self.average = 0.0;
        self.width = width;
    }
    pub async fn push(
        &mut self,
        previous: f64,
        current: f64,
        c: &mut super::ScalarCooperation,
    ) -> crate::BuiltinResult<bool> {
        if self.width == 0 {
            return Ok(true);
        }
        let change = (previous - current).abs();
        if self.changes.len() == self.width {
            self.changes.pop_front().expect("full window");
        }
        self.changes.push_back(change);
        let mut scale = 0.0_f64;
        for &value in &self.changes {
            scale = scale.max(value);
            c.completed(1).await?;
        }
        self.average = if scale == 0.0 || scale.is_infinite() {
            scale
        } else {
            let mut mean = 0.0;
            for &value in &self.changes {
                mean += (value / scale) / self.changes.len() as f64;
                c.completed(1).await?;
            }
            scale * mean
        };
        Ok(self.changes.len() == self.width)
    }
    pub fn average(&self) -> f64 {
        self.average
    }
    pub fn full(&self) -> bool {
        self.width == 0 || self.changes.len() == self.width
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Termination {
    Stall,
    Objective,
    Budget,
    Observer,
    Infeasible,
    Time,
}
impl Termination {
    pub fn flag(self) -> f64 {
        match self {
            Self::Stall => 1.0,
            Self::Objective => 5.0,
            Self::Budget => 0.0,
            Self::Observer => -1.0,
            Self::Infeasible => -2.0,
            Self::Time => -5.0,
        }
    }
    pub fn message(self) -> &'static str {
        match self {
            Self::Stall => "simulannealbnd: average objective change reached FunctionTolerance.",
            Self::Objective => "simulannealbnd: best objective is below ObjectiveLimit.",
            Self::Budget => "simulannealbnd: iteration or objective evaluation limit reached.",
            Self::Observer => "simulannealbnd: stopped by an output or plot function.",
            Self::Infeasible => "simulannealbnd: no point satisfies the bounds.",
            Self::Time => "simulannealbnd: MaxTime reached.",
        }
    }
}
