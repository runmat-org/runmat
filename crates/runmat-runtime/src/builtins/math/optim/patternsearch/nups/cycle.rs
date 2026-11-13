//! Nonuniform phase allocation uses actual poll progress and point counts.
use super::super::invalid;
use crate::BuiltinResult;
use runmat_builtins::OptimPatternAlgorithm;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Phase {
    Gps,
    Orthogonal,
}
impl Phase {
    pub fn index(self) -> usize {
        match self {
            Self::Gps => 0,
            Self::Orthogonal => 1,
        }
    }
}
pub(super) enum Observation {
    NoImprovement {
        points: usize,
    },
    Improved {
        change: super::super::poll::MeritChange,
        points: usize,
    },
}
#[derive(Clone, Copy)]
struct Progress {
    scale: f64,
    gain: f64,
    points: usize,
}
impl Progress {
    fn add(
        &mut self,
        change: Option<super::super::poll::MeritChange>,
        points: usize,
    ) -> BuiltinResult<()> {
        self.points = self
            .points
            .checked_add(points)
            .ok_or_else(|| invalid("NUPS point count overflow"))?;
        if let Some(change) = change {
            let scale = self
                .scale
                .max(change.before.abs())
                .max(change.after.abs())
                .max(1.0);
            self.gain = self.gain * (self.scale / scale)
                + (change.before / scale - change.after / scale).max(0.0);
            self.scale = scale;
        }
        if !self.gain.is_finite() {
            return Err(invalid(
                "NUPS cycle progress is numerically unrepresentable",
            ));
        }
        Ok(())
    }
}
pub(super) struct Cycle {
    algorithm: OptimPatternAlgorithm,
    gps_only: bool,
    position: u8,
    gps_count: u8,
    positive: [bool; 2],
    progress: [Option<Progress>; 2],
}
impl Cycle {
    pub fn new(algorithm: OptimPatternAlgorithm, equalities: bool) -> BuiltinResult<Self> {
        if algorithm == OptimPatternAlgorithm::Classic {
            return Err(invalid("classic polling has no NUPS cycle"));
        }
        if equalities && algorithm == OptimPatternAlgorithm::NupsMads {
            return Err(invalid("nups-mads does not support linear equalities"));
        }
        Ok(Self {
            algorithm,
            gps_only: equalities,
            position: 0,
            gps_count: 8,
            positive: [true, true],
            progress: [None, None],
        })
    }
    pub fn phase(&self) -> Phase {
        if self.gps_only
            || self.algorithm == OptimPatternAlgorithm::NupsGps
            || (self.algorithm == OptimPatternAlgorithm::Nups && self.position < self.gps_count)
        {
            Phase::Gps
        } else {
            Phase::Orthogonal
        }
    }
    pub fn positive(&self) -> bool {
        self.positive[self.phase().index()]
    }
    pub fn advance(&mut self, poll: Option<Observation>) -> BuiltinResult<()> {
        let phase = self.phase().index();
        if let Some(poll) = poll {
            let (change, points) = match poll {
                Observation::NoImprovement { points } => (None, points),
                Observation::Improved { change, points } => (Some(change), points),
            };
            if change.is_some_and(|change| {
                !change.before.is_finite()
                    || !change.after.is_finite()
                    || change.after >= change.before
            }) {
                return Err(invalid(
                    "NUPS progress does not match an admitted improving poll",
                ));
            }
            if points == 0 && change.is_some() {
                return Err(invalid(
                    "NUPS improvement has no actual objective evaluation",
                ));
            }
            if points > 0 {
                let index = self.phase().index();
                let progress = self.progress[index].get_or_insert(Progress {
                    scale: 1.0,
                    gain: 0.0,
                    points: 0,
                });
                progress.add(change, points)?;
            }
            self.positive[phase] = !self.positive[phase];
        }
        self.position += 1;
        if self.position == 16 {
            if self.algorithm == OptimPatternAlgorithm::Nups && !self.gps_only {
                let scale = self
                    .progress
                    .iter()
                    .flatten()
                    .fold(1.0f64, |scale, value| scale.max(value.scale));
                let rate = |index: usize| {
                    self.progress[index].map_or(0.0, |value| {
                        value.gain * (value.scale / scale) / (value.points as f64)
                    })
                };
                let gps = rate(0);
                let orthogonal = rate(1);
                self.gps_count = if gps + orthogonal > 0.0 {
                    (16.0 * gps / (gps + orthogonal)).round().clamp(2.0, 14.0) as u8
                } else {
                    8
                };
            }
            self.position = 0;
            self.progress = [None, None];
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rates_use_real_counts_scale_stably_and_never_remove_a_phase() {
        let mut cycle = Cycle::new(OptimPatternAlgorithm::Nups, false).unwrap();
        for index in 0..16 {
            assert_eq!(
                cycle.phase(),
                if index < 8 {
                    Phase::Gps
                } else {
                    Phase::Orthogonal
                }
            );
            assert_eq!(cycle.positive(), index % 2 == 0);
            cycle
                .advance(Some(Observation::Improved {
                    change: super::super::super::poll::MeritChange {
                        before: f64::MAX,
                        after: -f64::MAX,
                    },
                    points: if index < 8 { 1 } else { 10 },
                }))
                .unwrap();
        }
        assert!(cycle.gps_count > 8 && cycle.gps_count <= 14);
        for _ in 0..16 {
            cycle.advance(None).unwrap();
        }
        assert_eq!(cycle.gps_count, 8);
        let mut fixed = Cycle::new(OptimPatternAlgorithm::Nups, true).unwrap();
        for _ in 0..32 {
            assert_eq!(fixed.phase(), Phase::Gps);
            fixed.advance(None).unwrap();
        }
        assert!(Cycle::new(OptimPatternAlgorithm::NupsMads, true).is_err());
        let cycle = Cycle::new(OptimPatternAlgorithm::Nups, false).unwrap();
        assert_eq!(cycle.position, 0);
        assert!(cycle.positive());
    }
}
