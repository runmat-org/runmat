//! One resumable numerical authority for centered and uncentered directions.

const MAX_UNINTERRUPTED_ELEMENTS: u64 = 4096;

#[derive(Debug)]
pub(in crate::builtins::stats::ml) enum DirectionError {
    Resource(std::collections::TryReserveError),
}

pub(in crate::builtins::stats::ml) enum DirectionProgress {
    Pending(u64),
    Complete(Option<Vec<f64>>, u64),
}

#[derive(Clone, Copy)]
enum Phase {
    Validate,
    Differences,
    FallbackScale,
    FallbackDifferences,
    Copy,
    FirstScale,
    FirstDivide,
    Mean,
    SubtractMean,
    FinalScale,
    Squared,
    Normalize,
    Complete,
}

/// A scalar-step traversal so native and cooperatively scheduled callers
/// perform identical arithmetic, in the same order, without a duplicate
/// implementation of the centering/extreme-value policy.
pub(in crate::builtins::stats::ml) struct DirectionWork<'a> {
    row: &'a [f64],
    center: bool,
    values: Vec<f64>,
    phase: Phase,
    index: usize,
    scale: f64,
    mean: f64,
    squared: f64,
    difference_overflowed: bool,
}

impl<'a> DirectionWork<'a> {
    pub(in crate::builtins::stats::ml) fn new(row: &'a [f64], center: bool) -> Self {
        Self {
            row,
            center,
            values: Vec::new(),
            phase: Phase::Validate,
            index: 0,
            scale: 0.0,
            mean: 0.0,
            squared: 0.0,
            difference_overflowed: false,
        }
    }

    pub(in crate::builtins::stats::ml) fn advance(
        &mut self,
    ) -> Result<DirectionProgress, DirectionError> {
        let mut completed = 0;
        while completed < MAX_UNINTERRUPTED_ELEMENTS {
            match self.phase {
                Phase::Validate => {
                    if self.row.is_empty() {
                        self.phase = Phase::Complete;
                        continue;
                    }
                    if self.index == self.row.len() {
                        self.values
                            .try_reserve_exact(self.row.len())
                            .map_err(DirectionError::Resource)?;
                        self.begin(if self.center {
                            Phase::Differences
                        } else {
                            Phase::Copy
                        });
                    } else if !self.row[self.index].is_finite() {
                        self.phase = Phase::Complete;
                    } else {
                        self.index += 1;
                        completed += 1;
                    }
                }
                Phase::Differences => {
                    if self.index == self.row.len() {
                        if self.difference_overflowed {
                            self.values.clear();
                            self.begin(Phase::FallbackScale);
                        } else {
                            self.begin(Phase::FirstScale);
                        }
                    } else {
                        let difference = self.row[self.index] - self.row[0];
                        self.difference_overflowed |= !difference.is_finite();
                        self.values.push(difference);
                        self.index += 1;
                        completed += 1;
                    }
                }
                Phase::FallbackScale => {
                    if self.index == self.row.len() {
                        self.begin(Phase::FallbackDifferences);
                    } else {
                        self.scale = self.scale.max(self.row[self.index].abs());
                        self.index += 1;
                        completed += 1;
                    }
                }
                Phase::FallbackDifferences => {
                    if self.index == self.row.len() {
                        self.begin(Phase::FirstScale);
                        self.scale = 0.0;
                    } else {
                        self.values
                            .push(self.row[self.index] / self.scale - self.row[0] / self.scale);
                        self.index += 1;
                        completed += 1;
                    }
                }
                Phase::Copy => {
                    if self.index == self.row.len() {
                        self.begin(Phase::FirstScale);
                    } else {
                        self.values.push(self.row[self.index]);
                        self.index += 1;
                        completed += 1;
                    }
                }
                Phase::FirstScale => {
                    if self.index == self.values.len() {
                        if self.scale == 0.0 {
                            self.phase = Phase::Complete;
                        } else {
                            self.begin(Phase::FirstDivide);
                        }
                    } else {
                        self.scale = self.scale.max(self.values[self.index].abs());
                        self.index += 1;
                        completed += 1;
                    }
                }
                Phase::FirstDivide => {
                    if self.index == self.values.len() {
                        self.begin(if self.center {
                            Phase::Mean
                        } else {
                            Phase::FinalScale
                        });
                        self.scale = 0.0;
                    } else {
                        self.values[self.index] /= self.scale;
                        self.index += 1;
                        completed += 1;
                    }
                }
                Phase::Mean => {
                    if self.index == self.values.len() {
                        self.begin(Phase::SubtractMean);
                    } else {
                        self.mean += self.values[self.index] / self.values.len() as f64;
                        self.index += 1;
                        completed += 1;
                    }
                }
                Phase::SubtractMean => {
                    if self.index == self.values.len() {
                        self.begin(Phase::FinalScale);
                    } else {
                        self.values[self.index] -= self.mean;
                        self.index += 1;
                        completed += 1;
                    }
                }
                Phase::FinalScale => {
                    if self.index == self.values.len() {
                        if self.scale == 0.0 {
                            self.phase = Phase::Complete;
                        } else {
                            self.begin(Phase::Squared);
                        }
                    } else {
                        self.scale = self.scale.max(self.values[self.index].abs());
                        self.index += 1;
                        completed += 1;
                    }
                }
                Phase::Squared => {
                    if self.index == self.values.len() {
                        self.squared = self.squared.sqrt();
                        self.begin(Phase::Normalize);
                    } else {
                        self.squared += (self.values[self.index] / self.scale).powi(2);
                        self.index += 1;
                        completed += 1;
                    }
                }
                Phase::Normalize => {
                    if self.index == self.values.len() {
                        return Ok(DirectionProgress::Complete(
                            Some(std::mem::take(&mut self.values)),
                            completed,
                        ));
                    }
                    self.values[self.index] = (self.values[self.index] / self.scale) / self.squared;
                    self.index += 1;
                    completed += 1;
                }
                Phase::Complete => return Ok(DirectionProgress::Complete(None, completed)),
            }
        }
        Ok(DirectionProgress::Pending(completed))
    }

    fn begin(&mut self, phase: Phase) {
        self.phase = phase;
        self.index = 0;
    }
}

/// Test the synchronous driver of the same traversal used by cooperative
/// consumers, keeping resource failures distinct from degenerate directions.
#[cfg(test)]
pub(super) fn unit_direction(
    row: &[f64],
    center: bool,
) -> Result<Option<Vec<f64>>, DirectionError> {
    let mut work = DirectionWork::new(row, center);
    loop {
        match work.advance() {
            Ok(DirectionProgress::Pending(_)) => {}
            Ok(DirectionProgress::Complete(direction, _)) => return Ok(direction),
            Err(error) => return Err(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wide_row_resumes_across_each_normalization_phase() {
        let row = (0..8193)
            .map(|index| index as f64 + 1.0)
            .collect::<Vec<_>>();
        let mut work = DirectionWork::new(&row, true);
        let mut pending = 0;
        let actual = loop {
            match work.advance().unwrap() {
                DirectionProgress::Pending(units) => {
                    assert!(units > 0 && units <= MAX_UNINTERRUPTED_ELEMENTS);
                    pending += 1;
                }
                DirectionProgress::Complete(result, _) => break result.unwrap(),
            }
        };
        assert!(pending >= 5);
        assert_eq!(actual.len(), row.len());
        assert!(actual[0] < 0.0 && actual[row.len() - 1] > 0.0);
        let norm = actual.iter().map(|value| value * value).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1.0e-12);
    }
}
