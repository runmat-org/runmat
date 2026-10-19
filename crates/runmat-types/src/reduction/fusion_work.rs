//! Cold, unit-weighted algorithmic stage priors, not ISA counts or measured time.
//! These immutable descriptions confer neither numerical nor execution admission.
use super::*;
use crate::NumericClass;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct FusionReductionWorkPrior {
    pub sample_visits: u64,
    pub combines: u64,
    pub nan_classifications: u64,
    pub count_updates: u64,
    pub tie_checks: u64,
    pub finalizations: u64,
    pub divisions: u64,
    pub conversions: u64,
}
impl FusionReductionWorkPrior {
    pub fn checked_units(self) -> Option<u64> {
        [
            self.sample_visits,
            self.combines,
            self.nan_classifications,
            self.count_updates,
            self.tie_checks,
            self.finalizations,
            self.divisions,
            self.conversions,
        ]
        .into_iter()
        .try_fold(0_u64, u64::checked_add)
    }
}

impl FusionReductionContract {
    /// Checked owner geometry, including empty extents and identity forwarding.
    /// Returns (reduced elements per slice, output slices).
    pub fn checked_work_geometry(&self) -> Option<(u64, u64)> {
        if ![self.input_class, self.accumulator_class, self.output_class]
            .into_iter()
            .all(|class| matches!(class, NumericClass::Single | NumericClass::Double))
        {
            return None;
        }
        let valid = match self.operation {
            FusionReductionOperation::Sum => {
                self.accumulator_class == NumericClass::Double
                    && self.count == FusionReductionCount::None
                    && self.empty == FusionReductionEmpty::PositiveZero
                    && self.ties == FusionReductionTies::NotApplicable
            }
            FusionReductionOperation::Mean => {
                self.accumulator_class == NumericClass::Double
                    && self.count
                        == match self.nan_mode {
                            ReductionNaN::Include => FusionReductionCount::SliceLength,
                            ReductionNaN::Omit => FusionReductionCount::NonNan,
                        }
                    && self.empty == FusionReductionEmpty::NaN
                    && self.ties == FusionReductionTies::NotApplicable
            }
            FusionReductionOperation::Minimum | FusionReductionOperation::Maximum => {
                self.accumulator_class == self.input_class
                    && self.output_class == self.input_class
                    && self.count == FusionReductionCount::None
                    && self.empty == FusionReductionEmpty::NaN
                    && self.ties
                        == if self.operation == FusionReductionOperation::Minimum {
                            FusionReductionTies::FirstWithNegativeZero
                        } else {
                            FusionReductionTies::FirstWithPositiveZero
                        }
            }
        };
        if !valid {
            return None;
        }
        let mut axes = vec![false; self.input_shape.len()];
        for &axis in &self.axes {
            if axis >= axes.len() || axes[axis] {
                return None;
            }
            axes[axis] = true;
        }
        if self.identity {
            axes.fill(false);
        }
        let expected: Vec<_> = self
            .input_shape
            .iter()
            .zip(&axes)
            .map(|(&n, &reduced)| if reduced { 1 } else { n })
            .collect();
        if self.output_shape != expected
            && !(elements(&expected)? == 1 && self.output_shape == [1, 1])
        {
            return None;
        }
        let reduced: Vec<_> = self
            .input_shape
            .iter()
            .zip(&axes)
            .filter_map(|(&n, &selected)| selected.then_some(n))
            .collect();
        let slices: Vec<_> = self
            .input_shape
            .iter()
            .zip(&axes)
            .filter_map(|(&n, &selected)| (!selected).then_some(n))
            .collect();
        let geometry = (elements(&reduced)?, elements(&slices)?);
        if geometry.0.checked_mul(geometry.1)? != elements(&self.input_shape)? {
            return None;
        }
        Some(geometry)
    }

    /// Original-class seeded host-fold prior. Payload-dependent early exits do
    /// not shorten this cold prior. Empty slices still have publication work.
    pub fn algorithmic_work_prior(&self) -> Option<FusionReductionWorkPrior> {
        let (length, slices) = self.checked_work_geometry()?;
        let visits = if self.identity {
            0
        } else {
            length.checked_mul(slices)?
        };
        self.algorithmic_work_for_schedule(visits, visits, slices, false)
    }

    /// Reuse the same named stages for a physical owner's explicit schedule.
    /// `input_conversion_accounted` excludes only the conversion already priced
    /// by that owner's producer boundary (for example an integer word decoder).
    /// This is a descriptive prior, not a certificate that a schedule is emitted.
    pub fn algorithmic_work_for_schedule(
        &self,
        samples: u64,
        combines: u64,
        finishes: u64,
        input_conversion_accounted: bool,
    ) -> Option<FusionReductionWorkPrior> {
        let (length, _) = self.checked_work_geometry()?;
        let identity = self.identity;
        if identity && combines != 0 {
            return None;
        }
        let input_conversions = if !identity
            && !input_conversion_accounted
            && self.input_class != self.accumulator_class
        {
            samples
        } else {
            0
        };
        let publication_class = if identity {
            self.input_class
        } else {
            self.accumulator_class
        };
        let output_conversions = if publication_class != self.output_class {
            finishes
        } else {
            0
        };
        let divisions =
            if !identity && length != 0 && self.operation == FusionReductionOperation::Mean {
                finishes
            } else {
                0
            };
        let prior = FusionReductionWorkPrior {
            sample_visits: samples,
            combines,
            // Both include and omit policies classify NaNs; they differ in
            // propagation and denominator ownership, not classification count.
            nan_classifications: if identity { 0 } else { samples },
            count_updates: if !identity && self.count == FusionReductionCount::NonNan {
                combines
            } else {
                0
            },
            tie_checks: if !identity
                && matches!(
                    self.operation,
                    FusionReductionOperation::Minimum | FusionReductionOperation::Maximum
                ) {
                combines
            } else {
                0
            },
            finalizations: finishes,
            divisions,
            // Mean also converts its integer denominator before division.
            conversions: input_conversions
                .checked_add(output_conversions)?
                .checked_add(divisions)?,
        };
        prior.checked_units()?;
        Some(prior)
    }
}

fn elements(shape: &[usize]) -> Option<u64> {
    if shape.contains(&0) {
        return Some(0);
    }
    shape.iter().try_fold(1_u64, |n, &extent| {
        n.checked_mul(u64::try_from(extent).ok()?)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sum() -> FusionReductionContract {
        FusionReductionContract {
            operation: FusionReductionOperation::Sum,
            input_class: NumericClass::Single,
            accumulator_class: NumericClass::Double,
            output_class: NumericClass::Single,
            input_shape: vec![4, 2],
            output_shape: vec![1, 2],
            axes: vec![0],
            identity: false,
            nan_mode: ReductionNaN::Include,
            count: FusionReductionCount::None,
            empty: FusionReductionEmpty::PositiveZero,
            ties: FusionReductionTies::NotApplicable,
        }
    }

    #[test]
    fn host_fold_counts_original_classes_and_empty_identity_publication() {
        let mut contract = sum();
        let work = contract.algorithmic_work_prior().unwrap();
        assert_eq!(
            (work.sample_visits, work.combines, work.finalizations),
            (8, 8, 2)
        );
        assert_eq!(work.conversions, 8 + 2);
        contract.input_shape[0] = 0;
        let empty = contract.algorithmic_work_prior().unwrap();
        assert_eq!(
            (empty.sample_visits, empty.combines, empty.finalizations),
            (0, 0, 2)
        );
        assert!(empty.checked_units().unwrap() > 0);
        contract.input_shape[0] = 1;
        contract.identity = true;
        let identity = contract.algorithmic_work_prior().unwrap();
        assert_eq!(
            (
                identity.sample_visits,
                identity.combines,
                identity.finalizations
            ),
            (0, 0, 2)
        );
        assert_eq!((identity.nan_classifications, identity.conversions), (0, 0));
    }

    #[test]
    fn closed_mean_extrema_policies_and_overflow_are_not_silent_zero() {
        let mut contract = sum();
        contract.operation = FusionReductionOperation::Mean;
        contract.count = FusionReductionCount::NonNan;
        contract.nan_mode = ReductionNaN::Omit;
        contract.empty = FusionReductionEmpty::NaN;
        let work = contract.algorithmic_work_prior().unwrap();
        assert_eq!(
            (work.nan_classifications, work.count_updates, work.divisions),
            (8, 8, 2)
        );
        assert_eq!(work.conversions, 8 + 2 + 2);
        contract.input_shape[0] = 0;
        assert_eq!(contract.algorithmic_work_prior().unwrap().divisions, 0);
        contract = sum();
        contract.operation = FusionReductionOperation::Minimum;
        contract.accumulator_class = NumericClass::Single;
        contract.empty = FusionReductionEmpty::NaN;
        contract.ties = FusionReductionTies::FirstWithNegativeZero;
        assert_eq!(contract.algorithmic_work_prior().unwrap().tie_checks, 8);
        contract.axes.push(0);
        assert!(contract.algorithmic_work_prior().is_none());
        contract = sum();
        contract.input_shape = vec![usize::MAX; 3];
        contract.axes = vec![0, 1, 2];
        contract.output_shape = vec![1; 3];
        assert!(contract.algorithmic_work_prior().is_none());
        contract = sum();
        assert!(contract
            .algorithmic_work_for_schedule(u64::MAX, 0, 0, false)
            .is_none());
        contract.input_class = NumericClass::Int32;
        assert!(contract.algorithmic_work_prior().is_none());
    }
}
