//! Borrowed dateshift geometry and typed controls; no target/rule vectors.
use super::input::IntegerView;
use crate::builtins::datetime::{
    conversion::BorrowedDatetimeSerial,
    dateshift_model::{weekday_from_matlab_index, DateShiftRule, DayTarget},
    datetime_error,
};
use crate::BuiltinResult;
use runmat_builtins::DateShiftUnit;

pub(super) enum Target<'a> {
    Default,
    Text(DayTarget),
    Numeric(IntegerView<'a>),
}

impl<'a> Target<'a> {
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Numeric(view) => view.len(),
            _ => 1,
        }
    }
    fn shape(&self) -> &'a [usize] {
        const SCALAR: &[usize] = &[1, 1];
        match self {
            Self::Numeric(view) => view.shape(),
            _ => SCALAR,
        }
    }
    pub(super) fn at(&self, index: usize) -> BuiltinResult<DayTarget> {
        match self {
            Self::Default => Ok(DayTarget::Exact(chrono::Weekday::Sun)),
            Self::Text(target) => Ok(*target),
            Self::Numeric(view) => {
                weekday_from_matlab_index(view.at(index, "weekday")?).map(DayTarget::Exact)
            }
        }
    }
}

pub(super) enum Rule<'a> {
    Scalar(DateShiftRule),
    Numeric(IntegerView<'a>),
}

impl<'a> Rule<'a> {
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Numeric(view) => view.len(),
            _ => 1,
        }
    }
    fn shape(&self) -> &'a [usize] {
        const SCALAR: &[usize] = &[1, 1];
        match self {
            Self::Numeric(view) => view.shape(),
            _ => SCALAR,
        }
    }
    pub(super) fn at(&self, index: usize) -> BuiltinResult<DateShiftRule> {
        match self {
            Self::Scalar(rule) => Ok(*rule),
            Self::Numeric(view) => Ok(DateShiftRule::Occurrence(view.at(index, "rule")?)),
        }
    }
}

pub(super) struct Plan<'a> {
    pub(super) serials: BorrowedDatetimeSerial<'a>,
    pub(super) target: Target<'a>,
    pub(super) rule: Rule<'a>,
    pub(super) unit: Option<DateShiftUnit>,
    pub(super) len: usize,
    pub(super) shape: &'a [usize],
}

impl<'a> Plan<'a> {
    pub(super) fn new(
        serials: BorrowedDatetimeSerial<'a>,
        target: Target<'a>,
        rule: Rule<'a>,
        unit: Option<DateShiftUnit>,
    ) -> BuiltinResult<Self> {
        let lengths = [serials.len(), target.len(), rule.len()];
        let output_len = lengths.iter().copied().find(|len| *len != 1).unwrap_or(1);
        if lengths.iter().any(|len| *len != 1 && *len != output_len) {
            return Err(datetime_error(
                "dateshift: datetime, weekday, and rule arrays must have matching sizes or be scalar",
            ));
        }
        let serial_shape = serials.shape();
        let shapes = [serial_shape, target.shape(), rule.shape()];
        let mut selected = None;
        for (len, shape) in lengths.into_iter().zip(shapes) {
            if len != 1 {
                if selected.is_some_and(|prior: &[usize]| prior != shape) {
                    return Err(datetime_error(
                        "dateshift: non-scalar datetime, weekday, and rule arrays must have matching sizes",
                    ));
                }
                selected = Some(shape);
            }
        }
        const SCALAR: &[usize] = &[1, 1];
        Ok(Self {
            serials,
            target,
            rule,
            unit,
            len: output_len,
            shape: selected.unwrap_or(SCALAR),
        })
    }

    pub(super) fn serial_at(&self, index: usize) -> f64 {
        self.serials
            .at(if self.serials.len() == 1 { 0 } else { index })
    }
    pub(super) fn target_at(&self, index: usize) -> BuiltinResult<DayTarget> {
        self.target
            .at(if self.target.len() == 1 { 0 } else { index })
    }
    pub(super) fn rule_at(&self, index: usize) -> BuiltinResult<DateShiftRule> {
        self.rule.at(if self.rule.len() == 1 { 0 } else { index })
    }
}
