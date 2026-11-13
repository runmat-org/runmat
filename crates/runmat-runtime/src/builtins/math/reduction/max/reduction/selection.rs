use runmat_builtins::{MaxComparisonMethod as ComparisonMethod, ReductionNaN};
use std::cmp::Ordering;

#[derive(Debug, Clone)]
pub(super) struct BestReal {
    pub(super) value: f64,
    pub(super) reduce_index: usize,
    pub(super) full_index: usize,
    pub(super) has_value: bool,
    pub(super) nan_fixed: bool,
}

impl BestReal {
    pub(super) fn new() -> Self {
        Self {
            value: 0.0,
            reduce_index: 0,
            full_index: 0,
            has_value: false,
            nan_fixed: false,
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct BestComplex {
    pub(super) value: (f64, f64),
    pub(super) reduce_index: usize,
    pub(super) full_index: usize,
    pub(super) has_value: bool,
    pub(super) nan_fixed: bool,
}

impl BestComplex {
    pub(super) fn new() -> Self {
        Self {
            value: (0.0, 0.0),
            reduce_index: 0,
            full_index: 0,
            has_value: false,
            nan_fixed: false,
        }
    }
}

pub(super) fn update_best_real(
    best: &mut BestReal,
    value: f64,
    reduce_index: usize,
    full_index: usize,
    nan_mode: ReductionNaN,
    comparison: ComparisonMethod,
) {
    if value.is_nan() {
        match nan_mode {
            ReductionNaN::Include => {
                if !best.nan_fixed {
                    best.value = f64::NAN;
                    best.reduce_index = reduce_index;
                    best.full_index = full_index;
                    best.has_value = true;
                    best.nan_fixed = true;
                }
            }
            ReductionNaN::Omit => {}
        }
        return;
    }
    if best.nan_fixed {
        return;
    }

    if !best.has_value {
        best.value = value;
        best.reduce_index = reduce_index;
        best.full_index = full_index;
        best.has_value = true;
        return;
    }

    if should_replace_real(best.value, value, comparison) {
        best.value = value;
        best.reduce_index = reduce_index;
        best.full_index = full_index;
    }
}

pub(super) fn update_best_complex(
    best: &mut BestComplex,
    value: (f64, f64),
    reduce_index: usize,
    full_index: usize,
    nan_mode: ReductionNaN,
    comparison: ComparisonMethod,
) {
    if value.0.is_nan() || value.1.is_nan() {
        match nan_mode {
            ReductionNaN::Include => {
                if !best.nan_fixed {
                    best.value = (f64::NAN, f64::NAN);
                    best.reduce_index = reduce_index;
                    best.full_index = full_index;
                    best.has_value = true;
                    best.nan_fixed = true;
                }
            }
            ReductionNaN::Omit => {}
        }
        return;
    }
    if best.nan_fixed {
        return;
    }

    if !best.has_value {
        best.value = value;
        best.reduce_index = reduce_index;
        best.full_index = full_index;
        best.has_value = true;
        return;
    }

    if should_replace_complex(best.value, value, comparison) {
        best.value = value;
        best.reduce_index = reduce_index;
        best.full_index = full_index;
    }
}

pub(in crate::builtins::math::reduction::max) fn should_replace_real(
    current: f64,
    candidate: f64,
    comparison: ComparisonMethod,
) -> bool {
    match comparison {
        ComparisonMethod::Auto | ComparisonMethod::Real => {
            if candidate > current {
                return true;
            }
            if candidate < current {
                return false;
            }
            if candidate == 0.0 && current == 0.0 {
                return candidate.is_sign_positive() && !current.is_sign_positive();
            }
            false
        }
        ComparisonMethod::Abs => {
            let curr_abs = current.abs();
            let cand_abs = candidate.abs();
            if cand_abs > curr_abs {
                return true;
            }
            if cand_abs < curr_abs {
                return false;
            }
            if candidate > current {
                return true;
            }
            if candidate < current {
                return false;
            }
            if candidate == 0.0 && current == 0.0 {
                return candidate.is_sign_positive() && !current.is_sign_positive();
            }
            false
        }
    }
}

pub(in crate::builtins::math::reduction::max) fn should_replace_complex(
    current: (f64, f64),
    candidate: (f64, f64),
    comparison: ComparisonMethod,
) -> bool {
    match comparison {
        ComparisonMethod::Auto | ComparisonMethod::Abs => {
            compare_complex_auto(current, candidate) == Ordering::Less
        }
        ComparisonMethod::Real => compare_complex_real(current, candidate) == Ordering::Less,
    }
}

fn compare_complex_auto(a: (f64, f64), b: (f64, f64)) -> Ordering {
    let a_mag = magnitude_squared(a);
    let b_mag = magnitude_squared(b);
    if a_mag < b_mag {
        return Ordering::Less;
    }
    if a_mag > b_mag {
        return Ordering::Greater;
    }
    // Equal magnitude: tie-break using phase angle.
    let a_angle = a.1.atan2(a.0);
    let b_angle = b.1.atan2(b.0);
    if a_angle < b_angle {
        Ordering::Less
    } else if a_angle > b_angle {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

fn compare_complex_real(a: (f64, f64), b: (f64, f64)) -> Ordering {
    if a.0 < b.0 {
        return Ordering::Less;
    }
    if a.0 > b.0 {
        return Ordering::Greater;
    }
    // Equal real parts: use magnitude and phase tie-breakers.
    compare_complex_auto(a, b)
}

fn magnitude_squared(z: (f64, f64)) -> f64 {
    z.0.mul_add(z.0, z.1 * z.1)
}
