use super::*;

#[derive(Clone, Copy)]
pub(super) struct RealEntry {
    pub(super) value: NumericScalar,
    pub(super) index: usize,
    pub(super) source_index: usize,
}

#[derive(Clone, Copy)]
pub(super) struct ComplexEntry {
    pub(super) value: (f64, f64),
    pub(super) index: usize,
    pub(super) source_index: usize,
}

pub(super) fn compare_real_entries(
    kind: TopKKind,
    method: ComparisonMethod,
    a: &RealEntry,
    b: &RealEntry,
) -> Ordering {
    let ordering = compare_numeric_scalars(method, a.value, b.value);
    let ordering = match kind {
        TopKKind::Max => ordering.reverse(),
        TopKKind::Min => ordering,
    };
    ordering.then_with(|| a.index.cmp(&b.index))
}

pub(super) fn compare_numeric_scalars(
    method: ComparisonMethod,
    a: NumericScalar,
    b: NumericScalar,
) -> Ordering {
    macro_rules! compare_signed {
        ($left:expr, $right:expr) => {{
            match method {
                ComparisonMethod::Auto | ComparisonMethod::Real => $left.cmp(&$right),
                ComparisonMethod::Abs => u128::from($left.unsigned_abs())
                    .cmp(&u128::from($right.unsigned_abs()))
                    .then_with(|| $left.cmp(&$right).reverse()),
            }
        }};
    }
    macro_rules! compare_unsigned {
        ($left:expr, $right:expr) => {{
            match method {
                ComparisonMethod::Auto | ComparisonMethod::Real => $left.cmp(&$right),
                ComparisonMethod::Abs => $left.cmp(&$right),
            }
        }};
    }
    match (a, b) {
        (NumericScalar::F64(left), NumericScalar::F64(right)) => {
            compare_f64_values(method, left, right)
        }
        (NumericScalar::F32(left), NumericScalar::F32(right)) => {
            compare_f32_values(method, left, right)
        }
        (NumericScalar::I8(left), NumericScalar::I8(right)) => compare_signed!(left, right),
        (NumericScalar::I16(left), NumericScalar::I16(right)) => compare_signed!(left, right),
        (NumericScalar::I32(left), NumericScalar::I32(right)) => compare_signed!(left, right),
        (NumericScalar::I64(left), NumericScalar::I64(right)) => compare_signed!(left, right),
        (NumericScalar::U8(left), NumericScalar::U8(right)) => compare_unsigned!(left, right),
        (NumericScalar::U16(left), NumericScalar::U16(right)) => compare_unsigned!(left, right),
        (NumericScalar::U32(left), NumericScalar::U32(right)) => compare_unsigned!(left, right),
        (NumericScalar::U64(left), NumericScalar::U64(right)) => compare_unsigned!(left, right),
        (left, right) => left
            .numeric_dtype()
            .class_name()
            .cmp(right.numeric_dtype().class_name()),
    }
}

pub(super) fn compare_f64_values(method: ComparisonMethod, a: f64, b: f64) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => match method {
            ComparisonMethod::Auto | ComparisonMethod::Real => {
                a.partial_cmp(&b).unwrap_or(Ordering::Equal)
            }
            ComparisonMethod::Abs => a
                .abs()
                .partial_cmp(&b.abs())
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.partial_cmp(&b).unwrap_or(Ordering::Equal).reverse()),
        },
    }
}

pub(super) fn compare_f32_values(method: ComparisonMethod, a: f32, b: f32) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => match method {
            ComparisonMethod::Auto | ComparisonMethod::Real => {
                a.partial_cmp(&b).unwrap_or(Ordering::Equal)
            }
            ComparisonMethod::Abs => a
                .abs()
                .partial_cmp(&b.abs())
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.partial_cmp(&b).unwrap_or(Ordering::Equal).reverse()),
        },
    }
}

pub(super) fn compare_complex_entries(
    kind: TopKKind,
    method: ComparisonMethod,
    a: &ComplexEntry,
    b: &ComplexEntry,
) -> Ordering {
    let ordering = compare_complex_values(method, a.value, b.value);
    let ordering = match kind {
        TopKKind::Max => ordering.reverse(),
        TopKKind::Min => ordering,
    };
    ordering.then_with(|| a.index.cmp(&b.index))
}

pub(super) fn compare_complex_values(
    method: ComparisonMethod,
    a: (f64, f64),
    b: (f64, f64),
) -> Ordering {
    let a_nan = a.0.is_nan() || a.1.is_nan();
    let b_nan = b.0.is_nan() || b.1.is_nan();
    match (a_nan, b_nan) {
        (true, true) => return Ordering::Equal,
        (true, false) => return Ordering::Greater,
        (false, true) => return Ordering::Less,
        (false, false) => {}
    }
    match method {
        ComparisonMethod::Real => {
            a.0.partial_cmp(&b.0)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
        }
        ComparisonMethod::Auto | ComparisonMethod::Abs => {
            let amag = a.0.hypot(a.1);
            let bmag = b.0.hypot(b.1);
            amag.partial_cmp(&bmag)
                .unwrap_or(Ordering::Equal)
                .then_with(|| {
                    a.1.atan2(a.0)
                        .partial_cmp(&b.1.atan2(b.0))
                        .unwrap_or(Ordering::Equal)
                })
        }
    }
}
