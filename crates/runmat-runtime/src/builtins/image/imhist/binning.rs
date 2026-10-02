//! Exact class-wide and indexed histogram binning.
use super::*;

pub(super) fn linspace(start: f64, stop: f64, count: usize) -> Vec<f64> {
    if count == 0 {
        return Vec::new();
    }
    if count == 1 {
        return vec![start];
    }
    let step = (stop - start) / (count - 1) as f64;
    (0..count).map(|idx| start + step * idx as f64).collect()
}

pub(super) fn histogram_counts(
    storage: &NumericStorage,
    class_min: f64,
    class_max: f64,
    bins: usize,
) -> BuiltinResult<Vec<f64>> {
    match storage {
        NumericStorage::F64(values) => histogram_float_counts(values, class_min, class_max, bins),
        NumericStorage::F32(values) => {
            let values: Vec<f64> = values.iter().map(|value| f64::from(*value)).collect();
            histogram_float_counts(&values, class_min, class_max, bins)
        }
        NumericStorage::I8(values) => histogram_signed_counts(values, i8::MIN, bins),
        NumericStorage::I16(values) => histogram_signed_counts(values, i16::MIN, bins),
        NumericStorage::I32(values) => histogram_signed_counts(values, i32::MIN, bins),
        NumericStorage::I64(values) => histogram_signed_counts(values, i64::MIN, bins),
        NumericStorage::U8(values) => histogram_unsigned_counts(values, u8::MAX, bins),
        NumericStorage::U16(values) => histogram_unsigned_counts(values, u16::MAX, bins),
        NumericStorage::U32(values) => histogram_unsigned_counts(values, u32::MAX, bins),
        NumericStorage::U64(values) => histogram_unsigned_counts(values, u64::MAX, bins),
    }
}

fn histogram_float_counts(
    values: &[f64],
    class_min: f64,
    class_max: f64,
    bins: usize,
) -> BuiltinResult<Vec<f64>> {
    let mut counts = vec![0.0; bins];
    if bins == 0 {
        return Ok(counts);
    }
    if bins == 1 || (class_max - class_min).abs() <= f64::EPSILON {
        for &value in values {
            if !value.is_finite() || value < class_min || value > class_max {
                return Err(invalid(
                    "grayscale image values are outside the image class range",
                ));
            }
            counts[0] += 1.0;
        }
        return Ok(counts);
    }
    let scale = (bins - 1) as f64 / (class_max - class_min);
    for &value in values {
        if !value.is_finite() || value < class_min || value > class_max {
            return Err(invalid(
                "grayscale image values are outside the image class range",
            ));
        }
        let relative = ((value - class_min) * scale).round();
        let index = if relative <= 0.0 {
            0
        } else if relative >= (bins - 1) as f64 {
            bins - 1
        } else {
            relative as usize
        };
        counts[index] += 1.0;
    }
    Ok(counts)
}

fn nearest_integer_bin(offset: u128, range: u128, bins: usize) -> usize {
    if bins <= 1 || range == 0 {
        return 0;
    }
    let numerator = offset * (bins - 1) as u128;
    ((numerator * 2 + range) / (range * 2)) as usize
}

trait HistogramSigned: Copy {
    fn to_i128(self) -> i128;
}

macro_rules! impl_histogram_signed {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl HistogramSigned for $ty {
                fn to_i128(self) -> i128 {
                    self as i128
                }
            }
        )+
    };
}

impl_histogram_signed!(i8, i16, i32, i64);

fn histogram_signed_counts<T: HistogramSigned>(
    values: &[T],
    minimum: T,
    bins: usize,
) -> BuiltinResult<Vec<f64>> {
    let minimum = minimum.to_i128();
    let range = (-minimum * 2 - 1) as u128;
    let mut counts = vec![0.0; bins];
    for &value in values {
        let offset = (value.to_i128() - minimum) as u128;
        counts[nearest_integer_bin(offset, range, bins)] += 1.0;
    }
    Ok(counts)
}

trait HistogramUnsigned: Copy {
    fn to_u128(self) -> u128;
}

macro_rules! impl_histogram_unsigned {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl HistogramUnsigned for $ty {
                fn to_u128(self) -> u128 {
                    self as u128
                }
            }
        )+
    };
}

impl_histogram_unsigned!(u8, u16, u32, u64);

fn histogram_unsigned_counts<T: HistogramUnsigned>(
    values: &[T],
    maximum: T,
    bins: usize,
) -> BuiltinResult<Vec<f64>> {
    let range = maximum.to_u128();
    let mut counts = vec![0.0; bins];
    for &value in values {
        counts[nearest_integer_bin(value.to_u128(), range, bins)] += 1.0;
    }
    Ok(counts)
}

pub(super) fn count_indexed_values(
    storage: &NumericStorage,
    zero_based: bool,
    counts: &mut [f64],
) -> BuiltinResult<()> {
    match storage {
        NumericStorage::F64(values) => count_float_indices(values, zero_based, counts),
        NumericStorage::F32(values) => {
            let values: Vec<f64> = values.iter().map(|value| f64::from(*value)).collect();
            count_float_indices(&values, zero_based, counts)
        }
        NumericStorage::I8(values) => count_signed_indices(values, zero_based, counts),
        NumericStorage::I16(values) => count_signed_indices(values, zero_based, counts),
        NumericStorage::I32(values) => count_signed_indices(values, zero_based, counts),
        NumericStorage::I64(values) => count_signed_indices(values, zero_based, counts),
        NumericStorage::U8(values) => count_unsigned_indices(values, zero_based, counts),
        NumericStorage::U16(values) => count_unsigned_indices(values, zero_based, counts),
        NumericStorage::U32(values) => count_unsigned_indices(values, zero_based, counts),
        NumericStorage::U64(values) => count_unsigned_indices(values, zero_based, counts),
    }
}

fn count_float_indices(values: &[f64], zero_based: bool, counts: &mut [f64]) -> BuiltinResult<()> {
    for &value in values {
        if !value.is_finite() || (value.round() - value).abs() > INTEGER_TOL {
            return Err(invalid(
                "indexed image values must be finite integer indices",
            ));
        }
        let rounded = value.round();
        let adjusted = if zero_based { rounded } else { rounded - 1.0 };
        if adjusted < 0.0 || adjusted >= counts.len() as f64 {
            return Err(invalid(format!(
                "indexed image value {value} is outside the colormap range"
            )));
        }
        counts[adjusted as usize] += 1.0;
    }
    Ok(())
}

fn count_signed_indices<T: HistogramSigned + std::fmt::Display>(
    values: &[T],
    zero_based: bool,
    counts: &mut [f64],
) -> BuiltinResult<()> {
    for &value in values {
        let raw = value.to_i128();
        let adjusted = if zero_based { raw } else { raw - 1 };
        if adjusted < 0 || adjusted as u128 >= counts.len() as u128 {
            return Err(invalid(format!(
                "indexed image value {value} is outside the colormap range"
            )));
        }
        counts[adjusted as usize] += 1.0;
    }
    Ok(())
}

fn count_unsigned_indices<T: HistogramUnsigned + std::fmt::Display>(
    values: &[T],
    zero_based: bool,
    counts: &mut [f64],
) -> BuiltinResult<()> {
    for &value in values {
        let raw = value.to_u128();
        let Some(adjusted) = raw.checked_sub(u128::from(!zero_based)) else {
            return Err(invalid(format!(
                "indexed image value {value} is outside the colormap range"
            )));
        };
        if adjusted >= counts.len() as u128 {
            return Err(invalid(format!(
                "indexed image value {value} is outside the colormap range"
            )));
        }
        counts[adjusted as usize] += 1.0;
    }
    Ok(())
}
