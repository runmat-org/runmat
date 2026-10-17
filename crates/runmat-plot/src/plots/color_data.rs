//! Shared renderer-domain conversion for authoritative numeric color sources.

use glam::{Vec3, Vec4};
use runmat_value::{NumericDType, NumericScalar};

use super::surface::ColorMap;

pub(crate) fn rgb_component_valid(value: NumericScalar) -> bool {
    match value {
        NumericScalar::F64(value) => value.is_nan() || (0.0..=1.0).contains(&value),
        NumericScalar::F32(value) => value.is_nan() || (0.0..=1.0).contains(&value),
        _ => true,
    }
}

pub(crate) fn rgb_component(value: NumericScalar) -> f32 {
    let dtype = value.numeric_dtype();
    let raw = value.materialize_f64();
    let (low, high) = match dtype {
        NumericDType::F64 | NumericDType::F32 => (0.0, 1.0),
        NumericDType::I8 => (i8::MIN as f64, i8::MAX as f64),
        NumericDType::I16 => (i16::MIN as f64, i16::MAX as f64),
        NumericDType::I32 => (i32::MIN as f64, i32::MAX as f64),
        NumericDType::I64 => (i64::MIN as f64, i64::MAX as f64),
        NumericDType::U8 => (0.0, u8::MAX as f64),
        NumericDType::U16 => (0.0, u16::MAX as f64),
        NumericDType::U32 => (0.0, u32::MAX as f64),
        NumericDType::U64 => (0.0, u64::MAX as f64),
    };
    ((raw - low) / (high - low)) as f32
}

pub(crate) fn indexed_color(
    value: NumericScalar,
    colormap: &ColorMap,
    colormap_length: usize,
    limits: Option<(f64, f64)>,
    alpha: f32,
) -> Vec4 {
    let raw = value.materialize_f64();
    if raw.is_nan() {
        return Vec4::ZERO;
    }
    let t = if let Some((low, high)) = limits {
        ((raw - low) / (high - low).max(f64::MIN_POSITIVE)) as f32
    } else {
        let index = match value.numeric_dtype() {
            NumericDType::F64 | NumericDType::F32 => raw.floor() - 1.0,
            _ => raw,
        };
        (index / colormap_length.saturating_sub(1).max(1) as f64) as f32
    };
    let Vec3 { x, y, z } = colormap.map_value(t.clamp(0.0, 1.0));
    Vec4::new(x, y, z, alpha)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_indices_respect_floating_one_based_and_integer_zero_based_classes() {
        let map = ColorMap::Gray;
        for (floating, integer) in [(1.0, 0_u8), (2.0, 1_u8)] {
            assert_eq!(
                indexed_color(NumericScalar::F64(floating), &map, 4, None, 1.0),
                indexed_color(NumericScalar::U8(integer), &map, 4, None, 1.0)
            );
        }
        assert_eq!(
            indexed_color(NumericScalar::F64(f64::NAN), &map, 4, None, 1.0),
            Vec4::ZERO
        );
    }
}
