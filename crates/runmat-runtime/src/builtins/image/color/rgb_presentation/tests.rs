use super::*;

#[test]
fn typed_rgb_output_and_operation_scratch_bounds_cover_scalar_and_classes() {
    let one = ColorLayout::Truecolor { rows: 1, cols: 1 };
    let map = ColorLayout::Colormap { rows: 2 };
    assert_eq!(
        output_dtype(RgbConversion::Gray, one, Some(1), NumericDType::F32),
        NumericDType::F64
    );
    assert_eq!(
        output_dtype(RgbConversion::Gray, one, Some(1), NumericDType::U8),
        NumericDType::U8
    );
    assert_eq!(
        output_dtype(RgbConversion::Gray, map, Some(2), NumericDType::F64),
        NumericDType::F64
    );
    assert_eq!(
        output_dtype(RgbConversion::Hsv, one, Some(1), NumericDType::U16),
        NumericDType::F64
    );
    assert_eq!(
        host_conversion_scratch(RgbConversion::Gray, NumericDType::U8, Some(4)),
        ResourceByteBound::Known(32)
    );
    assert_eq!(
        host_conversion_scratch(RgbConversion::Hsv, NumericDType::F64, Some(4)),
        ResourceByteBound::Known(96)
    );
    assert_eq!(
        host_conversion_scratch(RgbConversion::Hsv, NumericDType::F32, Some(4)),
        ResourceByteBound::Known(192)
    );
    assert_eq!(
        host_conversion_scratch(RgbConversion::Hsv, NumericDType::F32, Some(u64::MAX)),
        ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoScratchBound)
    );
}
