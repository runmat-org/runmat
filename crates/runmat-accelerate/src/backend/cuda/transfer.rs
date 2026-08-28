use anyhow::{anyhow, Result};
use runmat_accelerate_api::{
    HostIntegerDataOwned, HostNumericDataOwned, HostNumericDataView, NumericElementType,
};

pub(super) fn numeric_view_bytes(data: HostNumericDataView<'_>) -> &[u8] {
    match data {
        HostNumericDataView::F64(values) => bytemuck::cast_slice(values),
        HostNumericDataView::F32(values) => bytemuck::cast_slice(values),
        HostNumericDataView::I8(values) => bytemuck::cast_slice(values),
        HostNumericDataView::I16(values) => bytemuck::cast_slice(values),
        HostNumericDataView::I32(values) => bytemuck::cast_slice(values),
        HostNumericDataView::I64(values) => bytemuck::cast_slice(values),
        HostNumericDataView::U8(values) => values,
        HostNumericDataView::U16(values) => bytemuck::cast_slice(values),
        HostNumericDataView::U32(values) => bytemuck::cast_slice(values),
        HostNumericDataView::U64(values) => bytemuck::cast_slice(values),
    }
}

pub(super) fn numeric_owned_bytes_mut(data: &mut HostNumericDataOwned) -> &mut [u8] {
    match data {
        HostNumericDataOwned::F64(values) => bytemuck::cast_slice_mut(values),
        HostNumericDataOwned::F32(values) => bytemuck::cast_slice_mut(values),
        HostNumericDataOwned::I8(values) => bytemuck::cast_slice_mut(values),
        HostNumericDataOwned::I16(values) => bytemuck::cast_slice_mut(values),
        HostNumericDataOwned::I32(values) => bytemuck::cast_slice_mut(values),
        HostNumericDataOwned::I64(values) => bytemuck::cast_slice_mut(values),
        HostNumericDataOwned::U8(values) => values,
        HostNumericDataOwned::U16(values) => bytemuck::cast_slice_mut(values),
        HostNumericDataOwned::U32(values) => bytemuck::cast_slice_mut(values),
        HostNumericDataOwned::U64(values) => bytemuck::cast_slice_mut(values),
    }
}

pub(super) fn zeroed_numeric(
    element_type: NumericElementType,
    length: usize,
) -> HostNumericDataOwned {
    match element_type {
        NumericElementType::F64 => HostNumericDataOwned::F64(vec![0.0; length]),
        NumericElementType::F32 => HostNumericDataOwned::F32(vec![0.0; length]),
        NumericElementType::I8 => HostNumericDataOwned::I8(vec![0; length]),
        NumericElementType::I16 => HostNumericDataOwned::I16(vec![0; length]),
        NumericElementType::I32 => HostNumericDataOwned::I32(vec![0; length]),
        NumericElementType::I64 => HostNumericDataOwned::I64(vec![0; length]),
        NumericElementType::U8 => HostNumericDataOwned::U8(vec![0; length]),
        NumericElementType::U16 => HostNumericDataOwned::U16(vec![0; length]),
        NumericElementType::U32 => HostNumericDataOwned::U32(vec![0; length]),
        NumericElementType::U64 => HostNumericDataOwned::U64(vec![0; length]),
    }
}

pub(super) fn numeric_to_integer(data: HostNumericDataOwned) -> Result<HostIntegerDataOwned> {
    match data {
        HostNumericDataOwned::I8(values) => Ok(HostIntegerDataOwned::I8(values)),
        HostNumericDataOwned::I16(values) => Ok(HostIntegerDataOwned::I16(values)),
        HostNumericDataOwned::I32(values) => Ok(HostIntegerDataOwned::I32(values)),
        HostNumericDataOwned::I64(values) => Ok(HostIntegerDataOwned::I64(values)),
        HostNumericDataOwned::U8(values) => Ok(HostIntegerDataOwned::U8(values)),
        HostNumericDataOwned::U16(values) => Ok(HostIntegerDataOwned::U16(values)),
        HostNumericDataOwned::U32(values) => Ok(HostIntegerDataOwned::U32(values)),
        HostNumericDataOwned::U64(values) => Ok(HostIntegerDataOwned::U64(values)),
        HostNumericDataOwned::F64(_) | HostNumericDataOwned::F32(_) => {
            Err(anyhow!("CUDA handle does not contain integer storage"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_numeric_classes_keep_their_native_byte_width() {
        let cases = [
            (NumericElementType::F64, 8),
            (NumericElementType::F32, 4),
            (NumericElementType::I8, 1),
            (NumericElementType::I16, 2),
            (NumericElementType::I32, 4),
            (NumericElementType::I64, 8),
            (NumericElementType::U8, 1),
            (NumericElementType::U16, 2),
            (NumericElementType::U32, 4),
            (NumericElementType::U64, 8),
        ];
        for (element_type, width) in cases {
            let mut data = zeroed_numeric(element_type, 3);
            assert_eq!(numeric_owned_bytes_mut(&mut data).len(), width * 3);
        }
    }
}
