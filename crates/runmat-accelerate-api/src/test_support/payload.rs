use crate::{HostNumericDataOwned, NumericElementType};

pub(super) fn zeroed(element: NumericElementType, count: usize) -> HostNumericDataOwned {
    match element {
        NumericElementType::F64 => HostNumericDataOwned::F64(vec![0.0; count]),
        NumericElementType::F32 => HostNumericDataOwned::F32(vec![0.0; count]),
        NumericElementType::I8 => HostNumericDataOwned::I8(vec![0; count]),
        NumericElementType::I16 => HostNumericDataOwned::I16(vec![0; count]),
        NumericElementType::I32 => HostNumericDataOwned::I32(vec![0; count]),
        NumericElementType::I64 => HostNumericDataOwned::I64(vec![0; count]),
        NumericElementType::U8 => HostNumericDataOwned::U8(vec![0; count]),
        NumericElementType::U16 => HostNumericDataOwned::U16(vec![0; count]),
        NumericElementType::U32 => HostNumericDataOwned::U32(vec![0; count]),
        NumericElementType::U64 => HostNumericDataOwned::U64(vec![0; count]),
    }
}

pub(super) fn address(data: &mut HostNumericDataOwned) -> u64 {
    if data.is_empty() {
        return 0;
    }
    match data {
        HostNumericDataOwned::F64(data) => data.as_mut_ptr() as u64,
        HostNumericDataOwned::F32(data) => data.as_mut_ptr() as u64,
        HostNumericDataOwned::I8(data) => data.as_mut_ptr() as u64,
        HostNumericDataOwned::I16(data) => data.as_mut_ptr() as u64,
        HostNumericDataOwned::I32(data) => data.as_mut_ptr() as u64,
        HostNumericDataOwned::I64(data) => data.as_mut_ptr() as u64,
        HostNumericDataOwned::U8(data) => data.as_mut_ptr() as u64,
        HostNumericDataOwned::U16(data) => data.as_mut_ptr() as u64,
        HostNumericDataOwned::U32(data) => data.as_mut_ptr() as u64,
        HostNumericDataOwned::U64(data) => data.as_mut_ptr() as u64,
    }
}
