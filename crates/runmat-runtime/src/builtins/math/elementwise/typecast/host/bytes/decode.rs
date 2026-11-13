use runmat_value::NumericStorage;

use super::super::super::target::Representation;

pub(super) fn storage(bytes: &[u8], target: Representation) -> NumericStorage {
    macro_rules! decode {
        ($ty:ty, $variant:ident) => {{
            let values = bytes
                .chunks_exact(std::mem::size_of::<$ty>())
                .map(|chunk| <$ty>::from_ne_bytes(chunk.try_into().expect("exact chunk")))
                .collect();
            NumericStorage::$variant(values)
        }};
    }
    match target {
        Representation::Numeric(runmat_value::NumericDType::F64) => decode!(f64, F64),
        Representation::Numeric(runmat_value::NumericDType::F32) => decode!(f32, F32),
        Representation::Numeric(runmat_value::NumericDType::I8) => {
            NumericStorage::I8(bytes.iter().map(|byte| *byte as i8).collect())
        }
        Representation::Numeric(runmat_value::NumericDType::I16) => decode!(i16, I16),
        Representation::Numeric(runmat_value::NumericDType::I32) => decode!(i32, I32),
        Representation::Numeric(runmat_value::NumericDType::I64) => decode!(i64, I64),
        Representation::Numeric(runmat_value::NumericDType::U8) => {
            NumericStorage::U8(bytes.to_vec())
        }
        Representation::Numeric(runmat_value::NumericDType::U16) => decode!(u16, U16),
        Representation::Numeric(runmat_value::NumericDType::U32) => decode!(u32, U32),
        Representation::Numeric(runmat_value::NumericDType::U64) => decode!(u64, U64),
        Representation::Logical => {
            NumericStorage::U8(bytes.iter().map(|byte| u8::from(*byte != 0)).collect())
        }
    }
}
