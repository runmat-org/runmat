use super::InProcessNumericData;
use runmat_accelerate_api::HostIntegerDataOwned;

impl InProcessNumericData {
    /// Vec capacity is the allocator request; initialized length is separately
    /// the payload/view extent. Neither is an allocator's hidden granularity.
    pub(super) fn requested_byte_len(&self) -> u64 {
        fn bytes<T>(values: &Vec<T>) -> u64 {
            (values.capacity() * std::mem::size_of::<T>()) as u64
        }
        match self {
            Self::F64(data) => bytes(data),
            Self::F32(data) => bytes(data),
            Self::Integer(data) => match data {
                HostIntegerDataOwned::I8(data) => bytes(data),
                HostIntegerDataOwned::I16(data) => bytes(data),
                HostIntegerDataOwned::I32(data) => bytes(data),
                HostIntegerDataOwned::I64(data) => bytes(data),
                HostIntegerDataOwned::U8(data) => bytes(data),
                HostIntegerDataOwned::U16(data) => bytes(data),
                HostIntegerDataOwned::U32(data) => bytes(data),
                HostIntegerDataOwned::U64(data) => bytes(data),
            },
        }
    }
}
