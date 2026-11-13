//! Physical integer buffer geometry shared by WGPU allocation and admission.
use runmat_accelerate_api::{IntegerElementType, ProviderElementType, ProviderRepresentation};

/// Requested raw-word storage, including WGPU's nonzero empty binding buffer.
/// Payload length remains separate; this is not driver allocation granularity.
pub(crate) fn raw_word_requested_bytes(words: u64) -> Option<u64> {
    words.checked_mul(4).map(|bytes| bytes.max(4))
}

pub(crate) const fn integer_words_per_element(element_type: IntegerElementType) -> usize {
    match element_type {
        IntegerElementType::I64 | IntegerElementType::U64 => 2,
        IntegerElementType::I8
        | IntegerElementType::I16
        | IntegerElementType::I32
        | IntegerElementType::U8
        | IntegerElementType::U16
        | IntegerElementType::U32 => 1,
    }
}

/// Provider allocation after upload, irrespective of the input's current
/// residency. This does not describe the native host payload or transfer size.
pub(crate) fn checked_allocation_bytes(value: &ProviderRepresentation) -> Option<u64> {
    let integer = match value.element_type {
        ProviderElementType::I8 => Some(IntegerElementType::I8),
        ProviderElementType::I16 => Some(IntegerElementType::I16),
        ProviderElementType::I32 => Some(IntegerElementType::I32),
        ProviderElementType::I64 => Some(IntegerElementType::I64),
        ProviderElementType::U8 => Some(IntegerElementType::U8),
        ProviderElementType::U16 => Some(IntegerElementType::U16),
        ProviderElementType::U32 => Some(IntegerElementType::U32),
        ProviderElementType::U64 => Some(IntegerElementType::U64),
        ProviderElementType::Logical => return None, // Physical float precision is absent.
        ProviderElementType::F32
        | ProviderElementType::F64
        | ProviderElementType::ComplexF32
        | ProviderElementType::ComplexF64 => None,
    };
    let width = integer.map_or_else(
        || value.element_type.byte_width(),
        |ty| integer_words_per_element(ty) as u64 * std::mem::size_of::<u32>() as u64,
    );
    value.checked_element_count()?.checked_mul(width)
}

/// Storage reserved by the WGPU numeric upload path. Empty uploads use the
/// provider's floating-element buffer even for integer and complex payloads.
#[cfg(feature = "wgpu")]
pub(crate) fn checked_upload_allocation_bytes(
    value: &ProviderRepresentation,
    precision: runmat_accelerate_api::ProviderPrecision,
) -> Option<u64> {
    use runmat_accelerate_api::ProviderStorage;
    let expected_storage = match value.element_type {
        ProviderElementType::ComplexF32 | ProviderElementType::ComplexF64 => {
            ProviderStorage::DenseComplexInterleaved
        }
        _ => ProviderStorage::DenseReal,
    };
    if value.storage != expected_storage {
        return None;
    }
    let count = value.checked_element_count()?;
    if count == 0 {
        return Some(ProviderElementType::from(precision).byte_width());
    }
    match value.element_type {
        ProviderElementType::Logical => {
            count.checked_mul(ProviderElementType::from(precision).byte_width())
        }
        ProviderElementType::I8
        | ProviderElementType::I16
        | ProviderElementType::I32
        | ProviderElementType::I64
        | ProviderElementType::U8
        | ProviderElementType::U16
        | ProviderElementType::U32
        | ProviderElementType::U64 => checked_allocation_bytes(value),
        ProviderElementType::F32
        | ProviderElementType::F64
        | ProviderElementType::ComplexF32
        | ProviderElementType::ComplexF64 => value.checked_byte_len(),
    }
}

/// Maximum physical native upload allocation over every admitted count,
/// including zero. An F32 bound on an F64-preferred owner can require more
/// storage for its empty realization than for one nonempty F32 element.
#[cfg(feature = "wgpu")]
pub(crate) fn checked_upload_bound_allocation_bytes(
    bound: &runmat_accelerate_api::ProviderTransferOutputBound,
    precision: runmat_accelerate_api::ProviderPrecision,
) -> Option<u64> {
    use runmat_accelerate_api::ProviderStorage;
    let storage = match bound.element_type {
        ProviderElementType::F32 | ProviderElementType::F64 => ProviderStorage::DenseReal,
        ProviderElementType::ComplexF32 | ProviderElementType::ComplexF64 => {
            ProviderStorage::DenseComplexInterleaved
        }
        _ => return None,
    };
    if bound.storage != storage {
        return None;
    }
    bound
        .max_elements
        .checked_mul(bound.element_type.byte_width())
        .map(|bytes| bytes.max(ProviderElementType::from(precision).byte_width()))
}
