//! Native pair-to-interleaved transfer allocation, shared with admission.
use runmat_accelerate_api::{ProviderElementType, ProviderRepresentation, ProviderStorage};
use runmat_value::ComplexElement;

pub(crate) fn interleave<T: Copy>(pairs: &[ComplexElement<T>]) -> Result<Vec<T>, String> {
    let count = pairs
        .len()
        .checked_mul(2)
        .ok_or_else(|| "complex upload extent overflow".to_owned())?;
    let mut values = Vec::new();
    // Avoid iterator collection's geometric spare capacity: the physical
    // continuation bound is the requested native interleaved payload.
    values
        .try_reserve_exact(count)
        .map_err(|cause| format!("complex upload allocation failed: {cause}"))?;
    for &ComplexElement(real, imag) in pairs {
        values.push(real);
        values.push(imag);
    }
    Ok(values)
}

/// Runtime-created interleaving only. Provider word packing/fences remain
/// charged separately by the provider's upload_physical_evidence owner.
pub fn native_complex_upload_workspace(output: &ProviderRepresentation) -> Option<u64> {
    match (output.element_type, output.storage) {
        (
            ProviderElementType::ComplexF32 | ProviderElementType::ComplexF64,
            ProviderStorage::DenseComplexInterleaved,
        ) => output.checked_byte_len(),
        _ => None,
    }
}
