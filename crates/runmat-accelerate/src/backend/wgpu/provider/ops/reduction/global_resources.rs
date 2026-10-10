//! Pure physical-resource admission for the selected global extrema tree.
use runmat_accelerate_api::unsupported_provider_operation;

pub(super) fn admit(
    len: usize,
    source_buffer_bytes: u64,
    element_bytes: usize,
    elements_per_group: usize,
    max_input_per_pass: Option<usize>,
    max_storage_binding_bytes: u64,
    max_buffer_bytes: u64,
) -> anyhow::Result<()> {
    let unsupported = || {
        unsupported_provider_operation(
            "global extrema source or intermediate exceeds physical resource limits",
        )
    };
    if element_bytes == 0 || elements_per_group < 2 || max_input_per_pass == Some(0) {
        return Err(unsupported());
    }
    let element_bytes = u64::try_from(element_bytes).map_err(|_| unsupported())?;
    if len == 0 {
        // The empty scalar is uploaded; no source or reduction pipeline is bound.
        return if element_bytes <= max_buffer_bytes {
            Ok(())
        } else {
            Err(unsupported())
        };
    }
    if len > u32::MAX as usize || max_input_per_pass.is_none() {
        return Err(unsupported());
    }
    let logical_bytes = u64::try_from(len)
        .ok()
        .and_then(|value| value.checked_mul(element_bytes))
        .ok_or_else(unsupported)?;
    if source_buffer_bytes < logical_bytes || source_buffer_bytes > max_buffer_bytes {
        return Err(unsupported());
    }
    if len == 1 {
        // The emitter returns a shape-only view of the same scalar allocation.
        return Ok(());
    }
    let first_output_elements = len.div_ceil(elements_per_group).max(1);
    let first_output_bytes = u64::try_from(first_output_elements)
        .ok()
        .and_then(|value| value.checked_mul(element_bytes))
        .ok_or_else(unsupported)?;
    if source_buffer_bytes > max_storage_binding_bytes
        || first_output_bytes > max_storage_binding_bytes
        || first_output_bytes > max_buffer_bytes
    {
        return Err(unsupported());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn exact_source_and_first_intermediate_boundaries() {
        let admitted = |len, source, binding, buffer| {
            admit(len, source, 4, 2, Some(16), binding, buffer).is_ok()
        };
        assert!(admitted(8, 32, 32, 32));
        assert!(!admitted(8, 36, 32, 36)); // Entire physical source is bound.
        assert!(!admitted(8, 32, 15, 32)); // Entire 32-byte source is bound.
        assert!(!admitted(8, 32, 32, 15)); // Source cannot fit max buffer.
        assert!(!admitted(8, 28, 32, 32)); // Logical source exceeds backing.
        assert!(admitted(1, 4, 0, 4)); // Singleton does not bind the source.
        assert!(admitted(0, 0, 0, 4)); // Empty scalar upload.
        assert!(!admitted(0, 0, 0, 3));
        assert!(admit(3, 12, 4, 2, Some(16), 12, 12).is_ok());
        assert!(admit(3, 12, 4, 2, None, 12, 12).is_err());
        assert!(admit(3, 12, 4, 0, Some(16), 12, 12).is_err());
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn overwide_length_is_unsupported_before_any_allocation() {
        if let Ok(overwide) = usize::try_from(u64::from(u32::MAX) + 1) {
            assert!(admit(overwide, u64::MAX, 4, 2, Some(16), u64::MAX, u64::MAX).is_err());
        }
    }
}
