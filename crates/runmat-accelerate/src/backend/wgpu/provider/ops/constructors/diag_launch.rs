//! Pure launch and storage-domain policy for diagonal constructor kernels.

use anyhow::{anyhow, ensure, Result};

pub(super) fn vector_dispatch_len(
    input_len: usize,
    out_rows: usize,
    out_cols: usize,
    offset: isize,
) -> usize {
    // Sized construction may receive a longer resident vector than the
    // requested matrix can represent. Those trailing values are outside the
    // operation's domain and must not enlarge its dispatch.
    input_len.min(super::diag_length(out_rows, out_cols, offset))
}

pub(super) fn ensure_storage_domain(
    input_allocated_bytes: u64,
    output_elements: usize,
    dispatched_elements: usize,
    element_size: usize,
    limits: &wgpu::Limits,
) -> Result<()> {
    let output_bytes = u64::try_from(output_elements)
        .ok()
        .and_then(|elements| elements.checked_mul(element_size as u64))
        .ok_or_else(|| anyhow!("diag: output byte extent exceeds GPU limits"))?;
    ensure!(
        output_bytes <= limits.max_buffer_size,
        "diag: output exceeds the enabled GPU buffer limit"
    );
    if dispatched_elements != 0 {
        let binding_limit =
            u64::from(limits.max_storage_buffer_binding_size).min(limits.max_buffer_size);
        ensure!(
            input_allocated_bytes <= binding_limit && output_bytes <= binding_limit,
            "diag: storage binding exceeds enabled GPU limits"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn vector_dispatch_covers_only_the_representable_output_diagonal() {
        assert_eq!(vector_dispatch_len(5, 3, 3, 0), 3);
        assert_eq!(vector_dispatch_len(5, 5, 3, -1), 3);
        assert_eq!(vector_dispatch_len(usize::MAX, 2, 2, 0), 2);
        assert_eq!(vector_dispatch_len(5, 3, 3, 3), 0);
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn every_bindable_floating_diagonal_fits_one_minimum_lane_dispatch() {
        let largest_storage_binding = u64::from(u32::MAX);
        assert!(32_767u64.pow(2) * 4 <= largest_storage_binding);
        assert!(32_768u64.pow(2) * 4 > largest_storage_binding);
        assert!(
            u64::from(crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS) > 32_767,
            "even a one-lane workgroup covers every bindable floating diagonal"
        );
    }
}
