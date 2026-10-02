//! Checked live-payload requests of the implemented physical hooks.
//!
//! This is not a device-memory availability probe: pooled capacity, allocator
//! metadata, and unrelated live allocations are outside a single hook's route.

use super::*;

// Four u32 fields in the WGPU Conv1dParams physical uniform.
const CONV1D_UNIFORM_BYTES: u64 = 4 * std::mem::size_of::<u32>() as u64;
#[cfg(feature = "wgpu")]
const _: [(); CONV1D_UNIFORM_BYTES as usize] =
    [(); std::mem::size_of::<crate::backend::wgpu::params::Conv1dParams>()];

pub(super) fn estimate(
    identity: ConvolutionIdentity,
    route: ProviderConvolutionWorkload,
    inputs: &[ProviderRepresentation],
    output_elements: u64,
    output_bytes: u64,
    wgpu: bool,
) -> Option<ProviderResourceEstimate> {
    let [signal, kernel] = inputs else {
        return None;
    };
    let signal_elements = signal.checked_element_count()?;
    let kernel_elements = kernel.checked_element_count()?;
    let signal_bytes = signal.checked_byte_len()?;
    let kernel_bytes = kernel.checked_byte_len()?;
    let retained_inputs = signal_bytes.checked_add(kernel_bytes)?;

    if wgpu {
        if identity != ConvolutionIdentity::Vector {
            return None;
        }
        let physical_output = output_bytes.max(signal.element_type.byte_width());
        let uniform = if output_elements == 0 {
            0
        } else {
            CONV1D_UNIFORM_BYTES
        };
        // An empty view may refer to a larger resident allocation. Shape-only
        // facts cannot prove its full backing footprint.
        let peak = if signal_elements == 0 || kernel_elements == 0 {
            None
        } else {
            Some(
                retained_inputs
                    .checked_add(physical_output)?
                    .checked_add(uniform)?,
            )
        };
        return Some(ProviderResourceEstimate {
            transient_bytes: peak,
            output_bytes: Some(physical_output),
            dispatches: Some(u32::from(output_elements != 0)),
        });
    }

    let normal_route = signal_elements != 0
        && kernel_elements != 0
        && !(identity == ConvolutionIdentity::Vector
            && route.mode == ConvolutionMode::Valid
            && signal_elements < kernel_elements);
    let peak = if normal_route {
        // InProcess clones both registered f64 operands. The full result and
        // shaped result coexist until the latter is moved into its registry.
        let full_elements = match identity {
            ConvolutionIdentity::Vector => signal_elements
                .checked_add(kernel_elements)?
                .checked_sub(1)?,
            ConvolutionIdentity::Matrix => {
                let (ar, ac) = matrix_dims(&signal.shape)?;
                let (br, bc) = matrix_dims(&kernel.shape)?;
                ar.checked_add(br)?
                    .checked_sub(1)?
                    .checked_mul(ac.checked_add(bc)?.checked_sub(1)?)?
            }
        };
        retained_inputs
            .checked_mul(2)?
            .checked_add(full_elements.checked_mul(8)?)?
            .checked_add(output_bytes)?
    } else {
        // Empty and short-Valid paths allocate only the published result.
        retained_inputs.checked_add(output_bytes)?
    };
    Some(ProviderResourceEstimate {
        transient_bytes: Some(peak),
        output_bytes: Some(output_bytes),
        dispatches: Some(0),
    })
}
