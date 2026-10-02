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
    backings: Option<[u64; 2]>,
) -> Option<(ProviderResourceEstimate, u64)> {
    let [signal, kernel] = inputs else {
        return None;
    };
    let signal_elements = signal.checked_element_count()?;
    let kernel_elements = kernel.checked_element_count()?;

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
        let peak = match backings {
            Some([a, b]) => Some(
                a.checked_add(b)?
                    .checked_add(physical_output)?
                    .checked_add(uniform)?,
            ),
            None => None,
        };
        return Some((
            ProviderResourceEstimate {
                transient_bytes: peak,
                output_bytes: Some(physical_output),
                dispatches: Some(u32::from(output_elements != 0)),
            },
            physical_output.max(uniform),
        ));
    }

    let normal_route = signal_elements != 0
        && kernel_elements != 0
        && !(identity == ConvolutionIdentity::Vector
            && route.mode == ConvolutionMode::Valid
            && signal_elements < kernel_elements);
    let (peak, largest_route_allocation) = if normal_route {
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
        let full_bytes = full_elements.checked_mul(8)?;
        let peak = match backings {
            Some([a, b]) => Some(
                a.checked_add(b)?
                    .checked_mul(2)?
                    .checked_add(full_bytes)?
                    .checked_add(output_bytes)?,
            ),
            None => None,
        };
        (peak, full_bytes.max(output_bytes))
    } else {
        // Empty and short-Valid paths allocate only the published result.
        let peak = match backings {
            Some([a, b]) => Some(a.checked_add(b)?.checked_add(output_bytes)?),
            None => None,
        };
        (peak, output_bytes)
    };
    Some((
        ProviderResourceEstimate {
            transient_bytes: peak,
            output_bytes: Some(output_bytes),
            dispatches: Some(0),
        },
        largest_route_allocation,
    ))
}
