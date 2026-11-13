use super::*;

mod reflection;
mod resources;
use reflection::reflection_stride;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct ConvParams {
    pub input_shape: [u32; 4],
    pub output_shape: [u32; 4],
    pub input_strides: [u32; 4],
    pub output_strides: [u32; 4],
    pub rank: u32,
    pub span_axis: u32,
    pub channel_axis: u32,
    pub batch_axis: u32,
    pub time_axis: u32,
    pub input_len: u32,
    pub output_len: u32,
    pub input_channels: u32,
    pub output_channels: u32,
    pub filter_size: u32,
    pub batch_size: u32,
    pub time_size: u32,
    pub stride: u32,
    pub dilation: u32,
    pub padding_left: u32,
    pub padding_mode: u32,
    /// Native bits of the admitted padding scalar. WGSL views this slot as
    /// two `u32` words for f32 or one aligned `f64` for f64.
    pub padding_bits: u64,
    pub chunk_offset: u32,
    pub chunk_count: u32,
    pub reflection_step: u32,
    pub reflection_inverse: u32,
    pub _reserved: [u32; 2],
}

#[derive(Clone, Copy)]
pub(super) struct Geometry {
    pub params: ConvParams,
    pub input_elements: usize,
    pub output_elements: usize,
    pub weight_elements: usize,
    pub bias_elements: usize,
}

#[derive(Clone, Copy)]
pub(super) struct Chunk {
    pub offset: u32,
    pub count: u32,
    pub workgroups: u32,
}

pub(super) fn next_chunk(total: u32, offset: u32) -> Option<Chunk> {
    if offset == total {
        return None;
    }
    let lanes = crate::backend::wgpu::shaders::deep_learning_convolution::CONV_WORKGROUP_SIZE;
    let capacity = crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS.saturating_mul(lanes);
    let count = (total - offset).min(capacity);
    Some(Chunk {
        offset,
        count,
        workgroups: count.div_ceil(lanes),
    })
}

fn product(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1usize, |total, &extent| total.checked_mul(extent))
}

fn physical_shape_and_strides(shape: &[usize]) -> Result<([u32; 4], [u32; 4])> {
    let mut physical = [1u32; 4];
    let mut strides = [1u32; 4];
    let mut stride = 1usize;
    for (axis, &extent) in shape.iter().enumerate() {
        physical[axis] = u32::try_from(extent)?;
        strides[axis] = u32::try_from(stride)?;
        stride = stride
            .checked_mul(extent)
            .ok_or_else(|| anyhow!("convolution physical stride overflows"))?;
    }
    Ok((physical, strides))
}

impl Geometry {
    /// Authenticate an already-lowered descriptor. This checks physical
    /// address safety; semantic padding/options still belong to host lowering.
    /// `None` means that the descriptor is semantically valid but exceeds the
    /// signed coordinate/index range of this shader implementation. Invalid
    /// descriptors remain errors, so callers can report ResourceLimit without
    /// treating malformed requests as a legal fallback.
    pub fn new(
        request: &DeepLearningConv1DRequest,
        precision: NumericPrecision,
    ) -> Result<Option<Self>> {
        let rank = request.input_shape.len();
        ensure!(
            (2..=4).contains(&rank) && request.output_shape.len() == rank,
            "convolution input/output rank differs from admitted geometry"
        );
        let mut axes = vec![request.span_axis, request.channel_axis];
        axes.extend(request.batch_axis);
        axes.extend(request.time_axis);
        axes.sort_unstable();
        ensure!(
            axes == (0..rank).collect::<Vec<_>>(),
            "convolution physical axes are not a complete unique layout"
        );
        let input_len = request.input_shape[request.span_axis];
        let input_channels = request.input_shape[request.channel_axis];
        let [filter_size, weight_channels, output_channels] = request.weights_shape;
        ensure!(
            input_len > 0
                && input_channels > 0
                && filter_size > 0
                && output_channels > 0
                && weight_channels == input_channels
                && request.bias_shape == [1, output_channels]
                && request.stride > 0
                && request.dilation > 0,
            "convolution physical input, parameter, or stride geometry is invalid"
        );
        let effective_filter = filter_size
            .checked_sub(1)
            .and_then(|size| size.checked_mul(request.dilation))
            .and_then(|size| size.checked_add(1))
            .ok_or_else(|| anyhow!("convolution effective filter overflows"))?;
        let padded = input_len
            .checked_add(request.padding_left)
            .and_then(|size| size.checked_add(request.padding_right))
            .ok_or_else(|| anyhow!("convolution padded span overflows"))?;
        let output_len = if padded < effective_filter {
            0
        } else {
            (padded - effective_filter) / request.stride + 1
        };
        ensure!(
            request.output_shape[request.span_axis] == output_len
                && request.output_shape[request.channel_axis] == output_channels
                && (0..rank).all(|axis| {
                    axis == request.span_axis
                        || axis == request.channel_axis
                        || request.input_shape[axis] == request.output_shape[axis]
                }),
            "convolution output differs from physical input/parameter geometry"
        );
        ensure!(
            !matches!(
                request.padding_value,
                DeepLearningConv1DPaddingValue::SymmetricExcludeEdge
            ) || input_len > 1
                || (request.padding_left == 0 && request.padding_right == 0),
            "symmetric-exclude-edge convolution requires two input samples"
        );
        let (padding_mode, padding_bits) = match request.padding_value {
            DeepLearningConv1DPaddingValue::Constant(value) => {
                ensure!(
                    value.is_finite(),
                    "convolution padding value must be finite"
                );
                match precision {
                    NumericPrecision::F32 => ensure!(
                        (value as f32).is_finite(),
                        "convolution padding value exceeds single precision"
                    ),
                    NumericPrecision::F64 => {}
                }
                (
                    0u32,
                    match precision {
                        NumericPrecision::F32 => (value as f32).to_bits() as u64,
                        NumericPrecision::F64 => value.to_bits(),
                    },
                )
            }
            DeepLearningConv1DPaddingValue::Replicate => (1, 0),
            DeepLearningConv1DPaddingValue::SymmetricIncludeEdge => (2, 0),
            DeepLearningConv1DPaddingValue::SymmetricExcludeEdge => (3, 0),
        };
        let (Some(input_elements), Some(output_elements), Some(weight_elements)) = (
            product(&request.input_shape),
            product(&request.output_shape),
            product(&request.weights_shape),
        ) else {
            return Ok(None);
        };
        let bias_elements = output_channels;
        // Signed shader coordinates permit inverse padding without floating
        // atomics. These bounds are deliberately checked independently of
        // adapter byte limits, which can be even narrower.
        let max_window = if output_len == 0 {
            0
        } else {
            let Some(window) = (output_len - 1)
                .checked_mul(request.stride)
                .and_then(|value| {
                    (filter_size - 1)
                        .checked_mul(request.dilation)
                        .and_then(|filter| value.checked_add(filter))
                })
            else {
                return Ok(None);
            };
            window
        };
        let signed_limit = (i32::MAX as usize) / 4;
        if ![
            input_len,
            output_len,
            input_channels,
            output_channels,
            filter_size,
            request.stride,
            request.dilation,
            request.padding_left,
            request.padding_right,
            max_window,
        ]
        .iter()
        .all(|&value| value <= signed_limit)
            || ![
                input_elements,
                output_elements,
                weight_elements,
                bias_elements,
            ]
            .iter()
            .all(|&value| value <= u32::MAX as usize)
            || request
                .input_shape
                .iter()
                .chain(request.output_shape.iter())
                .any(|&extent| extent > u32::MAX as usize)
        {
            return Ok(None);
        }
        // An empty later dimension can make the total zero while an earlier
        // column-major prefix stride is still wider than the shader index.
        for shape in [&request.input_shape, &request.output_shape] {
            let mut prefix = 1usize;
            for &extent in shape {
                if prefix > u32::MAX as usize {
                    return Ok(None);
                }
                prefix = match prefix.checked_mul(extent) {
                    Some(next) => next,
                    None => return Ok(None),
                };
            }
        }
        let (input_shape, input_strides) = physical_shape_and_strides(&request.input_shape)?;
        let (output_shape, output_strides) = physical_shape_and_strides(&request.output_shape)?;
        let (reflection_step, reflection_inverse) = reflection_stride(request, input_len);
        Ok(Some(Self {
            params: ConvParams {
                input_shape,
                output_shape,
                input_strides,
                output_strides,
                rank: u32::try_from(rank)?,
                span_axis: u32::try_from(request.span_axis)?,
                channel_axis: u32::try_from(request.channel_axis)?,
                batch_axis: request
                    .batch_axis
                    .map(u32::try_from)
                    .transpose()?
                    .unwrap_or(u32::MAX),
                time_axis: request
                    .time_axis
                    .map(u32::try_from)
                    .transpose()?
                    .unwrap_or(u32::MAX),
                input_len: u32::try_from(input_len)?,
                output_len: u32::try_from(output_len)?,
                input_channels: u32::try_from(input_channels)?,
                output_channels: u32::try_from(output_channels)?,
                filter_size: u32::try_from(filter_size)?,
                batch_size: request
                    .batch_axis
                    .map(|axis| u32::try_from(request.input_shape[axis]))
                    .transpose()?
                    .unwrap_or(1),
                time_size: request
                    .time_axis
                    .map(|axis| u32::try_from(request.input_shape[axis]))
                    .transpose()?
                    .unwrap_or(1),
                stride: u32::try_from(request.stride)?,
                dilation: u32::try_from(request.dilation)?,
                padding_left: u32::try_from(request.padding_left)?,
                padding_mode,
                padding_bits,
                chunk_offset: 0,
                chunk_count: 0,
                reflection_step,
                reflection_inverse,
                _reserved: [0; 2],
            },
            input_elements,
            output_elements,
            weight_elements,
            bias_elements,
        }))
    }
}
