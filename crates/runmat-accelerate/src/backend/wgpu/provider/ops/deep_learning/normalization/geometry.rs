use super::*;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct NormParams {
    pub(super) shape: [u32; 4],
    pub(super) rank: u32,
    pub(super) reduce_mask: u32,
    pub(super) groups: u32,
    pub(super) group_size: u32,
    pub(super) channel_axis: u32,
    pub(super) total: u32,
    pub(super) chunk_offset: u32,
    pub(super) chunk_count: u32,
    pub(super) epsilon_low: u32,
    pub(super) epsilon_high: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Chunk {
    pub(super) offset: u32,
    pub(super) count: u32,
    pub(super) workgroups: u32,
}

pub(super) fn next_chunk(total: u32, offset: u32) -> Option<Chunk> {
    let lanes =
        crate::backend::wgpu::shaders::deep_learning_normalization::NORMALIZATION_WORKGROUP_SIZE;
    let capacity = crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS.saturating_mul(lanes);
    if offset == total {
        return None;
    }
    let count = (total - offset).min(capacity);
    Some(Chunk {
        offset,
        count,
        workgroups: count.div_ceil(lanes),
    })
}

pub(super) fn shape_count(shape: &[usize]) -> Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |count, &extent| count.checked_mul(extent))
        .ok_or_else(|| anyhow!("normalization shape product overflows"))
}

pub(super) fn geometry(request: &DeepLearningLayerNormRequest) -> Result<(usize, usize)> {
    ensure!(
        (1..=4).contains(&request.shape.len()),
        "normalization rank must be one through four"
    );
    ensure!(
        request.channel_axis < request.shape.len(),
        "normalization channel axis exceeds rank"
    );
    ensure!(
        request.reduce_mask != 0 && (request.reduce_mask as usize >> request.shape.len()) == 0,
        "normalization reduction mask is invalid"
    );
    ensure!(
        request.reduce_mask & (1 << request.channel_axis) != 0,
        "normalization must reduce its channel axis"
    );
    ensure!(
        request.epsilon.is_finite() && request.epsilon > 0.0,
        "normalization epsilon must be positive and finite"
    );
    let channels = request.shape[request.channel_axis];
    let mut groups = 1usize;
    let mut group_size = 1usize;
    for (axis, &extent) in request.shape.iter().enumerate() {
        let target = if request.reduce_mask & (1 << axis) != 0 {
            &mut group_size
        } else {
            &mut groups
        };
        *target = target
            .checked_mul(extent)
            .ok_or_else(|| anyhow!("normalization geometry overflows"))?;
    }
    ensure!(
        channels != 0 && group_size != 0,
        "normalization reduction is empty"
    );
    let row = [1, channels];
    let column = [channels, 1];
    ensure!(
        request.scale_shape == request.offset_shape
            && (request.scale_shape.as_slice() == row.as_slice()
                || request.scale_shape.as_slice() == column.as_slice()),
        "normalization Scale/Offset shape mismatch"
    );
    Ok((groups, group_size))
}

pub(super) fn params(
    request: &DeepLearningLayerNormRequest,
    precision: crate::backend::wgpu::types::NumericPrecision,
) -> Result<NormParams> {
    let (groups, group_size) = geometry(request)?;
    let total = shape_count(&request.shape)?;
    let mut shape = [1u32; 4];
    for (target, &extent) in shape.iter_mut().zip(&request.shape) {
        *target = u32::try_from(extent)?;
    }
    let epsilon_bits = match precision {
        crate::backend::wgpu::types::NumericPrecision::F32 => {
            (request.epsilon as f32).to_bits() as u64
        }
        crate::backend::wgpu::types::NumericPrecision::F64 => request.epsilon.to_bits(),
    };
    Ok(NormParams {
        shape,
        rank: u32::try_from(request.shape.len())?,
        reduce_mask: u32::from(request.reduce_mask),
        groups: u32::try_from(groups)?,
        group_size: u32::try_from(group_size)?,
        channel_axis: u32::try_from(request.channel_axis)?,
        total: u32::try_from(total)?,
        chunk_offset: 0,
        chunk_count: 0,
        epsilon_low: epsilon_bits as u32,
        epsilon_high: (epsilon_bits >> 32) as u32,
    })
}
