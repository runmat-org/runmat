//! Device-limit admission for persisted layouts whose binding count is tagged.

use super::{storage_read_entry, storage_read_write_entry, uniform_entry};

pub(super) fn admit_counts(limits: &wgpu::Limits, storage: usize, uniforms: usize) -> Option<()> {
    let total = storage.checked_add(uniforms)?;
    (storage <= limits.max_storage_buffers_per_shader_stage as usize
        && uniforms <= limits.max_uniform_buffers_per_shader_stage as usize
        && total <= limits.max_bindings_per_bind_group as usize
        && u32::try_from(total).is_ok())
    .then_some(())
}

pub(super) fn admitted_count(
    emission: &runmat_accelerate_api::WgpuEmission<'_>,
    text: &str,
    other_storage: usize,
) -> Option<usize> {
    let count = text.parse::<usize>().ok()?;
    admit_counts(
        &emission.device().limits(),
        count.checked_add(other_storage)?,
        1,
    )?;
    Some(count)
}

pub(super) fn build_reduction_bgl(
    emission: &runmat_accelerate_api::WgpuEmission<'_>,
    inputs: usize,
) -> wgpu::BindGroupLayout {
    let mut entries = Vec::with_capacity(inputs + 2);
    for binding in 0..inputs {
        entries.push(storage_read_entry(binding as u32));
    }
    entries.push(storage_read_write_entry(inputs as u32));
    entries.push(uniform_entry((inputs + 1) as u32));
    emission
        .device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("runmat-reduction-dyn-bgl"),
            entries: &entries,
        })
}

#[cfg(test)]
mod tests {
    use super::admit_counts;

    #[test]
    fn storage_and_uniform_counts_obey_each_enabled_limit() {
        let mut limits = wgpu::Limits {
            max_bindings_per_bind_group: 4,
            max_storage_buffers_per_shader_stage: 2,
            max_uniform_buffers_per_shader_stage: 1,
            ..wgpu::Limits::default()
        };
        assert!(admit_counts(&limits, 2, 1).is_some());
        assert!(admit_counts(&limits, 3, 1).is_none());
        assert!(admit_counts(&limits, 2, 2).is_none());
        limits.max_storage_buffers_per_shader_stage = 4;
        assert!(admit_counts(&limits, 4, 1).is_none());
    }
}
