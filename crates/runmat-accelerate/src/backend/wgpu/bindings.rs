use wgpu::{BindGroupLayoutEntry, BindingType, BufferBindingType, ShaderStages};

pub fn storage_read_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub fn storage_read_write_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub fn uniform_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub fn build_bgl_for_layout_tag(device: &wgpu::Device, tag: &str) -> Option<wgpu::BindGroupLayout> {
    if let Some(rest) = tag.strip_prefix("runmat-fusion-layout-") {
        if let Ok(n_inputs) = rest.parse::<usize>() {
            let mut entries = Vec::with_capacity(n_inputs + 2);
            for i in 0..n_inputs {
                entries.push(storage_read_entry(i as u32));
            }
            entries.push(storage_read_write_entry(n_inputs as u32));
            entries.push(uniform_entry((n_inputs + 1) as u32));
            return Some(
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-fusion-bgl"),
                    entries: &entries,
                }),
            );
        }
    }
    if let Some(rest) = tag.strip_prefix("runmat-reduction-layout-") {
        if let Ok(n_inputs) = rest.parse::<usize>() {
            let mut entries = Vec::with_capacity(n_inputs + 2);
            for i in 0..n_inputs {
                entries.push(storage_read_entry(i as u32));
            }
            entries.push(storage_read_write_entry(n_inputs as u32));
            entries.push(uniform_entry((n_inputs + 1) as u32));
            return Some(
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("runmat-reduction-dyn-bgl"),
                    entries: &entries,
                }),
            );
        }
    }
    match tag {
        "runmat-reduction-bgl" | "runmat-reduction-p1-bgl" | "runmat-reduction-p2-bgl" => {
            let entries = [
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ];
            return Some(
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-reduction-bgl"),
                    entries: &entries,
                }),
            );
        }
        "runmat-repmat-layout" => {
            let entries = [
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ];
            return Some(
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-repmat-bgl"),
                    entries: &entries,
                }),
            );
        }
        "runmat-flip-layout" => {
            let entries = [
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ];
            return Some(
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-flip-bgl"),
                    entries: &entries,
                }),
            );
        }
        "runmat-cumsum-layout" => {
            let entries = [
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ];
            return Some(
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-cumsum-bgl"),
                    entries: &entries,
                }),
            );
        }
        "runmat-cumprod-layout" => {
            let entries = [
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ];
            return Some(
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-cumprod-bgl"),
                    entries: &entries,
                }),
            );
        }
        "runmat-diff-layout" => {
            let entries = [
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ];
            return Some(
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-diff-bgl"),
                    entries: &entries,
                }),
            );
        }
        "runmat-cummin-layout" => {
            let entries = [
                storage_read_entry(0),
                storage_read_write_entry(1),
                storage_read_write_entry(2),
                uniform_entry(3),
            ];
            return Some(
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-cummin-bgl"),
                    entries: &entries,
                }),
            );
        }
        "runmat-cummax-layout" => {
            let entries = [
                storage_read_entry(0),
                storage_read_write_entry(1),
                storage_read_write_entry(2),
                uniform_entry(3),
            ];
            return Some(
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-cummax-bgl"),
                    entries: &entries,
                }),
            );
        }
        "runmat-tril-layout" => {
            let entries = [
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ];
            return Some(
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-tril-bgl"),
                    entries: &entries,
                }),
            );
        }
        "runmat-triu-layout" => {
            let entries = [
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ];
            return Some(
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-triu-bgl"),
                    entries: &entries,
                }),
            );
        }
        "runmat-imfilter-layout" => {
            let entries = [
                storage_read_entry(0),
                storage_read_entry(1),
                storage_read_entry(2),
                storage_read_write_entry(3),
                uniform_entry(4),
            ];
            return Some(
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-imfilter-bgl"),
                    entries: &entries,
                }),
            );
        }
        "runmat-conv1d-layout" => {
            let entries = [
                storage_read_entry(0),
                storage_read_entry(1),
                storage_read_write_entry(2),
                uniform_entry(3),
            ];
            return Some(
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-conv1d-bgl"),
                    entries: &entries,
                }),
            );
        }
        "runmat-iir-filter-layout" => {
            let entries = [
                storage_read_entry(0),
                storage_read_entry(1),
                storage_read_entry(2),
                storage_read_entry(3),
                storage_read_write_entry(4),
                storage_read_write_entry(5),
                storage_read_write_entry(6),
                uniform_entry(7),
            ];
            return Some(
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-iir-filter-bgl"),
                    entries: &entries,
                }),
            );
        }
        _ => {}
    }
    None
}

pub fn build_fusion_bgl(device: &wgpu::Device, n_inputs: usize) -> wgpu::BindGroupLayout {
    let mut entries = Vec::with_capacity(n_inputs + 2);
    for i in 0..n_inputs {
        entries.push(storage_read_entry(i as u32));
    }
    entries.push(storage_read_write_entry(n_inputs as u32));
    entries.push(uniform_entry((n_inputs + 1) as u32));
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("runmat-fusion-bgl"),
        entries: &entries,
    })
}

pub fn build_scatter_col_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("runmat-scatter-col-bgl"),
        entries: &[
            storage_read_entry(0),
            storage_read_entry(1),
            storage_read_write_entry(2),
            uniform_entry(3),
        ],
    })
}

pub fn build_scatter_row_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("runmat-scatter-row-bgl"),
        entries: &[
            storage_read_entry(0),
            storage_read_entry(1),
            storage_read_write_entry(2),
            uniform_entry(3),
        ],
    })
}

pub fn build_sub2ind_bgl(device: &wgpu::Device, n_inputs: usize) -> wgpu::BindGroupLayout {
    let mut entries = Vec::with_capacity(n_inputs + 3);
    for i in 0..n_inputs {
        entries.push(storage_read_entry(i as u32));
    }
    entries.push(storage_read_write_entry(n_inputs as u32));
    entries.push(storage_read_write_entry((n_inputs + 1) as u32));
    entries.push(uniform_entry((n_inputs + 2) as u32));
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("runmat-sub2ind-bgl"),
        entries: &entries,
    })
}

pub fn build_ind2sub_bgl(device: &wgpu::Device, n_outputs: usize) -> wgpu::BindGroupLayout {
    let mut entries = Vec::with_capacity(n_outputs + 3);
    entries.push(storage_read_entry(0));
    for i in 0..n_outputs {
        entries.push(storage_read_write_entry((i + 1) as u32));
    }
    entries.push(storage_read_write_entry((n_outputs + 1) as u32));
    entries.push(uniform_entry((n_outputs + 2) as u32));
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("runmat-ind2sub-bgl"),
        entries: &entries,
    })
}
