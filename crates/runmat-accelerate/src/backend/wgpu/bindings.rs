use wgpu::{BindGroupLayoutEntry, BindingType, BufferBindingType, ShaderStages};

mod dynamic_layout;
mod validated_layout;

/// Serialized cache tag for the typed modulation binding recipe.
pub(crate) const MODULATION_LAYOUT_TAG: &str = "runmat-modulate-constellation-layout";
pub(crate) const SOFTMAX_ROWS_LAYOUT_TAG: &str = "runmat-softmax-rows-layout";
pub(crate) const TRIANGULAR_SOLVE_LAYOUT_TAG: &str = "runmat-triangular-solve-layout";
pub(crate) const MATMUL_EPILOGUE_LAYOUT_TAG: &str = "runmat-matmul-epilogue-layout";
pub(crate) const INTEGER_COMPARE_LAYOUT_TAG: &str = "runmat-integer-compare-bgl";
pub(crate) const ROUND_DECIMALS_LAYOUT_TAG: &str = "runmat-round-decimals";
pub(crate) const ROUND_SIGNIFICANT_LAYOUT_TAG: &str = "runmat-round-significant";

pub(crate) fn build_matmul_epilogue_bgl(
    emission: &runmat_accelerate_api::WgpuEmission<'_>,
) -> wgpu::BindGroupLayout {
    let entries = [
        storage_read_entry(0),
        storage_read_entry(1),
        storage_read_write_entry(2),
        uniform_entry(3),
        storage_read_entry(4),
        storage_read_entry(5),
        uniform_entry(6),
    ];
    emission
        .device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(MATMUL_EPILOGUE_LAYOUT_TAG),
            entries: &entries,
        })
}

pub(crate) fn build_integer_compare_bgl(
    emission: &runmat_accelerate_api::WgpuEmission<'_>,
) -> wgpu::BindGroupLayout {
    let entries = [
        storage_read_entry(0),
        storage_read_entry(1),
        storage_read_write_entry(2),
        uniform_entry(3),
    ];
    emission
        .device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(INTEGER_COMPARE_LAYOUT_TAG),
            entries: &entries,
        })
}

pub(crate) fn build_round_digits_bgl(
    emission: &runmat_accelerate_api::WgpuEmission<'_>,
) -> wgpu::BindGroupLayout {
    let entries = [
        storage_read_entry(0),
        storage_read_write_entry(1),
        uniform_entry(2),
    ];
    emission
        .device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("runmat-round-digits-bgl"),
            entries: &entries,
        })
}

pub(crate) fn build_triangular_solve_bgl(
    emission: &runmat_accelerate_api::WgpuEmission<'_>,
) -> wgpu::BindGroupLayout {
    let entries = [
        storage_read_entry(0),
        storage_read_entry(1),
        storage_read_entry(2),
        storage_read_write_entry(3),
        uniform_entry(4),
    ];
    emission
        .device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(TRIANGULAR_SOLVE_LAYOUT_TAG),
            entries: &entries,
        })
}

pub(crate) fn build_softmax_rows_bgl(
    emission: &runmat_accelerate_api::WgpuEmission<'_>,
) -> wgpu::BindGroupLayout {
    let entries = [
        storage_read_entry(0),
        storage_read_write_entry(1),
        uniform_entry(2),
    ];
    emission
        .device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(SOFTMAX_ROWS_LAYOUT_TAG),
            entries: &entries,
        })
}

pub(crate) fn build_modulation_bgl(
    emission: &runmat_accelerate_api::WgpuEmission<'_>,
) -> wgpu::BindGroupLayout {
    let entries = [
        storage_read_entry(0),
        storage_read_entry(1),
        storage_read_write_entry(2),
        storage_read_write_entry(3),
        uniform_entry(4),
    ];
    emission
        .device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("warmup-modulate-constellation-bgl"),
            entries: &entries,
        })
}

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

/// Scalar and vector views share the production image-normalization layout.
/// Bootstrap and tuned pipelines must admit all five declared shader bindings.
pub(crate) fn image_normalize_entries() -> Vec<BindGroupLayoutEntry> {
    vec![
        storage_read_entry(0),
        storage_read_write_entry(1),
        uniform_entry(2),
        storage_read_entry(3),
        storage_read_write_entry(4),
    ]
}

pub fn build_bgl_for_layout_tag(
    emission: &runmat_accelerate_api::WgpuEmission<'_>,
    tag: &str,
) -> Option<wgpu::BindGroupLayout> {
    if let Some(layout) = validated_layout::reconstruct(emission, tag) {
        return Some(layout);
    }
    if let Some(rest) = tag.strip_prefix("runmat-fusion-layout-") {
        let inputs = dynamic_layout::admitted_count(emission, rest, 1)?;
        return Some(build_fusion_bgl(emission, inputs));
    }
    if let Some(rest) = tag.strip_prefix("runmat-reduction-layout-") {
        let inputs = dynamic_layout::admitted_count(emission, rest, 1)?;
        return Some(dynamic_layout::build_reduction_bgl(emission, inputs));
    }
    match tag {
        "runmat-eye-layout" => {
            let entries = [storage_read_write_entry(0), uniform_entry(1)];
            return Some(emission.device().create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-eye-bgl"),
                    entries: &entries,
                },
            ));
        }
        "runmat-centered-gram-layout" => {
            let entries = [
                storage_read_entry(0),
                storage_read_entry(1),
                storage_read_write_entry(2),
                uniform_entry(3),
            ];
            return Some(emission.device().create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-centered-gram-bgl"),
                    entries: &entries,
                },
            ));
        }
        MODULATION_LAYOUT_TAG => return Some(build_modulation_bgl(emission)),
        MATMUL_EPILOGUE_LAYOUT_TAG => return Some(build_matmul_epilogue_bgl(emission)),
        INTEGER_COMPARE_LAYOUT_TAG => return Some(build_integer_compare_bgl(emission)),
        ROUND_DECIMALS_LAYOUT_TAG | ROUND_SIGNIFICANT_LAYOUT_TAG => {
            return Some(build_round_digits_bgl(emission));
        }
        "runmat-fill-layout" => {
            let entries = [storage_read_write_entry(0), uniform_entry(1)];
            return Some(emission.device().create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-fill-bgl"),
                    entries: &entries,
                },
            ));
        }
        "runmat-linspace-layout" => {
            let entries = [storage_read_write_entry(0), uniform_entry(1)];
            return Some(emission.device().create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-linspace-bgl"),
                    entries: &entries,
                },
            ));
        }
        "runmat-random-int-layout" => {
            let entries = [storage_read_write_entry(0), uniform_entry(1)];
            return Some(emission.device().create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-random-int-bgl"),
                    entries: &entries,
                },
            ));
        }
        "runmat-random-uniform-layout" => {
            let entries = [storage_read_write_entry(0), uniform_entry(1)];
            return Some(emission.device().create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-random-uniform-bgl"),
                    entries: &entries,
                },
            ));
        }
        "runmat-random-normal-layout" => {
            let entries = [storage_read_write_entry(0), uniform_entry(1)];
            return Some(emission.device().create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-random-normal-bgl"),
                    entries: &entries,
                },
            ));
        }
        "runmat-randperm-layout" => {
            let entries = [storage_read_write_entry(0), uniform_entry(1)];
            return Some(emission.device().create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-randperm-bgl"),
                    entries: &entries,
                },
            ));
        }
        "runmat-fspecial-layout" => {
            let entries = [storage_read_write_entry(0), uniform_entry(1)];
            return Some(emission.device().create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-fspecial-bgl"),
                    entries: &entries,
                },
            ));
        }
        "runmat-polyval-layout" => {
            let entries = [
                storage_read_entry(0),
                storage_read_entry(1),
                storage_read_write_entry(2),
                uniform_entry(3),
            ];
            return Some(emission.device().create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-polyval-bgl"),
                    entries: &entries,
                },
            ));
        }
        "runmat-polyder-layout" => {
            let entries = [
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ];
            return Some(emission.device().create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-polyder-bgl"),
                    entries: &entries,
                },
            ));
        }
        "runmat-polyint-layout" => {
            let entries = [
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ];
            return Some(emission.device().create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-polyint-bgl"),
                    entries: &entries,
                },
            ));
        }
        "runmat-reduce-dim-minmax-layout" => {
            let entries = [
                storage_read_entry(0),
                storage_read_write_entry(1),
                storage_read_write_entry(2),
                uniform_entry(3),
            ];
            return Some(emission.device().create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-reduce-dim-minmax-bgl"),
                    entries: &entries,
                },
            ));
        }
        "runmat-reduce-nd-moments-layout" => {
            let entries = [
                storage_read_entry(0),
                storage_read_write_entry(1),
                storage_read_write_entry(2),
                uniform_entry(3),
            ];
            return Some(emission.device().create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-reduce-nd-moments-bgl"),
                    entries: &entries,
                },
            ));
        }
        "runmat-find-layout" => {
            let entries = [
                storage_read_entry(0),
                storage_read_write_entry(1),
                storage_read_write_entry(2),
                storage_read_write_entry(3),
                storage_read_write_entry(4),
                storage_read_write_entry(5),
                uniform_entry(6),
            ];
            return Some(emission.device().create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-find-bgl"),
                    entries: &entries,
                },
            ));
        }
        "runmat-bandwidth-layout" => {
            let entries = [
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ];
            return Some(emission.device().create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-bandwidth-bgl"),
                    entries: &entries,
                },
            ));
        }
        "runmat-symmetry-layout" => {
            let entries = [
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ];
            return Some(emission.device().create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-symmetry-bgl"),
                    entries: &entries,
                },
            ));
        }
        "runmat-reduction-bgl" | "runmat-reduction-p1-bgl" | "runmat-reduction-p2-bgl" => {
            let entries = [
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ];
            return Some(emission.device().create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-reduction-bgl"),
                    entries: &entries,
                },
            ));
        }
        "runmat-repmat-layout" => {
            let entries = [
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ];
            return Some(emission.device().create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-repmat-bgl"),
                    entries: &entries,
                },
            ));
        }
        "runmat-flip-layout" => {
            let entries = [
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ];
            return Some(emission.device().create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-flip-bgl"),
                    entries: &entries,
                },
            ));
        }
        "runmat-cumsum-layout" => {
            let entries = [
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ];
            return Some(emission.device().create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-cumsum-bgl"),
                    entries: &entries,
                },
            ));
        }
        "runmat-trapezoid-layout" => {
            let entries = [
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
                storage_read_entry(3),
            ];
            return Some(emission.device().create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-trapezoid-bgl"),
                    entries: &entries,
                },
            ));
        }
        "runmat-cumprod-layout" => {
            let entries = [
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ];
            return Some(emission.device().create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-cumprod-bgl"),
                    entries: &entries,
                },
            ));
        }
        "runmat-diff-layout" => {
            let entries = [
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ];
            return Some(emission.device().create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-diff-bgl"),
                    entries: &entries,
                },
            ));
        }
        "runmat-cummin-layout" => {
            let entries = [
                storage_read_entry(0),
                storage_read_write_entry(1),
                storage_read_write_entry(2),
                uniform_entry(3),
            ];
            return Some(emission.device().create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-cummin-bgl"),
                    entries: &entries,
                },
            ));
        }
        "runmat-cummax-layout" => {
            let entries = [
                storage_read_entry(0),
                storage_read_write_entry(1),
                storage_read_write_entry(2),
                uniform_entry(3),
            ];
            return Some(emission.device().create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-cummax-bgl"),
                    entries: &entries,
                },
            ));
        }
        "runmat-tril-layout" => {
            let entries = [
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ];
            return Some(emission.device().create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-tril-bgl"),
                    entries: &entries,
                },
            ));
        }
        "runmat-triu-layout" => {
            let entries = [
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ];
            return Some(emission.device().create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-triu-bgl"),
                    entries: &entries,
                },
            ));
        }
        "runmat-imfilter-layout" => {
            let entries = [
                storage_read_entry(0),
                storage_read_entry(1),
                storage_read_entry(2),
                storage_read_write_entry(3),
                uniform_entry(4),
            ];
            return Some(emission.device().create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-imfilter-bgl"),
                    entries: &entries,
                },
            ));
        }
        "runmat-conv1d-layout" => {
            let entries = [
                storage_read_entry(0),
                storage_read_entry(1),
                storage_read_write_entry(2),
                uniform_entry(3),
            ];
            return Some(emission.device().create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-conv1d-bgl"),
                    entries: &entries,
                },
            ));
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
            return Some(emission.device().create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-iir-filter-bgl"),
                    entries: &entries,
                },
            ));
        }
        _ => {}
    }
    None
}

pub fn build_fusion_bgl(
    emission: &runmat_accelerate_api::WgpuEmission<'_>,
    n_inputs: usize,
) -> wgpu::BindGroupLayout {
    let mut entries = Vec::with_capacity(n_inputs + 2);
    for i in 0..n_inputs {
        entries.push(storage_read_entry(i as u32));
    }
    entries.push(storage_read_write_entry(n_inputs as u32));
    entries.push(uniform_entry((n_inputs + 1) as u32));
    emission
        .device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("runmat-fusion-bgl"),
            entries: &entries,
        })
}

/// Multi-output variant: N read-only inputs + `n_outputs` read-write outputs + 1 uniform params.
pub fn build_fusion_multi_bgl(
    emission: &runmat_accelerate_api::WgpuEmission<'_>,
    n_inputs: usize,
    n_outputs: usize,
) -> wgpu::BindGroupLayout {
    let mut entries = Vec::with_capacity(n_inputs + n_outputs + 1);
    for i in 0..n_inputs {
        entries.push(storage_read_entry(i as u32));
    }
    for k in 0..n_outputs {
        entries.push(storage_read_write_entry((n_inputs + k) as u32));
    }
    entries.push(uniform_entry((n_inputs + n_outputs) as u32));
    emission
        .device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("runmat-fusion-multi-bgl"),
            entries: &entries,
        })
}

pub fn build_scatter_col_bgl(
    emission: &runmat_accelerate_api::WgpuEmission<'_>,
) -> wgpu::BindGroupLayout {
    emission
        .device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("runmat-scatter-col-bgl"),
            entries: &[
                storage_read_entry(0),
                storage_read_entry(1),
                storage_read_write_entry(2),
                uniform_entry(3),
            ],
        })
}

pub fn build_scatter_row_bgl(
    emission: &runmat_accelerate_api::WgpuEmission<'_>,
) -> wgpu::BindGroupLayout {
    emission
        .device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("runmat-scatter-row-bgl"),
            entries: &[
                storage_read_entry(0),
                storage_read_entry(1),
                storage_read_write_entry(2),
                uniform_entry(3),
            ],
        })
}

pub fn build_sub2ind_bgl(
    emission: &runmat_accelerate_api::WgpuEmission<'_>,
    n_inputs: usize,
) -> wgpu::BindGroupLayout {
    let mut entries = Vec::with_capacity(n_inputs + 3);
    for i in 0..n_inputs {
        entries.push(storage_read_entry(i as u32));
    }
    entries.push(storage_read_write_entry(n_inputs as u32));
    entries.push(storage_read_write_entry((n_inputs + 1) as u32));
    entries.push(uniform_entry((n_inputs + 2) as u32));
    emission
        .device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("runmat-sub2ind-bgl"),
            entries: &entries,
        })
}

pub fn build_ind2sub_bgl(
    emission: &runmat_accelerate_api::WgpuEmission<'_>,
    n_outputs: usize,
) -> wgpu::BindGroupLayout {
    let mut entries = Vec::with_capacity(n_outputs + 3);
    entries.push(storage_read_entry(0));
    for i in 0..n_outputs {
        entries.push(storage_read_write_entry((i + 1) as u32));
    }
    entries.push(storage_read_write_entry((n_outputs + 1) as u32));
    entries.push(uniform_entry((n_outputs + 2) as u32));
    emission
        .device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("runmat-ind2sub-bgl"),
            entries: &entries,
        })
}
