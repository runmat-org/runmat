//! Reconstructible layouts for disk receipts of migrated validated routes.
//! A layout tag identifies bindings only; it never identifies a runtime kind.

use super::*;

fn build(
    emission: &runmat_accelerate_api::WgpuEmission<'_>,
    tag: &str,
    entries: &[BindGroupLayoutEntry],
) -> wgpu::BindGroupLayout {
    emission
        .device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(tag),
            entries,
        })
}

pub(super) fn reconstruct(
    emission: &runmat_accelerate_api::WgpuEmission<'_>,
    tag: &str,
) -> Option<wgpu::BindGroupLayout> {
    if let Some(rest) = tag.strip_prefix("runmat-fusion-multi-layout-") {
        let (inputs, outputs) = rest.split_once('-')?;
        let inputs = inputs.parse::<usize>().ok()?;
        let outputs = outputs.parse::<usize>().ok()?;
        super::dynamic_layout::admit_counts(
            &emission.device().limits(),
            inputs.checked_add(outputs)?,
            1,
        )?;
        return Some(build_fusion_multi_bgl(emission, inputs, outputs));
    }
    if let Some(rest) = tag.strip_prefix("runmat-sub2ind-layout-") {
        return Some(build_sub2ind_bgl(
            emission,
            super::dynamic_layout::admitted_count(emission, rest, 2)?,
        ));
    }
    if let Some(rest) = tag.strip_prefix("runmat-ind2sub-layout-") {
        return Some(build_ind2sub_bgl(
            emission,
            super::dynamic_layout::admitted_count(emission, rest, 2)?,
        ));
    }
    match tag {
        SOFTMAX_ROWS_LAYOUT_TAG => Some(build_softmax_rows_bgl(emission)),
        TRIANGULAR_SOLVE_LAYOUT_TAG => Some(build_triangular_solve_bgl(emission)),
        "runmat-scatter-col-bgl" => Some(build_scatter_col_bgl(emission)),
        "runmat-scatter-row-bgl" => Some(build_scatter_row_bgl(emission)),
        "runmat-integer-sum-floating-cast-bgl"
        | "runmat-nd-sum-f64-bgl"
        | "runmat-integer-cast-bgl"
        | "runmat-floating-cast-bgl"
        | "runmat-reduce-global-validated-bgl"
        | "runmat-std-dim-sum-validated-bgl" => Some(build(
            emission,
            tag,
            &[
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ],
        )),
        "runmat-std-binary-validated-bgl"
        | "runmat-integer-arithmetic-bgl"
        | "runmat-integer-minmax-bgl" => Some(build(
            emission,
            tag,
            &[
                storage_read_entry(0),
                storage_read_entry(1),
                storage_read_write_entry(2),
                uniform_entry(3),
            ],
        )),
        "runmat-reduce-dim-minmax-validated-bgl"
        | "runmat-integer-reduce-dim-extrema-validated-bgl" => Some(build(
            emission,
            tag,
            &[
                storage_read_entry(0),
                storage_read_write_entry(1),
                storage_read_write_entry(2),
                uniform_entry(3),
            ],
        )),
        "runmat-std-fill-validated-bgl" => Some(build(
            emission,
            tag,
            &[storage_read_write_entry(0), uniform_entry(1)],
        )),
        _ => None,
    }
}
