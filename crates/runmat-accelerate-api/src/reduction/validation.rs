//! Check physical consistency; never infer reduction semantics or regenerate a
//! producer's strides, shader source, or scheduling decision.
use super::{types::width, *};
use anyhow::{anyhow, ensure, Result};
use runmat_types::NumericClass;

pub(super) fn elements(shape: &[usize]) -> Result<u64> {
    if shape.contains(&0) {
        return Ok(0);
    }
    shape.iter().try_fold(1u64, |n, &extent| {
        n.checked_mul(u64::try_from(extent)?)
            .ok_or_else(|| anyhow!("reduction cardinality overflow"))
    })
}

pub(super) fn recipe(
    d: &ReductionRecipeDescriptor,
    sources: &ReductionShaderSources,
) -> Result<()> {
    let c = d.config;
    let g = &d.geometry;
    let p = d.partial;
    let w = &d.work;
    semantic(d)?;
    if d.producer_transport == crate::scalar_program::ScalarTransport::Binary32Words {
        ensure!(
            d.execution_precision == crate::ProviderPrecision::F32
                && d.inputs
                    .iter()
                    .all(|input| input.precision == crate::ProviderPrecision::F32),
            "binary32 word transport requires actual Single data ABI"
        );
    }
    ensure!(
        c.workgroup_size.is_power_of_two() && c.workgroup_size <= c.max_workgroup_size,
        "invalid reduction workgroup size"
    );
    ensure!(
        c.chunk_elements != 0 && c.max_dispatch_dimension != 0,
        "zero reduction geometry limit"
    );
    ensure!(
        d.iteration_shape.len() <= c.max_rank && d.output_shape.len() <= c.max_rank,
        "reduction rank limit"
    );
    ensure!(
        elements(&d.output_shape)? == u64::from(g.slices),
        "reduction output geometry mismatch"
    );
    ensure!(
        elements(&d.iteration_shape)? == u64::from(g.slices) * u64::from(g.reduce_elements),
        "reduction iteration geometry mismatch"
    );
    ensure!(
        g.chunks == g.reduce_elements.div_ceil(c.chunk_elements).max(1),
        "reduction chunk geometry mismatch"
    );
    ensure!(
        Some(g.partial_elements) == g.slices.checked_mul(g.chunks),
        "reduction partial geometry mismatch"
    );
    dispatch(g.single, g.slices, c.max_dispatch_dimension)?;
    // Validate exact representable alternate geometry, not its selection.
    dispatch(g.pass1, g.partial_elements, u32::MAX)?;
    dispatch(g.pass2, g.slices, c.max_dispatch_dimension)?;
    let scalar = width(p.precision) as u32;
    ensure!(
        p.alignment == scalar
            && p.stride == scalar + 8
            && p.count_offset == scalar
            && p.flags_offset == scalar + 4,
        "invalid reduction partial ABI"
    );
    ensure!(
        w.output_payload_bytes == u64::from(g.slices) * width(d.output_precision)
            && w.output_requested_bytes == w.output_payload_bytes.max(4),
        "invalid reduction output bytes"
    );
    ensure!(
        w.partial_payload_bytes == u64::from(g.partial_elements) * u64::from(p.stride)
            && w.partial_requested_bytes
                == if g.partial_elements == 0 {
                    0
                } else {
                    w.partial_payload_bytes.max(4)
                },
        "invalid reduction partial bytes"
    );
    ensure!(
        w.output_requested_bytes <= c.max_storage_binding_bytes,
        "reduction output storage binding limit"
    );
    ensure!(
        w.workgroup_bytes == 0
            || w.workgroup_bytes == u64::from(c.workgroup_size) * u64::from(p.stride),
        "invalid reduction workgroup bytes"
    );
    ensure!(
        w.workgroup_bytes <= c.max_workgroup_storage_bytes,
        "reduction workgroup storage limit"
    );
    ensure!(
        w.single_dispatches == u32::from(g.slices != 0)
            && w.two_pass_dispatches == if g.slices == 0 { 0 } else { 2 }
            && w.uniform_bytes == 0,
        "invalid reduction dispatch work"
    );
    for input in &d.inputs {
        binding(input, &d.iteration_shape, c)?;
    }
    ensure!(
        !sources.single.trim().is_empty()
            && !sources.pass1.trim().is_empty()
            && !sources.pass2.trim().is_empty(),
        "missing reduction shader source"
    );
    Ok(())
}

fn semantic(d: &ReductionRecipeDescriptor) -> Result<()> {
    let contract = &d.contract;
    ensure!(
        matches!(
            contract.input_class,
            NumericClass::Single | NumericClass::Double
        ),
        "reduction owner input must be real floating"
    );
    ensure!(
        contract.input_shape == d.iteration_shape && contract.output_shape == d.output_shape,
        "reduction owner geometry differs from physical geometry"
    );
    let output_class = match d.output_precision {
        crate::ProviderPrecision::F32 => NumericClass::Single,
        crate::ProviderPrecision::F64 => NumericClass::Double,
    };
    let partial_class = match d.partial.precision {
        crate::ProviderPrecision::F32 => NumericClass::Single,
        crate::ProviderPrecision::F64 => NumericClass::Double,
    };
    ensure!(
        contract.output_class == output_class
            && partial_class
                == if contract.identity {
                    contract.input_class
                } else {
                    contract.accumulator_class
                },
        "reduction owner classes differ from physical representation"
    );
    let (reduced_elements, slices) = contract
        .checked_work_geometry()
        .ok_or_else(|| anyhow!("invalid reduction owner geometry or arithmetic policy"))?;
    ensure!(
        reduced_elements == u64::from(d.geometry.reduce_elements)
            && slices == u64::from(d.geometry.slices),
        "reduction owner axes differ from physical dispatch"
    );
    Ok(())
}

fn dispatch(dispatch: ReductionDispatch, groups: u32, limit: u32) -> Result<()> {
    ensure!(
        dispatch.logical_groups == groups,
        "reduction logical dispatch mismatch"
    );
    if groups == 0 {
        ensure!(
            dispatch.workgroups == [0; 3],
            "empty reduction dispatch must be absent"
        );
        return Ok(());
    }
    ensure!(
        dispatch.workgroups.iter().all(|&n| n != 0 && n <= limit),
        "reduction dispatch dimension limit"
    );
    let [x, y, z] = dispatch.workgroups.map(u64::from);
    let rows = y * z;
    ensure!(
        rows <= u64::from(u32::MAX) && rows.checked_mul(x).is_some_and(|n| n >= u64::from(groups)),
        "invalid reduction dispatch grid"
    );
    Ok(())
}

fn binding(
    input: &ReductionInputBinding,
    iteration: &[usize],
    c: ReductionRecipeConfig,
) -> Result<()> {
    ensure!(
        input.broadcast_shape.len() == iteration.len() && input.strides.len() == iteration.len(),
        "reduction binding rank mismatch"
    );
    let count = elements(&input.shape)?;
    let broadcast_count = if input.broadcast_shape.contains(&0) {
        0
    } else {
        input.broadcast_shape.iter().try_fold(1u64, |n, &extent| {
            n.checked_mul(u64::from(extent))
                .ok_or_else(|| anyhow!("reduction broadcast cardinality overflow"))
        })?
    };
    ensure!(
        broadcast_count == count,
        "reduction binding shape cardinality mismatch"
    );
    ensure!(
        Some(input.payload_bytes) == count.checked_mul(width(input.precision)),
        "reduction binding byte mismatch"
    );
    ensure!(
        input.payload_bytes.max(4) <= c.max_storage_binding_bytes,
        "reduction input binding limit"
    );
    let reads = elements(iteration)? != 0;
    let mut last = 0u64;
    for ((&root, &extent), &stride) in iteration
        .iter()
        .zip(&input.broadcast_shape)
        .zip(&input.strides)
    {
        ensure!(
            u64::from(extent) == u64::try_from(root)? || extent == 1,
            "incompatible reduction broadcast extent"
        );
        ensure!(
            extent != 1 || stride == 0,
            "singleton reduction binding must not advance"
        );
        if reads {
            let term = u64::try_from(root - 1)?
                .checked_mul(u64::from(stride))
                .ok_or_else(|| anyhow!("reduction address overflow"))?;
            last = last
                .checked_add(term)
                .ok_or_else(|| anyhow!("reduction address overflow"))?;
        }
    }
    // Empty iteration has no read, even if other extents/strides are large.
    if reads {
        ensure!(
            last < count && last <= u64::from(u32::MAX),
            "reduction input address exceeds payload"
        );
    }
    Ok(())
}
