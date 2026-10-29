use super::*;
use runmat_runtime::{builtins::common::spec::checked_reduction_elements, value_fact::value_fact};
use runmat_value::Value;

pub(super) fn inputs(
    program: &ResolvedReductionProgram<'_>,
    config: ReductionRecipeConfig,
) -> Result<Vec<ReductionInputLayout>> {
    ensure!(
        program.data_inputs().len() == program.data_classes().len(),
        "reduction input class mapping mismatch"
    );
    let mut result = runmat_value::exact_capacity_vec(program.data_inputs().len())?;
    for (input, &class) in program.data_inputs().iter().zip(program.data_classes()) {
        let fact = value_fact(input.value);
        let source_precision = precision(
            fact.numeric()
                .ok_or_else(|| anyhow!("missing source numeric class"))?
                .class,
        )?;
        let bound_precision = precision(class)?;
        let shape = match &fact.shape {
            runmat_types::ShapeFact::Scalar => {
                runmat_value::exact_capacity_vec_from_slice(&[1usize, 1])?
            }
            runmat_types::ShapeFact::Shaped { dims } => {
                let mut shape = runmat_value::exact_capacity_vec(dims.len())?;
                for dimension in dims {
                    let runmat_types::DimensionFact::Known(extent) = dimension else {
                        return Err(anyhow!("unknown source dimensions"));
                    };
                    shape.push(*extent);
                }
                shape
            }
            _ => return Err(anyhow!("unknown source dimensions")),
        };
        let (source_layout, residency, source_physical) = match input.value {
            Value::GpuTensor(handle) => (
                handle.view().layout(),
                ProviderResidency::Device,
                Some(*handle.physical()),
            ),
            Value::Num(_) | Value::Tensor(_) => {
                (GpuTensorLayout::Contiguous, ProviderResidency::Host, None)
            }
            _ => return Err(anyhow!("unsupported recipe data operand")),
        };
        // Existing casts consume contiguous data. A cast of a lazy view requires
        // an explicit materialization; its shader strides are then contiguous.
        let transpose = match source_layout {
            GpuTensorLayout::Transpose(info) if source_precision == bound_precision => Some(info),
            _ => None,
        };
        let rank = program.contract().input_shape.len();
        let mut broadcast_shape = runmat_value::exact_capacity_vec_with(rank, |_| 0u32)?;
        let mut strides = runmat_value::exact_capacity_vec_with(rank, |_| 0u32)?;
        crate::backend::fusion_operand_layout::resolve_into(
            &shape,
            transpose,
            &program.contract().input_shape,
            config.max_rank,
            &mut broadcast_shape,
            &mut strides,
        )?;
        let elements = u64::try_from(
            checked_reduction_elements(&shape)
                .ok_or_else(|| anyhow!("source cardinality overflow"))?,
        )?;
        let source_payload_bytes = elements
            .checked_mul(bytes(source_precision))
            .ok_or_else(|| anyhow!("source byte size overflow"))?;
        let bound_payload_bytes = elements
            .checked_mul(bytes(bound_precision))
            .ok_or_else(|| anyhow!("bound byte size overflow"))?;
        validate_binding_extents(
            source_physical,
            bound_payload_bytes,
            config.max_storage_binding_bytes,
        )?;
        result.push(ReductionInputLayout {
            value_id: input.value_id,
            request_slot: input.request_slot,
            source_precision,
            bound_precision,
            source_layout,
            source_physical,
            residency,
            shape,
            broadcast_shape,
            strides,
            source_payload_bytes,
            bound_payload_bytes,
        });
    }
    Ok(result)
}

fn validate_binding_extents(
    source: Option<GpuPhysicalDescriptor>,
    bound_payload_bytes: u64,
    limit: u64,
) -> Result<()> {
    if let Some(source) = source {
        ensure!(
            source.requested_bytes() <= limit,
            "source allocation exceeds storage binding limit"
        );
    }
    // Host staging has no device source binding. The future executor must also
    // validate the actual destination buffer returned by allocation/upload.
    ensure!(
        bound_payload_bytes.max(4) <= limit,
        "input storage binding limit"
    );
    Ok(())
}

pub(super) fn geometry(
    contract: &FusionReductionContract,
    config: ReductionRecipeConfig,
) -> Result<ReductionGeometry> {
    let rank = contract.input_shape.len();
    ensure!(
        rank <= config.max_rank,
        "reduction rank exceeds recipe limit"
    );
    if !contract.identity {
        for (position, &axis) in contract.axes.iter().enumerate() {
            ensure!(
                axis < rank && !contract.axes[..position].contains(&axis),
                "invalid reduction axis receipt"
            );
        }
    }
    let count = |shape: &[usize]| -> Result<u32> {
        Ok(u32::try_from(
            checked_reduction_elements(shape)
                .ok_or_else(|| anyhow!("reduction cardinality overflow"))?,
        )?)
    };
    let selected_count = |reduced: bool| -> Result<u32> {
        let mut extents = contract
            .input_shape
            .iter()
            .enumerate()
            .filter(|(axis, _)| (!contract.identity && contract.axes.contains(axis)) == reduced)
            .map(|(_, &extent)| extent);
        // As with checked_reduction_elements, a zero extent wins before any
        // nonzero prefix can overflow. An empty selection has product one.
        if extents.clone().any(|extent| extent == 0) {
            return Ok(0);
        }
        let product = extents
            .try_fold(1usize, |n, extent| n.checked_mul(extent))
            .ok_or_else(|| anyhow!("reduction cardinality overflow"))?;
        Ok(u32::try_from(product)?)
    };
    let reduce_elements = selected_count(true)?;
    let slices = selected_count(false)?;
    ensure!(
        count(&contract.output_shape)? == slices,
        "owner output shape disagrees with reduction axes"
    );
    ensure!(
        u64::from(reduce_elements) * u64::from(slices) == u64::from(count(&contract.input_shape)?),
        "reduction input geometry mismatch"
    );
    let chunks = reduce_elements.div_ceil(config.chunk_elements).max(1);
    let partial_elements = slices
        .checked_mul(chunks)
        .ok_or_else(|| anyhow!("partial cardinality overflow"))?;
    Ok(ReductionGeometry {
        reduce_elements,
        slices,
        chunks,
        partial_elements,
        single: dispatch(slices, config.max_dispatch_dimension)?,
        // Prefer the enabled 3D grid for an executable TwoPass. If only that
        // alternate stage exceeds the device grid, carry exact representable
        // geometry for source validation and let selection reject TwoPass.
        pass1: dispatch(partial_elements, config.max_dispatch_dimension)
            .or_else(|_| dispatch(partial_elements, u32::MAX))?,
        pass2: dispatch(slices, config.max_dispatch_dimension)?,
    })
}

fn dispatch(groups: u32, limit: u32) -> Result<ReductionDispatch> {
    ensure!(limit != 0, "zero dispatch dimension limit");
    if groups == 0 {
        return Ok(ReductionDispatch {
            workgroups: [0, 0, 0],
            logical_groups: 0,
        });
    }
    let x = groups.min(limit);
    let rows = groups.div_ceil(x);
    let y = rows.min(limit);
    let z = rows.div_ceil(y);
    ensure!(
        z <= limit && u64::from(y) * u64::from(z) <= u64::from(u32::MAX),
        "reduction dispatch grid exceeds limits"
    );
    Ok(ReductionDispatch {
        workgroups: [x, y, z],
        logical_groups: groups,
    })
}

/// One coordinate-to-buffer mapping is used verbatim by single pass and pass 1.
pub(super) fn write_source(
    code: &mut dyn std::fmt::Write,
    contract: &FusionReductionContract,
    inputs: &[ReductionInputBinding],
) -> Result<()> {
    for (slot, input) in inputs.iter().enumerate() {
        write!(
            code,
            "fn rr_index{slot}(slice: u32, r: u32) -> u32 {{ return "
        )?;
        let mut emitted = false;
        let (mut reduce_stride, mut slice_stride) = (1u64, 1u64);
        if !contract.input_shape.contains(&0) {
            for (axis, (&extent, &stride)) in
                contract.input_shape.iter().zip(&input.strides).enumerate()
            {
                let reduced = !contract.identity && contract.axes.contains(&axis);
                let (coordinate, divisor) = if reduced {
                    ("r", &mut reduce_stride)
                } else {
                    ("slice", &mut slice_stride)
                };
                if stride != 0 {
                    if emitted {
                        code.write_str(" + ")?;
                    }
                    write!(
                        code,
                        "(({coordinate} / {divisor}u) % {extent}u) * {stride}u"
                    )?;
                    emitted = true;
                }
                *divisor = divisor
                    .checked_mul(extent as u64)
                    .ok_or_else(|| anyhow!("coordinate stride overflow"))?;
            }
        }
        if !emitted {
            code.write_str("0u")?;
        }
        code.write_str("; }\n")?;
    }
    Ok(())
}

#[cfg(test)]
mod binding_extent_tests {
    use super::*;
    use runmat_accelerate_api::{
        test_support::TestAllocationOwner, GpuTensorStorage, NumericElementType,
    };

    #[test]
    fn native_f64_cast_source_must_fit_even_when_f32_destination_fits() {
        let owner = TestAllocationOwner::new();
        let input = owner
            .allocate_zeroed(vec![16, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let physical = *input.handle.physical();
        assert_eq!(physical.requested_bytes(), 128);
        assert!(validate_binding_extents(Some(physical), 16 * 4, 64).is_err());
        assert!(validate_binding_extents(Some(physical), 16 * 4, 128).is_ok());
        assert!(validate_binding_extents(None, 16 * 4, 64).is_ok());
        drop(input.handle);
        assert_eq!(input.probe.retirements(), 1);
    }
}
