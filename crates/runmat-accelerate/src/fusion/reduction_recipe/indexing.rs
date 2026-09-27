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
    program
        .data_inputs()
        .iter()
        .zip(program.data_classes())
        .map(|(input, &class)| {
            let fact = value_fact(input.value);
            let source_precision = precision(
                fact.numeric()
                    .ok_or_else(|| anyhow!("missing source numeric class"))?
                    .class,
            )?;
            let bound_precision = precision(class)?;
            let shape = fact
                .shape
                .known_dims()
                .and_then(|dims| dims.into_iter().collect::<Option<Vec<_>>>())
                .ok_or_else(|| anyhow!("unknown source dimensions"))?;
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
                GpuTensorLayout::Transpose(info) if source_precision == bound_precision => {
                    Some(info)
                }
                _ => None,
            };
            let (broadcast_shape, strides) = crate::backend::fusion_operand_layout::resolve(
                &shape,
                transpose,
                &program.contract().input_shape,
                config.max_rank,
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
            Ok(ReductionInputLayout {
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
            })
        })
        .collect()
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

pub(super) fn geometry(
    contract: &FusionReductionContract,
    config: ReductionRecipeConfig,
) -> Result<ReductionGeometry> {
    let rank = contract.input_shape.len();
    ensure!(
        rank <= config.max_rank,
        "reduction rank exceeds recipe limit"
    );
    let mut reduced = vec![false; rank];
    if !contract.identity {
        for &axis in &contract.axes {
            ensure!(
                axis < rank && !reduced[axis],
                "invalid reduction axis receipt"
            );
            reduced[axis] = true;
        }
    }
    let reduce_shape: Vec<_> = contract
        .input_shape
        .iter()
        .enumerate()
        .filter_map(|(axis, &extent)| reduced[axis].then_some(extent))
        .collect();
    let slice_shape: Vec<_> = contract
        .input_shape
        .iter()
        .enumerate()
        .filter_map(|(axis, &extent)| (!reduced[axis]).then_some(extent))
        .collect();
    let count = |shape: &[usize]| -> Result<u32> {
        Ok(u32::try_from(
            checked_reduction_elements(shape)
                .ok_or_else(|| anyhow!("reduction cardinality overflow"))?,
        )?)
    };
    let reduce_elements = count(&reduce_shape)?;
    let slices = count(&slice_shape)?;
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
        pass1: dispatch(partial_elements, config.max_dispatch_dimension)?,
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
pub(super) fn source(
    program: &ResolvedReductionProgram<'_>,
    inputs: &[ReductionInputLayout],
) -> Result<String> {
    let contract = program.contract();
    let mut code = String::new();
    for (slot, input) in inputs.iter().enumerate() {
        let mut terms = Vec::new();
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
                    terms.push(format!(
                        "(({coordinate} / {divisor}u) % {extent}u) * {stride}u"
                    ));
                }
                *divisor = divisor
                    .checked_mul(extent as u64)
                    .ok_or_else(|| anyhow!("coordinate stride overflow"))?;
            }
        }
        let expression = if terms.is_empty() {
            "0u".into()
        } else {
            terms.join(" + ")
        };
        code.push_str(&format!(
            "fn rr_index{slot}(slice: u32, r: u32) -> u32 {{ return {expression}; }}\n"
        ));
    }
    Ok(code)
}
