//! Recipe-bound checked source mapping and effect-free physical proposals.
use super::{invalid, preflight_sources, InputAction, PreflightedInputs, TypedInputPlan};
use crate::fusion::reduction_recipe::ReductionRecipe;
use runmat_accelerate_api::reduction::{
    ProviderReductionInputInventory, ProviderReductionInputPreparation,
    ProviderReductionInputStage, ReductionSelection,
};
use runmat_accelerate_api::{
    class_preserving_upload_query, GpuTensorLayout, ProviderElementType, ProviderExecutionError,
    ProviderExecutionResult, ProviderLayout, ProviderRealElementwiseOperation,
    ProviderRealElementwiseRecipe, ProviderRealElementwiseRequest, ProviderResidency,
    ProviderStorage, ProviderTypedReductionWorkload,
};
use runmat_value::Value;

/// Bind the query-free source proof to the exact checked selected recipe. A
/// fresh source/lease revalidation and selected binding-cap check happen before
/// any strategy-candidate, upload evidence, or cast proposal query.
pub(in crate::fusion_exec::reduction) fn plan<'a>(
    preflight: PreflightedInputs<'_, 'a>,
    recipe: &ReductionRecipe,
    selection: &ReductionSelection<'_>,
) -> ProviderExecutionResult<Option<TypedInputPlan<'a>>> {
    let provider = preflight.owner;
    let program = preflight.program;
    let refreshed = preflight_sources(provider, program)?;
    if refreshed.sources != preflight.sources {
        return Err(invalid(
            "typed source representation changed after preflight",
        ));
    }
    let sources = preflight.sources;
    if !std::ptr::eq(recipe.physical(), selection.recipe()) {
        return Err(invalid(
            "typed selection does not borrow the selected recipe",
        ));
    }
    let binding_cap = selection
        .recipe()
        .descriptor()
        .config
        .max_storage_binding_bytes;
    let selected = ProviderTypedReductionWorkload::from_selection(selection);
    if program.data_inputs().len() != recipe.inputs().len()
        || selected.inputs.len() != recipe.inputs().len()
        || sources.len() != recipe.inputs().len()
    {
        return Err(invalid("typed reduction input count changed"));
    }
    for (((input, layout), prepared), source) in program
        .data_inputs()
        .iter()
        .zip(recipe.inputs())
        .zip(&selected.inputs)
        .zip(&sources)
    {
        if input.value_id != layout.value_id || input.request_slot != layout.request_slot {
            return Err(invalid("typed reduction data leaf order changed"));
        }
        let source_layout = match layout.source_layout {
            GpuTensorLayout::Contiguous => ProviderLayout::ColumnMajorContiguous,
            GpuTensorLayout::Transpose(_) => ProviderLayout::Strided,
        };
        let expected_shape: Vec<_> = layout
            .shape
            .iter()
            .map(|&extent| {
                u64::try_from(extent).map_err(|_| invalid("typed source extent exceeds u64"))
            })
            .collect::<ProviderExecutionResult<_>>()?;
        if source.element_type != ProviderElementType::from(layout.source_precision)
            || source.layout != source_layout
            || source.shape != expected_shape
            || source.residency != layout.residency
            || source.storage != ProviderStorage::DenseReal
            || prepared.element_type != ProviderElementType::from(layout.bound_precision)
            || prepared.shape != source.shape
        {
            return Err(invalid(
                "typed reduction actual source differs from checked recipe",
            ));
        }
        match input.value {
            Value::GpuTensor(handle) => {
                if layout.source_physical != Some(*handle.physical())
                    || handle.physical().requested_bytes() > binding_cap
                {
                    return Err(invalid(
                        "typed resident source violates selected binding limit",
                    ));
                }
            }
            Value::Num(_) | Value::Tensor(_) => {
                if layout.source_physical.is_some() {
                    return Err(invalid(
                        "typed host source has unexpected device allocation",
                    ));
                }
            }
            _ => unreachable!("query-free preflight checked carrier"),
        }
    }

    let mut entries = Vec::with_capacity(sources.len());
    let mut actions = Vec::with_capacity(sources.len());
    let mut casts = Vec::new();
    for (((input, layout), source), prepared) in program
        .data_inputs()
        .iter()
        .zip(recipe.inputs())
        .zip(sources)
        .zip(&selected.inputs)
    {
        let stage = match input.value {
            Value::GpuTensor(handle) => {
                if layout.source_precision == layout.bound_precision {
                    if &source != prepared {
                        return Err(invalid("resident borrow differs from binding"));
                    }
                    // A checked same-class view may be borrowed directly; its
                    // exact view and physical descriptor were preflighted above.
                    actions.push(InputAction::Borrow(handle));
                    ProviderReductionInputStage::BorrowResident {
                        device_id: provider.device_id(),
                    }
                } else {
                    // Class-changing strided materialization needs its own
                    // executable checked owner; no guessed cast proposal.
                    if layout.source_layout != GpuTensorLayout::Contiguous {
                        return Ok(None);
                    }
                    let request = ProviderRealElementwiseRequest {
                        operation: ProviderRealElementwiseOperation::FloatingCast {
                            target: layout.bound_precision,
                        },
                        source: source.clone(),
                        source_buffer_bytes: handle.physical().requested_bytes(),
                    };
                    let Some(cast) = ProviderRealElementwiseRecipe::plan(provider, request)? else {
                        return Ok(None);
                    };
                    if cast.output() != prepared || !cast.is_owned_by(provider) {
                        return Err(invalid(
                            "resident cast proposal differs from selected binding",
                        ));
                    }
                    if cast.physical().output_buffer_bytes > binding_cap {
                        return Err(invalid(
                            "typed resident cast exceeds selected binding limit",
                        ));
                    }
                    let recipe_index = casts.len();
                    actions.push(InputAction::Cast {
                        source: handle,
                        recipe_index,
                    });
                    let stage = ProviderReductionInputStage::ResidentCast {
                        device_id: provider.device_id(),
                        recipe_identity: cast.identity().map_err(ProviderExecutionError::from)?,
                        physical: cast.physical().clone(),
                    };
                    casts.push(cast);
                    stage
                }
            }
            Value::Num(_) | Value::Tensor(_) => {
                if prepared.residency != ProviderResidency::Device
                    || prepared.layout != ProviderLayout::ColumnMajorContiguous
                {
                    return Ok(None);
                }
                let Some(query) = class_preserving_upload_query(vec![prepared.clone()]) else {
                    return Ok(None);
                };
                let Some(upload) = provider.upload_physical_evidence(&query) else {
                    return Ok(None);
                };
                if upload.output_requested_bytes < layout.bound_payload_bytes.max(4)
                    || upload.output_requested_bytes > binding_cap
                {
                    return Err(invalid(
                        "typed upload evidence violates selected binding limit",
                    ));
                }
                actions.push(InputAction::Upload {
                    source: input.value,
                    target: layout.bound_precision,
                });
                if layout.source_precision == layout.bound_precision {
                    ProviderReductionInputStage::UploadExact { upload }
                } else {
                    ProviderReductionInputStage::HostCastThenUpload {
                        upload,
                        conversion_workspace_bytes: layout.bound_payload_bytes,
                        conversion_elements: source.checked_element_count().ok_or_else(|| {
                            invalid("typed host conversion cardinality overflows")
                        })?,
                    }
                }
            }
            _ => unreachable!("source carrier validated in whole-set preflight"),
        };
        entries.push(ProviderReductionInputPreparation {
            source,
            prepared: prepared.clone(),
            source_physical: layout.source_physical,
            stage,
        });
    }
    let inventory =
        ProviderReductionInputInventory::new_checked(recipe.identity(), entries, &selected.inputs)
            .map_err(ProviderExecutionError::from)?;
    Ok(Some(TypedInputPlan {
        inventory,
        owner: provider,
        binding_cap,
        actions,
        casts,
    }))
}
