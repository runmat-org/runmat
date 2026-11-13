//! Exact selected offsets from the shared object index planner.

use super::super::super::{authority, format_error, MemmapDataType, MemmapFormat};
use super::{base_shape, field_width, mapped_error, scalar_format, Selected};
use crate::context::RuntimeContext;
use crate::execution::CpuCooperationBudget;
use crate::object::indexing::{ObjectIndexKind, ObjectIndexSelector, ObjectSubscriptPath};
use crate::BuiltinResult;
use runmat_builtins::{MEMMAPFILE_ERROR_READ_FAILED, MEMMAPFILE_ERROR_WRITE_FAILED};
use runmat_value::MemmapReceiverState;

async fn index_plan(
    shape: &[usize],
    step: &crate::object::indexing::ObjectSubscript,
    assignment: bool,
) -> BuiltinResult<crate::indexing::plan::IndexPlan> {
    if assignment {
        crate::object::indexing::plan_for_fixed_assignment(shape, step).await
    } else {
        crate::object::indexing::plan_for_shape(shape, step, false).await
    }
}

pub(super) async fn plan(
    runtime: Option<&RuntimeContext>,
    budget: &mut Option<CpuCooperationBudget>,
    state: &MemmapReceiverState,
    path: &ObjectSubscriptPath,
    assignment: bool,
) -> BuiltinResult<(authority::SharedBacking, Selected)> {
    let operation_error = if assignment {
        &MEMMAPFILE_ERROR_WRITE_FAILED
    } else {
        &MEMMAPFILE_ERROR_READ_FAILED
    };
    let backing = authority::resolve(runtime, state)?;
    let opened = if let Some(budget) = budget {
        budget
            .await_cancellable(backing.lock())
            .await
            .map_err(|source| source.into_runtime_error())?
    } else {
        backing.lock().await
    };
    opened
        .check_current()
        .map_err(|source| mapped_error(operation_error, source, 0))?;
    let extent = opened.opened_extent();
    drop(opened);
    let format = MemmapFormat::from_receiver(state)?;
    let record_bytes = format.validate_layout(state.repeat)?;
    let base =
        u64::try_from(state.offset).map_err(|source| mapped_error(operation_error, source, 0))?;
    let available = extent
        .checked_sub(base)
        .ok_or_else(|| format_error("memmapfile: Offset exceeds opened extent"))?;
    let available =
        usize::try_from(available).map_err(|source| mapped_error(operation_error, source, 0))?;
    let records = state
        .repeat
        .unwrap_or(available / record_bytes)
        .min(available / record_bytes);
    let steps = &path.steps()[1..];
    let mut next = 0usize;
    let mut record_positions = Vec::new();
    let mut record_shape = vec![records, 1];
    if format.field.is_some() {
        if steps
            .first()
            .is_some_and(|step| step.kind() == ObjectIndexKind::Paren)
        {
            let shape = [records, 1];
            let planning = index_plan(&shape, &steps[0], assignment);
            let planned = if let Some(budget) = budget {
                budget
                    .await_cancellable(planning)
                    .await
                    .map_err(|source| source.into_runtime_error())??
            } else {
                planning.await?
            };
            record_positions = planned
                .indices
                .iter()
                .map(|index| *index as usize)
                .collect();
            record_shape = planned.output_shape.to_vec();
            next = 1;
        } else if records == 1 {
            record_positions.push(0);
        } else {
            return Err(format_error(
                "memmapfile: select one mapped record before its field",
            ));
        }
        if next == steps.len() {
            let mut offsets = Vec::with_capacity(record_positions.len());
            for record in record_positions {
                offsets.push(
                    base.checked_add(
                        u64::try_from(
                            record.checked_mul(record_bytes).ok_or_else(|| {
                                format_error("memmapfile: record offset overflow")
                            })?,
                        )
                        .map_err(|source| mapped_error(operation_error, source, 0))?,
                    )
                    .ok_or_else(|| format_error("memmapfile: record offset overflow"))?,
                );
            }
            return Ok((
                backing,
                Selected {
                    shape: record_shape,
                    offsets,
                    unit_bytes: record_bytes,
                    format,
                    records: true,
                },
            ));
        }
        let step = &steps[next];
        if step.kind() != ObjectIndexKind::Member {
            return Err(format_error(
                "memmapfile: mapped record field must use member access",
            ));
        }
        let ObjectIndexSelector::Member(name) = step.selector() else {
            return Err(format_error("memmapfile: invalid mapped field selector"));
        };
        next += 1;
        if record_positions.len() != 1 {
            return Err(format_error(
                "memmapfile: select one record before selecting a field",
            ));
        }
        let mut prefix = 0usize;
        let mut selected = None;
        for field in &state.fields {
            let dtype = match field.element {
                runmat_value::MemmapElementType::Numeric(dtype) => MemmapDataType::Numeric(dtype),
                runmat_value::MemmapElementType::Character => MemmapDataType::Character,
            };
            if field.name.as_deref() == Some(name.as_str()) {
                selected = Some((dtype, field.shape.clone(), prefix));
                break;
            }
            prefix = prefix
                .checked_add(field_width(dtype, &field.shape)?)
                .ok_or_else(|| format_error("memmapfile: field offset overflow"))?;
        }
        let (dtype, shape, field_prefix) =
            selected.ok_or_else(|| format_error("memmapfile: unknown mapped field"))?;
        let record_prefix = record_positions[0]
            .checked_mul(record_bytes)
            .and_then(|value| value.checked_add(field_prefix))
            .ok_or_else(|| format_error("memmapfile: field offset overflow"))?;
        let field_base = base
            .checked_add(record_prefix as u64)
            .ok_or_else(|| format_error("memmapfile: field offset overflow"))?;
        if next == steps.len() {
            let unit_bytes = field_width(dtype, &shape)?;
            return Ok((
                backing,
                Selected {
                    offsets: vec![field_base],
                    unit_bytes,
                    shape: shape.clone(),
                    format: scalar_format(dtype, shape),
                    records: false,
                },
            ));
        }
        if next + 1 != steps.len() || steps[next].kind() != ObjectIndexKind::Paren {
            return Err(format_error(
                "memmapfile: unsupported mapped field subscript path",
            ));
        }
        let planning = index_plan(&shape, &steps[next], assignment);
        let planned = if let Some(budget) = budget {
            budget
                .await_cancellable(planning)
                .await
                .map_err(|source| source.into_runtime_error())??
        } else {
            planning.await?
        };
        let width = dtype.byte_size();
        let mut offsets = Vec::with_capacity(planned.indices.len());
        for &index in &planned.indices {
            let byte = (index as usize)
                .checked_mul(width)
                .ok_or_else(|| format_error("memmapfile: field index overflow"))?;
            offsets.push(
                field_base
                    .checked_add(byte as u64)
                    .ok_or_else(|| format_error("memmapfile: field index overflow"))?,
            );
        }
        return Ok((
            backing,
            Selected {
                offsets,
                unit_bytes: width,
                shape: planned.output_shape.to_vec(),
                format: scalar_format(dtype, planned.output_shape.to_vec()),
                records: false,
            },
        ));
    }
    if steps.len() != 1 || steps[0].kind() != ObjectIndexKind::Paren {
        return Err(format_error(
            "memmapfile: unsupported mapped Data subscript path",
        ));
    }
    let shape = base_shape(&format, records, state.repeat)?;
    let planning = index_plan(&shape, &steps[0], assignment);
    let planned = if let Some(budget) = budget {
        budget
            .await_cancellable(planning)
            .await
            .map_err(|source| source.into_runtime_error())??
    } else {
        planning.await?
    };
    let width = format.dtype.byte_size();
    let mut offsets = Vec::with_capacity(planned.indices.len());
    for &index in &planned.indices {
        let byte = (index as usize)
            .checked_mul(width)
            .ok_or_else(|| format_error("memmapfile: Data index overflow"))?;
        offsets.push(
            base.checked_add(byte as u64)
                .ok_or_else(|| format_error("memmapfile: Data index overflow"))?,
        );
    }
    Ok((
        backing,
        Selected {
            offsets,
            unit_bytes: width,
            shape: planned.output_shape.to_vec(),
            format: scalar_format(format.dtype, planned.output_shape.to_vec()),
            records: false,
        },
    ))
}
