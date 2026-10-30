//! String indexing over retained text and one admitted final slot owner.
use super::{
    primitive_output,
    scalar::{checked_shape_len, shape_error},
};
use crate::{
    context::{
        host_allocation::{
            initialize_host_array, HostAllocationCheckpoint, HostAllocationError,
            HostAllocationPolicy, HostAllocationReceipt,
        },
        RuntimeContext,
    },
    indexing::{
        plan::{admission::Owner, IndexPlan},
        write_slice::rhs_shape::AssignmentRhsLayout,
    },
    runtime_error::semantic_error,
    RuntimeError,
};
use runmat_value::{
    CharArray, ExactArrayInitializationError, ExactArrayStorage, SharedShape, StringArray,
    StringElement, Utf16TextCopyError, Utf8Text, Value,
};

enum TextSource {
    Scalar(StringElement),
    Tensor(ExactArrayStorage<StringElement>),
}
/// A checked ordinal view. Singleton-axis alignment is the canonical shared
/// RHS rule, not a second coordinate/stride planner.
pub struct StringRhsView {
    source: TextSource,
    layout: AssignmentRhsLayout,
    checkpoint: Option<HostAllocationCheckpoint>,
}
impl StringRhsView {
    fn check(&self, context: Option<&RuntimeContext>) -> Result<(), RuntimeError> {
        if let Some(checkpoint) = &self.checkpoint {
            checkpoint
                .check(context.ok_or_else(|| {
                    primitive_output::allocation(context, HostAllocationError::StaleAuthority)
                })?)
                .map_err(|e| primitive_output::allocation(context, e))?;
        } else if HostAllocationReceipt::policy(context)
            .map_err(|e| primitive_output::allocation(context, e))?
            != HostAllocationPolicy::Unbounded
        {
            return Err(primitive_output::allocation(
                context,
                HostAllocationError::StaleAuthority,
            ));
        }
        Owner::new(context)?.check()
    }
    pub(in crate::indexing) fn count(&self) -> usize {
        self.layout.count()
    }
    pub(in crate::indexing) fn value_at(&self, ordinal: usize) -> &StringElement {
        let index = self.layout.source_index(ordinal);
        match &self.source {
            TextSource::Scalar(value) => value,
            TextSource::Tensor(values) => &values[index],
        }
    }
}

fn retain(
    text: &StringElement,
    context: Option<&RuntimeContext>,
) -> Result<StringElement, RuntimeError> {
    match text {
        StringElement::Missing => {
            Owner::new(context)?.check()?;
            Ok(StringElement::Missing)
        }
        StringElement::Text(text) => {
            crate::call::argument_text::convert(&Value::String(text.clone().into()), context)
                .map(StringElement::Text)
        }
    }
}
fn chars_text(
    chars: &CharArray,
    context: Option<&RuntimeContext>,
    message: &str,
) -> Result<StringElement, RuntimeError> {
    if chars.rows != 1 {
        return Err(semantic_error("InvalidSliceAssignmentRhs", message));
    }
    let owner = Owner::new(context)?;
    Utf8Text::requested_bytes_for_utf16_with(chars.data(), |_| owner.check()).map_err(|error| {
        match error {
            ExactArrayInitializationError::Allocation(cause) => {
                primitive_output::allocation(context, cause.into())
            }
            ExactArrayInitializationError::Element(Utf16TextCopyError::Checkpoint(error)) => error,
            ExactArrayInitializationError::Element(Utf16TextCopyError::Decode(cause)) => {
                semantic_error(
                    "InvalidSliceAssignmentRhs",
                    format!("character RHS contains malformed UTF-16: {cause}"),
                )
            }
        }
    })?;
    crate::call::argument_text::copy_chars(chars.data(), context).map(StringElement::Text)
}
enum CellText<'a> {
    Text(&'a StringElement),
    Characters(&'a CharArray),
}
fn cell_text(value: &Value) -> Result<CellText<'_>, RuntimeError> {
    match value {
        Value::String(text) => Ok(CellText::Text(text)),
        Value::CharArray(chars) if chars.rows == 1 => Ok(CellText::Characters(chars)),
        Value::CharArray(_) => Err(semantic_error(
            "InvalidSliceAssignmentRhs",
            "rhs cell character arrays must be row vectors",
        )),
        Value::StringArray(strings) if strings.data().len() == 1 => {
            Ok(CellText::Text(&strings.data()[0]))
        }
        other => Err(semantic_error(
            "InvalidSliceAssignmentRhs",
            format!("rhs cell elements must be string scalars or character vectors, got {other:?}"),
        )),
    }
}

pub fn validate_rhs(rhs: &Value, selection: &[usize]) -> Result<(), RuntimeError> {
    match rhs {
        Value::String(_) => Ok(()),
        Value::CharArray(chars) if chars.rows == 1 => Ok(()),
        Value::CharArray(_) => Err(semantic_error(
            "InvalidSliceAssignmentRhs",
            "rhs character array must be a row vector",
        )),
        Value::StringArray(strings) => {
            AssignmentRhsLayout::multidimensional(&strings.shape, strings.data().len(), selection)
                .map(|_| ())
        }
        Value::Cell(cell) => {
            for value in cell.data() {
                cell_text(value)?;
            }
            AssignmentRhsLayout::multidimensional(&cell.shape, cell.data().len(), selection)
                .map(|_| ())
        }
        _ => Err(semantic_error(
            "InvalidSliceAssignmentRhs",
            "rhs must be string, string array, or cellstr",
        )),
    }
}

pub fn build_rhs(
    context: Option<&RuntimeContext>,
    rhs: &Value,
    selection: &[usize],
) -> Result<StringRhsView, RuntimeError> {
    let owner = Owner::new(context)?;
    let checkpoint = HostAllocationReceipt::capture_owner(context)
        .map_err(|e| primitive_output::allocation(context, e))?;
    let (source, layout) = match rhs {
        Value::String(text) => (
            TextSource::Scalar(retain(text, context)?),
            AssignmentRhsLayout::multidimensional(&[1, 1], 1, selection)?,
        ),
        Value::CharArray(chars) => (
            TextSource::Scalar(chars_text(
                chars,
                context,
                "rhs character array must be a row vector",
            )?),
            AssignmentRhsLayout::multidimensional(&[1, 1], 1, selection)?,
        ),
        Value::StringArray(strings) => {
            let layout = AssignmentRhsLayout::multidimensional(
                &strings.shape,
                strings.data().len(),
                selection,
            )?;
            (TextSource::Tensor(strings.data_owner().clone()), layout)
        }
        Value::Cell(cell) => {
            // Original type/UTF-16 conversion precedes shape validation.
            let data = initialize_host_array(context, cell.data().len(), |index| {
                if index % 4096 == 0 {
                    owner.check()?;
                }
                match cell_text(&cell.data()[index])? {
                    CellText::Text(text) => retain(text, context),
                    CellText::Characters(chars) => chars_text(
                        chars,
                        context,
                        "rhs cell character arrays must be row vectors",
                    ),
                }
            })?;
            let layout =
                AssignmentRhsLayout::multidimensional(&cell.shape, cell.data().len(), selection)?;
            (TextSource::Tensor(data), layout)
        }
        _ => {
            return Err(semantic_error(
                "InvalidSliceAssignmentRhs",
                "rhs must be string, string array, or cellstr",
            ))
        }
    };
    owner.check()?;
    Ok(StringRhsView {
        source,
        layout,
        checkpoint,
    })
}

pub fn read(
    context: Option<&RuntimeContext>,
    source: &[StringElement],
    plan: &IndexPlan,
) -> Result<Value, RuntimeError> {
    let owner = Owner::new(context)?;
    let at = |index: usize| {
        source.get(index).ok_or_else(|| {
            semantic_error(
                "IndexOutOfBounds",
                "Slice error: string index out of bounds",
            )
        })
    };
    if let [index] = plan.indices.as_slice() {
        return retain(at(*index as usize)?, context).map(|text| Value::String(text.into()));
    }
    for (ordinal, &index) in plan.indices.iter().enumerate() {
        if ordinal % 4096 == 0 {
            owner.check()?;
        }
        at(index as usize)?;
    }
    let shape = primitive_output::shape(context, &plan.output_shape)?;
    construct(
        context,
        shape,
        plan.indices.len(),
        |ordinal| retain(at(plan.indices[ordinal] as usize)?, context),
        |_| Ok(()),
    )
    .map(Value::StringArray)
}

pub fn assign(
    context: Option<&RuntimeContext>,
    mut value: StringArray,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    let target = checked_shape_len(&plan.base_shape)?;
    let view = build_rhs(context, rhs, &plan.selection_lengths)?;
    let count = value.data().len().max(target);
    validate_plan(context, plan, &view, count)?;
    if target <= value.data().len() {
        if let Some(data) = value.try_data_mut_unique() {
            scatter(context, data, plan, &view)?;
            return Ok(Value::StringArray(value));
        }
    }
    let shape = primitive_output::shape(
        context,
        if target > value.data().len() {
            &plan.base_shape
        } else {
            &value.shape
        },
    )?;
    construct(
        context,
        shape,
        count,
        |index| match value.data().get(index) {
            Some(text) => retain(text, context),
            None => Ok(StringElement::default()),
        },
        |data| scatter(context, data, plan, &view),
    )
    .map(Value::StringArray)
}

pub(super) fn assign_scalar(
    context: Option<&RuntimeContext>,
    value: StringElement,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    let target = checked_shape_len(&plan.base_shape)?;
    let view = build_rhs(context, rhs, &plan.selection_lengths)?;
    let count = target.max(1);
    validate_plan(context, plan, &view, count)?;
    let shape = if target > 1 {
        primitive_output::shape(context, &plan.base_shape)?
    } else {
        crate::indexing::plan::admission::shape(context, 2, |_| Ok(1))?
    };
    construct(
        context,
        shape,
        count,
        |index| {
            if index == 0 {
                retain(&value, context)
            } else {
                Ok(StringElement::default())
            }
        },
        |data| scatter(context, data, plan, &view),
    )
    .map(Value::StringArray)
}

pub fn scatter_existing(
    context: Option<&RuntimeContext>,
    value: &mut StringArray,
    plan: &IndexPlan,
    view: &StringRhsView,
) -> Result<(), RuntimeError> {
    validate_plan(context, plan, view, value.data().len())?;
    if plan.indices.is_empty() {
        return Ok(());
    }
    if let Some(data) = value.try_data_mut_unique() {
        return scatter(context, data, plan, view);
    }
    let shape = primitive_output::shape(context, &value.shape)?;
    let replacement = construct(
        context,
        shape,
        value.data().len(),
        |index| retain(&value.data()[index], context),
        |data| scatter(context, data, plan, view),
    )?;
    *value = replacement;
    Ok(())
}

fn validate_plan(
    context: Option<&RuntimeContext>,
    plan: &IndexPlan,
    view: &StringRhsView,
    count: usize,
) -> Result<(), RuntimeError> {
    view.check(context)?;
    if view.count() != plan.indices.len() {
        return Err(semantic_error(
            "ShapeMismatch",
            "shape mismatch for slice assign",
        ));
    }
    let owner = Owner::new(context)?;
    for (ordinal, &index) in plan.indices.iter().enumerate() {
        if ordinal % 4096 == 0 {
            owner.check()?;
        }
        if index as usize >= count {
            return Err(semantic_error(
                "IndexOutOfBounds",
                "string assignment index out of bounds",
            ));
        }
    }
    Ok(())
}
fn scatter(
    context: Option<&RuntimeContext>,
    data: &mut [StringElement],
    plan: &IndexPlan,
    view: &StringRhsView,
) -> Result<(), RuntimeError> {
    for (ordinal, &index) in plan.indices.iter().enumerate() {
        if ordinal % 4096 == 0 {
            view.check(context)?;
        }
        data[index as usize] = retain(view.value_at(ordinal), context)?;
    }
    view.check(context)
}
fn construct(
    context: Option<&RuntimeContext>,
    shape: SharedShape,
    count: usize,
    mut initialize: impl FnMut(usize) -> Result<StringElement, RuntimeError>,
    finish: impl FnOnce(&mut [StringElement]) -> Result<(), RuntimeError>,
) -> Result<StringArray, RuntimeError> {
    let owner = Owner::new(context)?;
    let mut data = initialize_host_array(context, count, |index| {
        if index % 4096 == 0 {
            owner.check()?;
        }
        initialize(index)
    })?;
    finish(
        data.try_as_mut_slice_unique()
            .expect("new string backing is unique"),
    )?;
    owner.check()?;
    let result = if data.is_exact() && shape.is_admitted() {
        StringArray::from_admitted(data, shape)
    } else {
        StringArray::from_shared_storage(data, shape)
    }
    .map_err(shape_error)?;
    owner.check()?;
    Ok(result)
}
