use super::{
    ObjectIndexComponent, ObjectIndexKind, ObjectIndexSelector, ObjectSubscript,
    ObjectSubscriptPath,
};
use crate::runtime_error::semantic_error;
use crate::RuntimeError;
use runmat_value::Value;

pub fn parse_standard_substruct(value: &Value) -> Result<ObjectSubscriptPath, RuntimeError> {
    let (types, selectors): (Vec<&Value>, Vec<&Value>) = match value {
        Value::Struct(structure) => (
            vec![required_field(structure.fields.get("type"), "type")?],
            vec![required_field(structure.fields.get("subs"), "subs")?],
        ),
        Value::StructArray(array) => (
            required_column(array.field_values("type"), "type")?
                .iter()
                .collect(),
            required_column(array.field_values("subs"), "subs")?
                .iter()
                .collect(),
        ),
        _ => return Err(invalid("subscript descriptor must be a structure array")),
    };
    if types.len() != selectors.len() {
        return Err(invalid(
            "subscript descriptor fields have inconsistent lengths",
        ));
    }
    let steps = types
        .into_iter()
        .zip(selectors)
        .map(|(kind, selector)| parse_step(kind, selector))
        .collect::<Result<Vec<_>, _>>()?;
    ObjectSubscriptPath::new(steps)
}

impl ObjectSubscript {
    fn selector_value_in_context(
        &self,
        runtime: &crate::context::RuntimeContext,
    ) -> Result<crate::execution::RootedValue, RuntimeError> {
        use crate::execution::{RootedValue, RootedValues};
        let slots = match &self.selector {
            ObjectIndexSelector::ScalarIndices { indices } => {
                RootedValues::try_initialize_in_context(indices.len(), runtime, |index| {
                    Ok::<_, RuntimeError>(Value::Num(indices[index] as f64))
                })?
            }
            ObjectIndexSelector::IndexValues { components } => {
                RootedValues::try_initialize_in_context(components.len(), runtime, |index| {
                    components[index].protocol_value_in_context(Some(runtime))
                })?
            }
            ObjectIndexSelector::Member(name) => {
                return RootedValue::in_context(Value::String(name.clone().into()), runtime);
            }
        };
        let count = slots.len();
        crate::execution::structure::cell::from_rooted_values(runtime, &[1, count], slots)
    }

    pub fn selector_value(&self) -> Result<Value, RuntimeError> {
        match &self.selector {
            ObjectIndexSelector::ScalarIndices { indices } => protocol_index_cell(
                indices
                    .iter()
                    .map(|index| Value::Num(*index as f64))
                    .collect(),
            ),
            ObjectIndexSelector::IndexValues { components } => protocol_index_cell(
                components
                    .iter()
                    .map(ObjectIndexComponent::protocol_value)
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            ObjectIndexSelector::Member(name) => Ok(Value::String(name.clone().into())),
        }
    }
}

impl ObjectSubscriptPath {
    /// The managed protocol boundary constructs directly into canonical final
    /// columns/cells. The path's original GC children stay rooted throughout
    /// all initializer calls, including previously initialized selector cells.
    pub(crate) fn to_standard_substruct_value_in_context(
        &self,
        runtime: &crate::context::RuntimeContext,
    ) -> Result<crate::execution::RootedValue, RuntimeError> {
        let source_values = self.0.iter().flat_map(|step| {
            let components: &[ObjectIndexComponent] = match &step.selector {
                ObjectIndexSelector::IndexValues { components } => components,
                _ => &[],
            };
            components.iter().filter_map(|component| match component {
                ObjectIndexComponent::Value(value) => Some(value),
                ObjectIndexComponent::Colon | ObjectIndexComponent::End => None,
            })
        });
        let _source_roots = crate::execution::RootedValueLease::for_values_iterator_in_context(
            source_values,
            runtime,
        )?;
        crate::execution::structure::construct_generated_structure_owned(
            Some(runtime),
            ["type", "subs"].into_iter(),
            &[1, self.0.len()],
            |field, index| {
                let step = &self.0[index];
                if field == "type" {
                    crate::call::argument_text::copy_str(
                        standard_subscript_type(step.kind),
                        Some(runtime),
                    )
                    .map(|text| Value::String(text.into()))
                } else {
                    step.selector_value_in_context(runtime)
                        .map(|value| value.with_owned(|value| value))
                }
            },
        )
    }

    pub fn to_standard_substruct_value(&self) -> Result<Value, RuntimeError> {
        let mut types = Vec::with_capacity(self.0.len());
        let mut selectors = Vec::with_capacity(self.0.len());
        for step in &self.0 {
            types.push(Value::String(standard_subscript_type(step.kind).into()));
            selectors.push(step.selector_value()?);
        }
        let length = types.len();
        types.extend(selectors);
        runmat_value::StructArray::normalize_field_major(
            vec!["type".to_string(), "subs".to_string()],
            types,
            vec![1, length],
        )
        .map_err(|error| semantic_error("InvalidObjectSubscriptPath", error))
    }
}

fn protocol_index_cell(values: Vec<Value>) -> Result<Value, RuntimeError> {
    let cols = values.len();
    let cell = runmat_value::CellArray::new(values, 1, cols)
        .map_err(|error| semantic_error("ShapeMismatch", format!("standard substruct: {error}")))?;
    Ok(Value::Cell(cell))
}

fn standard_subscript_type(kind: ObjectIndexKind) -> &'static str {
    match kind {
        ObjectIndexKind::Paren => "()",
        ObjectIndexKind::Brace => "{}",
        ObjectIndexKind::Member => ".",
    }
}

fn parse_step(kind: &Value, selector: &Value) -> Result<ObjectSubscript, RuntimeError> {
    let kind = scalar_text(kind)?;
    match ObjectIndexKind::from_protocol_name(&kind) {
        Some(ObjectIndexKind::Paren) => Ok(ObjectSubscript::parentheses(index_values(selector)?)),
        Some(ObjectIndexKind::Brace) => Ok(ObjectSubscript::braces(index_values(selector)?)),
        Some(ObjectIndexKind::Member) => Ok(ObjectSubscript::member(scalar_text(selector)?)),
        None => Err(invalid(format!("unsupported subscript type '{kind}'"))),
    }
}

fn index_values(value: &Value) -> Result<ObjectIndexSelector, RuntimeError> {
    let Value::Cell(cell) = value else {
        return Err(invalid(
            "parentheses and brace subscripts must be a cell array",
        ));
    };
    Ok(ObjectIndexSelector::IndexValues {
        components: runmat_value::ExactArrayStorage::shared_unbounded(
            cell.data()
                .iter()
                .cloned()
                .map(ObjectIndexComponent::from_protocol_value)
                .collect::<Vec<_>>(),
        ),
    })
}

fn scalar_text(value: &Value) -> Result<String, RuntimeError> {
    match value {
        Value::String(value) => value
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| invalid("subscript type and member selectors must be scalar text")),
        Value::CharArray(value) if value.rows == 1 => value
            .try_row_string()
            .map_err(|source| {
                crate::runtime_error::semantic_error_with_source(
                    "InvalidObjectSubscriptPath",
                    "subscript text contains malformed UTF-16",
                    source,
                )
            })?
            .ok_or_else(|| invalid("subscript type and member selectors must be scalar text")),
        _ => Err(invalid(
            "subscript type and member selectors must be scalar text",
        )),
    }
}

fn required_field<'a>(value: Option<&'a Value>, field: &str) -> Result<&'a Value, RuntimeError> {
    value.ok_or_else(|| invalid(format!("subscript descriptor is missing field '{field}'")))
}

fn required_column<'a>(
    value: Option<&'a [Value]>,
    field: &str,
) -> Result<&'a [Value], RuntimeError> {
    value.ok_or_else(|| invalid(format!("subscript descriptor is missing field '{field}'")))
}

fn invalid(message: impl Into<String>) -> RuntimeError {
    semantic_error("InvalidObjectSubscriptPath", message)
}
