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
                    .collect(),
            ),
            ObjectIndexSelector::Member(name) => Ok(Value::String(name.0.clone())),
        }
    }
}

impl ObjectSubscriptPath {
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
    match kind.as_str() {
        "()" => Ok(ObjectSubscript::parentheses(index_values(selector)?)),
        "{}" => Ok(ObjectSubscript::braces(index_values(selector)?)),
        "." => Ok(ObjectSubscript::member(scalar_text(selector)?)),
        _ => Err(invalid(format!("unsupported subscript type '{kind}'"))),
    }
}

fn index_values(value: &Value) -> Result<ObjectIndexSelector, RuntimeError> {
    let Value::Cell(cell) = value else {
        return Err(invalid(
            "parentheses and brace subscripts must be a cell array",
        ));
    };
    Ok(ObjectIndexSelector::IndexValues {
        components: cell
            .data
            .iter()
            .cloned()
            .map(ObjectIndexComponent::from_protocol_value)
            .collect(),
    })
}

fn scalar_text(value: &Value) -> Result<String, RuntimeError> {
    match value {
        Value::String(value) => Ok(value.clone()),
        Value::CharArray(value) if value.rows == 1 => Ok(value.data.iter().collect()),
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
