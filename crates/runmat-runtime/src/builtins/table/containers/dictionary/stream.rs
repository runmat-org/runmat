//! Borrow native bits until a value really enters retained dictionary storage.
use super::*;
use crate::object::indexing::{ObjectIndexComponent, ObjectIndexSelector};
use runmat_value::{NumericDType, StringElement, Utf16CodeUnit};

#[derive(Clone, Copy)]
pub(super) enum Element<'a> {
    Borrowed(&'a Value),
    Numeric(NumericScalar),
    Logical(bool),
    Text(&'a StringElement),
    Chars(&'a [Utf16CodeUnit]),
    Literal(&'static str),
}

impl<'a> Element<'a> {
    pub(super) fn numeric(self) -> Option<NumericScalar> {
        match self {
            Self::Numeric(value) => Some(value),
            Self::Borrowed(Value::Num(value)) => Some(NumericScalar::F64(*value)),
            Self::Borrowed(Value::Int(value)) => Some(value.clone().into()),
            Self::Borrowed(Value::Tensor(value))
                if value.len() == 1 && value.numeric_dtype() == NumericDType::F32 =>
            {
                value.numeric_value_at(0)
            }
            _ => None,
        }
    }
    pub(super) fn normalize(self, prototype: Option<Self>) -> BuiltinResult<Self> {
        let Some(prototype) = prototype else {
            return Ok(self);
        };
        if let Some(target) = prototype.numeric() {
            let value = self.numeric().ok_or_else(|| {
                invalid_argument(
                    "dictionary: entry is not convertible to the configured numeric type",
                )
            })?;
            // Scalar Single aliases retain their original payload and shape.
            if matches!(self, Self::Borrowed(Value::Tensor(t)) if t.len() == 1
                && t.numeric_dtype() == NumericDType::F32)
                && target.numeric_dtype() == NumericDType::F32
            {
                return Ok(self);
            }
            let value = match target {
                NumericScalar::F32(_) => NumericScalar::F32(
                    crate::builtins::math::elementwise::floating_conversions::single::numeric_scalar_to_f32(value)),
                NumericScalar::F64(_) => NumericScalar::F64(value.materialize_f64()),
                target => {
                    let target = target.into_int_value().expect("integer prototype");
                    let class = crate::builtins::common::integer_conversion::IntegerClass::from_int_value(&target);
                    match value.into_int_value() {
                        Some(value) => class.cast_int(&value).into(),
                        None => class.cast_scalar(value.materialize_f64()).into(),
                    }
                }
            };
            return Ok(Self::Numeric(value));
        }
        if prototype.text()?.is_some() {
            return match self {
                Self::Borrowed(Value::CharArray(array)) if array.rows == 1 => {
                    Ok(Self::Chars(array.data()))
                }
                value if value.text()?.is_some() => Ok(value),
                _ => Err(invalid_argument(
                    "dictionary: entry is not convertible to string",
                )),
            };
        }
        match (prototype, self) {
            (
                Self::Logical(_) | Self::Borrowed(Value::Bool(_)),
                Self::Logical(_) | Self::Borrowed(Value::Bool(_)),
            ) => Ok(self),
            (Self::Borrowed(first), Self::Borrowed(value))
                if configured_classes_equal(first, value) =>
            {
                Ok(self)
            }
            _ => Err(invalid_argument(
                "dictionary: entries must use one compatible configured type",
            )),
        }
    }
    // A character row is text only after normalizing it to a string class.
    fn text(self) -> BuiltinResult<Option<Text<'a>>> {
        Ok(match self {
            Self::Text(value) | Self::Borrowed(Value::String(value)) => {
                Some(value.as_str().map(Text::Utf8).unwrap_or(Text::Missing))
            }
            Self::Chars(value) => Some(Text::Owned(
                runmat_value::decode_utf16_code_units(value).map_err(|source| {
                    with_utf16_source(
                        invalid_argument("dictionary: character key is not valid UTF-16 text"),
                        source,
                    )
                })?,
            )),
            Self::Literal(value) => Some(Text::Utf8(value)),
            _ => None,
        })
    }
    pub(super) fn equal(self, other: Self) -> BuiltinResult<bool> {
        if let (Some(left), Some(right)) = (self.numeric(), other.numeric()) {
            return Ok(left == right);
        }
        let text = |value| -> BuiltinResult<Option<Text<'_>>> {
            match value {
                Self::Borrowed(Value::CharArray(array)) if array.rows == 1 => {
                    Element::Chars(array.data()).text()
                }
                value => value.text(),
            }
        };
        if let (Some(left), Some(right)) = (text(self)?, text(other)?) {
            return Ok(match (left, right) {
                (Text::Missing, Text::Missing) => true,
                (Text::Missing, _) | (_, Text::Missing) => false,
                (Text::Utf8(a), Text::Utf8(b)) => a == b,
                (Text::Owned(a), Text::Owned(b)) => a == b,
                (Text::Utf8(a), Text::Owned(b)) | (Text::Owned(b), Text::Utf8(a)) => a == b,
            });
        }
        Ok(match (self, other) {
            (Self::Logical(a), Self::Logical(b)) => a == b,
            (Self::Logical(a), Self::Borrowed(Value::Bool(b)))
            | (Self::Borrowed(Value::Bool(b)), Self::Logical(a)) => a == *b,
            (Self::Borrowed(a), Self::Borrowed(b)) => a == b,
            _ => false,
        })
    }
}

// Generic Value discriminants do not distinguish runtime object classes.
// Borrow canonical identity owners, including a handle's actual GC target,
// without object_receiver_class_name's allocating ClassIdentity::clone.
fn with_object_class<R>(
    value: &Value,
    mut compare: impl FnMut(&runmat_types::ClassIdentity) -> R,
) -> Option<R> {
    match value {
        Value::Object(object) => Some(compare(object.class_name())),
        Value::ObjectArray(array) => Some(compare(array.class_name())),
        Value::HandleObject(handle) => Some(
            runmat_gc::gc_with_value(&handle.target, |target| match target {
                Value::Object(object) => compare(object.class_name()),
                _ => compare(handle.class_name.as_identity()),
            })
            .unwrap_or_else(|_| compare(handle.class_name.as_identity())),
        ),
        _ => None,
    }
}
fn configured_classes_equal(first: &Value, value: &Value) -> bool {
    if std::mem::discriminant(first) != std::mem::discriminant(value) {
        return false;
    }
    match first {
        Value::Object(_) | Value::ObjectArray(_) | Value::HandleObject(_) => {
            with_object_class(first, |first_class| {
                with_object_class(value, |value_class| first_class == value_class).unwrap_or(false)
            })
            .unwrap_or(false)
        }
        _ => true,
    }
}
#[derive(Clone)]
enum Text<'a> {
    Missing,
    Utf8(&'a str),
    Owned(String),
}

pub(super) fn element_count(value: &Value) -> usize {
    match value {
        Value::Cell(value) => value.data().len(),
        Value::Tensor(value) => value.len(),
        Value::LogicalArray(value) => value.len(),
        Value::StringArray(value) => value.data().len(),
        Value::CharArray(value) => value.rows,
        _ => 1,
    }
}
pub(super) fn element_at(value: &Value, index: usize) -> BuiltinResult<Element<'_>> {
    let element = match value {
        Value::Cell(value) => Element::Borrowed(
            value
                .get_column_major(index)
                .ok_or_else(|| invalid_index("dictionary: cell index out of bounds"))?,
        ),
        Value::Tensor(tensor)
            if tensor.len() == 1
                && tensor.shape() == [1, 1]
                && tensor.numeric_dtype() == NumericDType::F32
                && index == 0 =>
        {
            Element::Borrowed(value)
        }
        Value::Tensor(value) => Element::Numeric(
            value
                .numeric_value_at(index)
                .ok_or_else(|| invalid_index("dictionary: tensor index out of bounds"))?,
        ),
        Value::LogicalArray(value) => Element::Logical(
            *value
                .data
                .get(index)
                .ok_or_else(|| invalid_index("dictionary: logical index out of bounds"))?
                != 0,
        ),
        Value::StringArray(value) => Element::Text(
            value
                .data()
                .get(index)
                .ok_or_else(|| invalid_index("dictionary: string index out of bounds"))?,
        ),
        Value::CharArray(value) if index < value.rows => {
            let start = index * value.cols;
            Element::Chars(&value.data()[start..start + value.cols])
        }
        _ if index == 0 => Element::Borrowed(value),
        _ => return Err(invalid_index("dictionary: element index out of bounds")),
    };
    Ok(element)
}

#[derive(Clone, Copy)]
pub(super) enum Shape<'a> {
    Borrowed(&'a [usize]),
    Pair([usize; 2]),
}
impl Shape<'_> {
    pub(super) fn as_slice(&self) -> &[usize] {
        match self {
            Self::Borrowed(value) => value,
            Self::Pair(value) => value,
        }
    }
}
pub(super) fn element_shape(value: &Value) -> Shape<'_> {
    match value {
        Value::Tensor(value) => Shape::Borrowed(value.shape()),
        Value::LogicalArray(value) => Shape::Borrowed(&value.shape),
        Value::StringArray(value) => Shape::Borrowed(&value.shape),
        Value::Cell(value) => Shape::Borrowed(value.shape()),
        Value::CharArray(value) => Shape::Pair([value.rows, 1]),
        _ => Shape::Pair([1, 1]),
    }
}

#[derive(Clone, Copy)]
pub(in crate::builtins::table) enum Query<'a> {
    #[cfg_attr(not(test), allow(dead_code))]
    Payload(&'a Value),
    Typed(&'a ObjectIndexSelector),
}
impl<'a> Query<'a> {
    pub(super) fn roots(
        self,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> BuiltinResult<Option<crate::execution::RootedValueLease>> {
        runtime
            .map(|runtime| {
                crate::execution::RootedValueLease::for_values_iterator_in_context(
                    (0..self.components()).filter_map(move |index| match self.component(index) {
                        Ok(Component::Value(value)) => Some(value),
                        _ => None,
                    }),
                    runtime,
                )
            })
            .transpose()
    }
    fn components(self) -> usize {
        match self {
            Self::Payload(Value::Cell(cell)) => cell.data().len(),
            Self::Payload(_) => 1,
            Self::Typed(ObjectIndexSelector::IndexValues { components }) => components.len(),
            Self::Typed(ObjectIndexSelector::ScalarIndices { indices }) => indices.len(),
            Self::Typed(ObjectIndexSelector::Member(_)) => 0,
        }
    }
    fn component(self, index: usize) -> BuiltinResult<Component<'a>> {
        Ok(match self {
            Self::Payload(Value::Cell(cell)) => Component::Value(
                cell.get_column_major(index)
                    .ok_or_else(|| invalid_index("dictionary: cell query index out of bounds"))?,
            ),
            Self::Payload(value) if index == 0 => Component::Value(value),
            Self::Typed(ObjectIndexSelector::IndexValues { components }) => {
                match &components[index] {
                    ObjectIndexComponent::Value(value) => Component::Value(value),
                    ObjectIndexComponent::Colon => Component::Scalar(Element::Literal(":")),
                    ObjectIndexComponent::End => {
                        return Err(crate::object::indexing::unresolved_end())
                    }
                }
            }
            Self::Typed(ObjectIndexSelector::ScalarIndices { indices }) => {
                Component::Scalar(Element::Numeric(NumericScalar::F64(indices[index] as f64)))
            }
            _ => return Err(invalid_index("dictionary: invalid key selector")),
        })
    }
    pub(super) fn count(self) -> BuiltinResult<usize> {
        (0..self.components()).try_fold(0usize, |count, index| {
            count
                .checked_add(self.component(index)?.count())
                .ok_or_else(|| invalid_index("dictionary: selector size overflow"))
        })
    }
    pub(super) fn at(self, mut index: usize) -> BuiltinResult<Element<'a>> {
        for component in 0..self.components() {
            let component = self.component(component)?;
            if index < component.count() {
                return component.at(index);
            }
            index -= component.count();
        }
        Err(invalid_index("dictionary: query index out of bounds"))
    }
    pub(super) fn shape(self, count: usize) -> BuiltinResult<Shape<'a>> {
        let shape = if self.components() == 1 {
            match self.component(0)? {
                Component::Value(value) => element_shape(value),
                Component::Scalar(_) => Shape::Pair([1, 1]),
            }
        } else {
            match self {
                Self::Payload(value) => element_shape(value),
                _ => Shape::Pair([1, self.components()]),
            }
        };
        Ok(
            if shape
                .as_slice()
                .iter()
                .try_fold(1usize, |a, b| a.checked_mul(*b))
                == Some(count)
            {
                shape
            } else {
                Shape::Pair([1, count])
            },
        )
    }
}
enum Component<'a> {
    Value(&'a Value),
    Scalar(Element<'a>),
}
impl<'a> Component<'a> {
    fn count(&self) -> usize {
        match self {
            Self::Value(value) => element_count(value),
            Self::Scalar(_) => 1,
        }
    }
    fn at(self, index: usize) -> BuiltinResult<Element<'a>> {
        match self {
            Self::Value(value) => element_at(value, index),
            Self::Scalar(value) if index == 0 => Ok(value),
            _ => Err(invalid_index("dictionary: component index out of bounds")),
        }
    }
}
