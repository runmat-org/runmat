use runmat_types::StaticMethodName;
use runmat_value::Value;

#[cfg(test)]
use crate::{runtime_error::semantic_error, RuntimeError};

mod standard_substruct;
pub use standard_substruct::parse_standard_substruct;
mod path;
pub use path::{ObjectSubscript, ObjectSubscriptOrigin, ObjectSubscriptPath};

pub const OBJECT_PROTOCOL_SUBSREF: StaticMethodName = crate::OBJECT_SUBSREF_METHOD;
pub const OBJECT_PROTOCOL_SUBSASGN: StaticMethodName = crate::OBJECT_SUBSASGN_METHOD;
pub const OBJECT_PROTOCOL_KIND_PAREN: &str = crate::OBJECT_INDEX_PAREN;
pub const OBJECT_PROTOCOL_KIND_BRACE: &str = crate::OBJECT_INDEX_BRACE;
pub const OBJECT_PROTOCOL_KIND_MEMBER: &str = crate::OBJECT_INDEX_MEMBER;
pub const OBJECT_SELECTOR_COLON: &str = ":";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectIndexOp {
    Subsref,
    Subsasgn,
}

impl ObjectIndexOp {
    pub fn protocol_name(self) -> StaticMethodName {
        match self {
            Self::Subsref => OBJECT_PROTOCOL_SUBSREF,
            Self::Subsasgn => OBJECT_PROTOCOL_SUBSASGN,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectIndexKind {
    Paren,
    Brace,
    Member,
}

impl ObjectIndexKind {
    pub fn protocol_name(self) -> &'static str {
        match self {
            Self::Paren => OBJECT_PROTOCOL_KIND_PAREN,
            Self::Brace => OBJECT_PROTOCOL_KIND_BRACE,
            Self::Member => OBJECT_PROTOCOL_KIND_MEMBER,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ObjectIndexComponent {
    Value(Value),
    Colon,
}

impl From<Value> for ObjectIndexComponent {
    fn from(value: Value) -> Self {
        Self::Value(value)
    }
}

impl ObjectIndexComponent {
    /// Decode the public standard-substruct representation at its boundary.
    pub fn from_protocol_value(value: Value) -> Self {
        if matches!(&value, Value::String(text) if text == OBJECT_SELECTOR_COLON) {
            Self::Colon
        } else {
            Self::Value(value)
        }
    }

    pub fn protocol_value(&self) -> Value {
        match self {
            Self::Value(value) => value.clone(),
            Self::Colon => Value::String(OBJECT_SELECTOR_COLON.to_string()),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ObjectIndexSelector {
    ScalarIndices {
        indices: Vec<usize>,
    },
    IndexValues {
        components: Vec<ObjectIndexComponent>,
    },
    Member(runmat_types::MemberName),
}

#[cfg(test)]
pub(crate) fn standard_substruct_fixture_from_parts(
    kind: &str,
    payload: Value,
) -> Result<Value, RuntimeError> {
    let step = match kind {
        OBJECT_PROTOCOL_KIND_MEMBER => {
            let Value::String(member) = payload else {
                return Err(semantic_error(
                    "InvalidObjectIndex",
                    "member name must be text",
                ));
            };
            ObjectSubscript::member(member)
        }
        OBJECT_PROTOCOL_KIND_PAREN | OBJECT_PROTOCOL_KIND_BRACE => {
            let Value::Cell(cell) = payload else {
                return Err(semantic_error(
                    "InvalidObjectIndex",
                    "subscript selectors must be provided in a cell array",
                ));
            };
            let selector = ObjectIndexSelector::IndexValues {
                components: cell
                    .data
                    .into_iter()
                    .map(ObjectIndexComponent::from_protocol_value)
                    .collect(),
            };
            if kind == OBJECT_PROTOCOL_KIND_PAREN {
                ObjectSubscript::parentheses(selector)
            } else {
                ObjectSubscript::braces(selector)
            }
        }
        _ => {
            return Err(semantic_error(
                "InvalidObjectIndex",
                "unknown subscript kind in test fixture",
            ))
        }
    };
    ObjectSubscriptPath::single(step).to_standard_substruct_value()
}

pub fn class_name_from_base(base: &Value) -> Option<&runmat_types::ClassIdentity> {
    match base {
        Value::Object(obj) => Some(&obj.class_name),
        Value::ObjectArray(array) => Some(array.class_name()),
        Value::HandleObject(handle) => Some(&handle.class_name),
        _ => None,
    }
}

#[cfg(test)]
#[path = "indexing/tests.rs"]
mod tests;
