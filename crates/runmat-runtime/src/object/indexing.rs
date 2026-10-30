use runmat_types::StaticMethodName;
use runmat_value::Value;

#[cfg(test)]
use crate::{runtime_error::semantic_error, RuntimeError};

mod standard_substruct;
pub use standard_substruct::parse_standard_substruct;
mod path;
pub use path::{ObjectSubscript, ObjectSubscriptOrigin, ObjectSubscriptPath};
mod plan;
pub use plan::{
    plan_for_components, plan_for_components_in_context, plan_for_fixed_assignment,
    plan_for_fixed_assignment_in_context, plan_for_shape, plan_for_shape_in_context,
};
pub(crate) use plan::{plan_for_components_with_context, plan_for_virtual_components_with_context};
mod admission;
pub(crate) use admission::{allocation_error, array_in_context};
pub use admission::{
    prepare_components_in_context, prepare_subscripts_in_context, PreparedObjectIndexComponents,
    PreparedObjectSubscripts,
};

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
    /// Decode an indexing kind at the public protocol boundary.
    pub fn from_protocol_name(name: &str) -> Option<Self> {
        match name {
            OBJECT_PROTOCOL_KIND_PAREN => Some(Self::Paren),
            OBJECT_PROTOCOL_KIND_BRACE => Some(Self::Brace),
            OBJECT_PROTOCOL_KIND_MEMBER => Some(Self::Member),
            _ => None,
        }
    }

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
    /// Contextual syntax retained only until its actual receiver is known.
    End,
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

    pub fn protocol_value(&self) -> Result<Value, crate::RuntimeError> {
        match self {
            Self::Value(value) => Ok(value.clone()),
            Self::Colon => Ok(Value::string(OBJECT_SELECTOR_COLON.to_string())),
            Self::End => Err(unresolved_end()),
        }
    }
    pub(crate) fn protocol_value_in_context(
        &self,
        context: Option<&crate::context::RuntimeContext>,
    ) -> Result<Value, crate::RuntimeError> {
        match self {
            Self::Value(value) => Ok(value.clone()),
            Self::End => Err(unresolved_end()),
            Self::Colon => crate::call::argument_text::copy_str(OBJECT_SELECTOR_COLON, context)
                .map(|text| Value::String(text.into())),
        }
    }
}

pub(crate) fn unresolved_end() -> crate::RuntimeError {
    crate::runtime_error::semantic_error(
        "UnresolvedObjectIndexEnd",
        "contextual end requires its actual indexing receiver",
    )
}

impl ObjectSubscriptPath {
    pub(crate) fn has_deferred_end(&self) -> bool {
        self.steps().iter().any(|step| matches!(step.selector(),
            ObjectIndexSelector::IndexValues { components }
                if components.iter().any(|component| matches!(component, ObjectIndexComponent::End))))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ObjectIndexSelector {
    ScalarIndices {
        indices: Vec<usize>,
    },
    IndexValues {
        components: runmat_value::ExactArrayStorage<ObjectIndexComponent>,
    },
    Member(runmat_value::Utf8Text),
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
            ObjectSubscript::member(member.as_str().ok_or_else(|| {
                semantic_error("InvalidObjectIndex", "member name must be nonmissing text")
            })?)
        }
        OBJECT_PROTOCOL_KIND_PAREN | OBJECT_PROTOCOL_KIND_BRACE => {
            let Value::Cell(cell) = payload else {
                return Err(semantic_error(
                    "InvalidObjectIndex",
                    "subscript selectors must be provided in a cell array",
                ));
            };
            let selector = ObjectIndexSelector::IndexValues {
                components: runmat_value::ExactArrayStorage::shared_unbounded(
                    cell.data()
                        .iter()
                        .cloned()
                        .map(ObjectIndexComponent::from_protocol_value)
                        .collect::<Vec<_>>(),
                ),
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
        Value::Object(obj) => Some(obj.class_name()),
        Value::ObjectArray(array) => Some(array.class_name()),
        Value::HandleObject(handle) => Some(&handle.class_name),
        _ => None,
    }
}

pub(crate) fn class_name_owner_from_base(
    base: &Value,
) -> Option<&runmat_value::SharedClassIdentity> {
    match base {
        Value::Object(value) => Some(value.class_name_owner()),
        Value::ObjectArray(value) => Some(value.class_name_owner()),
        Value::HandleObject(value) => Some(&value.class_name),
        _ => None,
    }
}

#[cfg(test)]
#[path = "indexing/tests.rs"]
mod tests;
