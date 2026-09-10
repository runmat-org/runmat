use super::{ObjectIndexKind, ObjectIndexSelector};
use crate::runtime_error::semantic_error;
use crate::RuntimeError;
use runmat_value::Value;

#[derive(Debug, Clone, PartialEq)]
pub struct ObjectSubscript {
    pub(super) kind: ObjectIndexKind,
    pub(super) selector: ObjectIndexSelector,
    pub(super) origin: ObjectSubscriptOrigin,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectSubscriptOrigin {
    Ordinary,
    DottedInvokeMember,
    DottedInvokeArguments,
}

impl ObjectSubscript {
    pub fn parentheses(selector: ObjectIndexSelector) -> Self {
        Self {
            kind: ObjectIndexKind::Paren,
            selector,
            origin: ObjectSubscriptOrigin::Ordinary,
        }
    }

    pub fn braces(selector: ObjectIndexSelector) -> Self {
        Self {
            kind: ObjectIndexKind::Brace,
            selector,
            origin: ObjectSubscriptOrigin::Ordinary,
        }
    }

    pub fn member(name: impl Into<runmat_types::MemberName>) -> Self {
        Self {
            kind: ObjectIndexKind::Member,
            selector: ObjectIndexSelector::Member(name.into()),
            origin: ObjectSubscriptOrigin::Ordinary,
        }
    }

    /// Preserve a source dotted invocation as two standard substruct steps
    /// while retaining typed provenance for default method dispatch.
    pub fn dotted_invoke(
        name: impl Into<runmat_types::MemberName>,
        arguments: Vec<Value>,
    ) -> [Self; 2] {
        Self::dotted_invoke_components(name, arguments.into_iter().map(Into::into).collect())
    }

    pub fn dotted_invoke_components(
        name: impl Into<runmat_types::MemberName>,
        arguments: Vec<super::ObjectIndexComponent>,
    ) -> [Self; 2] {
        [
            Self {
                kind: ObjectIndexKind::Member,
                selector: ObjectIndexSelector::Member(name.into()),
                origin: ObjectSubscriptOrigin::DottedInvokeMember,
            },
            Self {
                kind: ObjectIndexKind::Paren,
                selector: ObjectIndexSelector::IndexValues {
                    components: arguments,
                },
                origin: ObjectSubscriptOrigin::DottedInvokeArguments,
            },
        ]
    }

    pub const fn kind(&self) -> ObjectIndexKind {
        self.kind
    }
    pub const fn selector(&self) -> &ObjectIndexSelector {
        &self.selector
    }
    pub const fn origin(&self) -> ObjectSubscriptOrigin {
        self.origin
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObjectSubscriptPath(pub(super) Vec<ObjectSubscript>);

impl ObjectSubscriptPath {
    pub fn new(steps: Vec<ObjectSubscript>) -> Result<Self, RuntimeError> {
        if steps.is_empty() {
            return Err(semantic_error(
                "InvalidObjectSubscriptPath",
                "object subscript path must contain at least one indexing step",
            ));
        }
        validate_dotted_invoke_pairs(&steps)?;
        Ok(Self(steps))
    }

    pub fn single(step: ObjectSubscript) -> Self {
        Self(vec![step])
    }
    pub fn steps(&self) -> &[ObjectSubscript] {
        &self.0
    }
    pub fn into_steps(self) -> Vec<ObjectSubscript> {
        self.0
    }
}

fn validate_dotted_invoke_pairs(steps: &[ObjectSubscript]) -> Result<(), RuntimeError> {
    for (index, step) in steps.iter().enumerate() {
        let valid = match step.origin {
            ObjectSubscriptOrigin::Ordinary => true,
            ObjectSubscriptOrigin::DottedInvokeMember => steps.get(index + 1).is_some_and(|next| {
                next.origin == ObjectSubscriptOrigin::DottedInvokeArguments
                    && next.kind == ObjectIndexKind::Paren
            }),
            ObjectSubscriptOrigin::DottedInvokeArguments => {
                index.checked_sub(1).is_some_and(|prev| {
                    steps[prev].origin == ObjectSubscriptOrigin::DottedInvokeMember
                        && steps[prev].kind == ObjectIndexKind::Member
                })
            }
        };
        if !valid {
            return Err(semantic_error(
                "InvalidObjectSubscriptPath",
                "dotted invocation provenance must be a paired member and parentheses step",
            ));
        }
    }
    Ok(())
}
