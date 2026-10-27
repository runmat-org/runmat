use super::{ObjectIndexKind, ObjectIndexSelector};
use crate::runtime_error::semantic_error;
use crate::RuntimeError;
use runmat_value::Value;

#[derive(Debug, Clone, PartialEq)]
pub struct ObjectSubscript {
    pub(super) kind: ObjectIndexKind,
    pub(super) selector: ObjectIndexSelector,
    pub(super) origin: ObjectSubscriptOrigin,
    pub(super) semantic_fallback: Option<std::sync::Arc<runmat_types::CallableIdentity>>,
    input_callsite: Option<crate::callsite::CallsiteInfo>,
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
            semantic_fallback: None,
            input_callsite: None,
        }
    }

    pub fn braces(selector: ObjectIndexSelector) -> Self {
        Self {
            kind: ObjectIndexKind::Brace,
            selector,
            origin: ObjectSubscriptOrigin::Ordinary,
            semantic_fallback: None,
            input_callsite: None,
        }
    }

    pub fn member(name: impl Into<runmat_types::MemberName>) -> Self {
        Self::member_retained(runmat_value::Utf8Text::from(name.into().0))
    }

    pub fn member_retained(name: runmat_value::Utf8Text) -> Self {
        Self {
            kind: ObjectIndexKind::Member,
            selector: ObjectIndexSelector::Member(name),
            origin: ObjectSubscriptOrigin::Ordinary,
            semantic_fallback: None,
            input_callsite: None,
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
        Self::dotted_invoke_components_with_fallback(name, arguments, None)
    }

    pub fn dotted_invoke_components_with_fallback(
        name: impl Into<runmat_types::MemberName>,
        arguments: Vec<super::ObjectIndexComponent>,
        semantic_fallback: Option<runmat_types::CallableIdentity>,
    ) -> [Self; 2] {
        Self::dotted_invoke_retained_with_fallback(
            runmat_value::Utf8Text::from(name.into().0),
            runmat_value::ExactArrayStorage::shared_unbounded(arguments),
            semantic_fallback.map(std::sync::Arc::new),
        )
    }

    pub fn dotted_invoke_retained_with_fallback(
        name: runmat_value::Utf8Text,
        arguments: runmat_value::ExactArrayStorage<super::ObjectIndexComponent>,
        semantic_fallback: Option<std::sync::Arc<runmat_types::CallableIdentity>>,
    ) -> [Self; 2] {
        [
            Self {
                kind: ObjectIndexKind::Member,
                selector: ObjectIndexSelector::Member(name),
                origin: ObjectSubscriptOrigin::DottedInvokeMember,
                semantic_fallback,
                input_callsite: None,
            },
            Self {
                kind: ObjectIndexKind::Paren,
                selector: ObjectIndexSelector::IndexValues {
                    components: arguments,
                },
                origin: ObjectSubscriptOrigin::DottedInvokeArguments,
                semantic_fallback: None,
                input_callsite: None,
            },
        ]
    }

    pub const fn kind(&self) -> ObjectIndexKind {
        self.kind
    }
    pub fn with_input_callsite(mut self, callsite: Option<crate::callsite::CallsiteInfo>) -> Self {
        self.input_callsite = callsite;
        self
    }

    pub(crate) fn input_callsite(&self) -> Option<&crate::callsite::CallsiteInfo> {
        self.input_callsite.as_ref()
    }
    pub const fn selector(&self) -> &ObjectIndexSelector {
        &self.selector
    }
    /// Move the selector and typed fallback out after the kind and origin
    /// have been validated. Chained call arguments retain their original
    /// Value payloads instead of cloning them at this path boundary.
    pub(crate) fn into_parts(
        self,
    ) -> (
        ObjectIndexSelector,
        Option<std::sync::Arc<runmat_types::CallableIdentity>>,
    ) {
        (self.selector, self.semantic_fallback)
    }
    pub const fn origin(&self) -> ObjectSubscriptOrigin {
        self.origin
    }

    pub fn semantic_fallback(&self) -> Option<&runmat_types::CallableIdentity> {
        self.semantic_fallback.as_deref()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObjectSubscriptPath(pub(super) runmat_value::ExactArrayStorage<ObjectSubscript>);

impl ObjectSubscriptPath {
    pub fn new(steps: Vec<ObjectSubscript>) -> Result<Self, RuntimeError> {
        Self::from_storage(runmat_value::ExactArrayStorage::shared_unbounded(steps))
    }

    pub fn from_storage(
        steps: runmat_value::ExactArrayStorage<ObjectSubscript>,
    ) -> Result<Self, RuntimeError> {
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
        Self(runmat_value::ExactArrayStorage::shared_unbounded(vec![
            step,
        ]))
    }
    pub fn single_in_context(
        context: Option<&crate::context::RuntimeContext>,
        step: ObjectSubscript,
    ) -> Result<Self, RuntimeError> {
        let mut step = Some(step);
        Self::from_storage(super::array_in_context(context, 1, |_| {
            Ok(step.take().expect("one path step"))
        })?)
    }
    pub(crate) fn from_remaining_in_context(
        context: Option<&crate::context::RuntimeContext>,
        mut steps: runmat_value::ExactArrayIntoIter<ObjectSubscript>,
    ) -> Result<Self, RuntimeError> {
        Self::from_storage(super::array_in_context(context, steps.len(), |_| {
            Ok(steps.next().expect("exact path width"))
        })?)
    }
    pub fn steps(&self) -> &[ObjectSubscript] {
        &self.0
    }
    pub fn into_steps_in_context(
        self,
        context: Option<&crate::context::RuntimeContext>,
    ) -> Result<runmat_value::OwnedExactArrayValues<ObjectSubscript>, RuntimeError> {
        match self.0.try_into_owned() {
            Ok(owned) => Ok(owned),
            Err(shared) => {
                let copy = super::array_in_context(context, shared.len(), |index| {
                    Ok(shared[index].clone())
                })?;
                Ok(copy
                    .try_into_owned()
                    .expect("fresh subscript path is unique"))
            }
        }
    }

    pub fn into_steps(self) -> runmat_value::OwnedExactArrayValues<ObjectSubscript> {
        self.into_steps_in_context(None)
            .expect("ordinary path ownership")
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
