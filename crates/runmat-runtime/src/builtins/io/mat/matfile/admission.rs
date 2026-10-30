//! Typed, bounded identity for the parsed suffix of a MAT-file assignment.
//! Unsupported selector payloads remain executable in unbounded contexts,
//! but cannot claim an exact finite placement signature before effects.

use crate::object::indexing::{
    ObjectIndexComponent, ObjectIndexKind, ObjectIndexSelector, ObjectSubscriptOrigin,
    ObjectSubscriptPath,
};
use runmat_execution::Digest;
use runmat_value::{IntValue, Value};

const MAX_STEPS: usize = 128;
const MAX_COMPONENTS: usize = 4096;
const MAX_TEXT_BYTES: usize = 64 * 1024;

#[derive(serde::Serialize)]
enum StepKind {
    Member,
    Parentheses,
    Braces,
}

#[derive(serde::Serialize)]
enum Origin {
    Ordinary,
    DottedMember,
    DottedArguments,
}

#[derive(serde::Serialize)]
enum Selector<'a> {
    Member(&'a str),
    Indices(&'a [usize]),
    Components(Vec<Component<'a>>),
}

#[derive(serde::Serialize)]
enum Component<'a> {
    Colon,
    DoubleBits(u64),
    Integer(&'a IntValue),
    Logical(bool),
    Text(&'a str),
    Resident(runmat_accelerate_api::GpuTensorWireDescriptor),
}

#[derive(serde::Serialize)]
struct Step<'a> {
    kind: StepKind,
    origin: Origin,
    selector: Selector<'a>,
}

pub(super) fn suffix_digest(path: &ObjectSubscriptPath) -> Option<Digest> {
    let suffix = path.steps().get(1..)?;
    if suffix.len() > MAX_STEPS {
        return None;
    }
    let mut facts = Vec::new();
    facts.try_reserve(suffix.len()).ok()?;
    let mut components = 0usize;
    let mut text_bytes = 0usize;
    for step in suffix {
        let kind = match step.kind() {
            ObjectIndexKind::Member => StepKind::Member,
            ObjectIndexKind::Paren => StepKind::Parentheses,
            ObjectIndexKind::Brace => StepKind::Braces,
        };
        let origin = match step.origin() {
            ObjectSubscriptOrigin::Ordinary => Origin::Ordinary,
            ObjectSubscriptOrigin::DottedInvokeMember => Origin::DottedMember,
            ObjectSubscriptOrigin::DottedInvokeArguments => Origin::DottedArguments,
        };
        let selector = match step.selector() {
            ObjectIndexSelector::Member(name) => {
                charge_text(&mut text_bytes, name.as_str())?;
                Selector::Member(name.as_str())
            }
            ObjectIndexSelector::ScalarIndices { indices } => {
                charge_count(&mut components, indices.len())?;
                Selector::Indices(indices)
            }
            ObjectIndexSelector::IndexValues { components: values } => {
                charge_count(&mut components, values.len())?;
                let mut projected = Vec::new();
                projected.try_reserve(values.len()).ok()?;
                for value in values {
                    projected.push(match value {
                        ObjectIndexComponent::Colon => Component::Colon,
                        ObjectIndexComponent::Value(Value::Num(number)) => {
                            Component::DoubleBits(number.to_bits())
                        }
                        ObjectIndexComponent::Value(Value::Int(integer)) => {
                            Component::Integer(integer)
                        }
                        ObjectIndexComponent::Value(Value::Bool(logical)) => {
                            Component::Logical(*logical)
                        }
                        ObjectIndexComponent::Value(Value::String(
                            runmat_value::StringElement::Text(text),
                        )) => {
                            charge_text(&mut text_bytes, text)?;
                            Component::Text(text)
                        }
                        ObjectIndexComponent::Value(Value::GpuTensor(handle))
                            if handle.shape().len() <= 64 =>
                        {
                            Component::Resident(handle.wire_descriptor())
                        }
                        _ => return None,
                    });
                }
                Selector::Components(projected)
            }
        };
        facts.push(Step {
            kind,
            origin,
            selector,
        });
    }
    Some(Digest::sha256(serde_json::to_vec(&facts).ok()?))
}

fn charge_count(total: &mut usize, count: usize) -> Option<()> {
    *total = total.checked_add(count)?;
    (*total <= MAX_COMPONENTS).then_some(())
}

fn charge_text(total: &mut usize, text: &str) -> Option<()> {
    *total = total.checked_add(text.len())?;
    (*total <= MAX_TEXT_BYTES).then_some(())
}
