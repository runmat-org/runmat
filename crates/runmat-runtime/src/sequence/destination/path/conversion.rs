use super::{AssignmentStepSpec, SequenceEndpointSpec};
use crate::object::indexing::{ObjectIndexSelector, ObjectSubscript};

pub(super) fn object_subscript_from_step(step: AssignmentStepSpec) -> ObjectSubscript {
    match step {
        AssignmentStepSpec::Member(member) => ObjectSubscript::member(member),
        AssignmentStepSpec::Parentheses { selectors } => {
            ObjectSubscript::parentheses(ObjectIndexSelector::IndexValues {
                components: selectors,
            })
        }
        AssignmentStepSpec::Braces(components) => {
            ObjectSubscript::braces(ObjectIndexSelector::IndexValues { components })
        }
    }
}

pub(super) fn object_subscript_from_endpoint(endpoint: SequenceEndpointSpec) -> ObjectSubscript {
    match endpoint {
        SequenceEndpointSpec::Member(member) => ObjectSubscript::member(member),
        SequenceEndpointSpec::CellContents(components) => {
            ObjectSubscript::braces(ObjectIndexSelector::IndexValues { components })
        }
    }
}
