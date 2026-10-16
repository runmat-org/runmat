use crate::{MirIndexing, MirLocalId, MirOperand};
use runmat_hir::{AssignmentCreationPolicy, BindingId, MemberName};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MirPlace {
    Local(MirLocalId),
    Binding(BindingId),
    Member(Box<MirPlace>, MemberName),
    DynamicMember(Box<MirPlace>, MirOperand),
    Index(Box<MirPlace>, MirIndexing),
}

impl MirPlace {
    /// Canonical outer-place creation policy for assignment MIR. A nested
    /// place's root carrier may have a different first-segment policy.
    pub fn assignment_creation_policy(&self, deletion: bool) -> AssignmentCreationPolicy {
        if deletion {
            return AssignmentCreationPolicy::ExistingOnly;
        }
        match self {
            Self::Local(_) | Self::Binding(_) => AssignmentCreationPolicy::CreateBinding,
            Self::Index(_, _) => AssignmentCreationPolicy::CreateArrayByIndex,
            Self::Member(_, _) | Self::DynamicMember(_, _) => {
                AssignmentCreationPolicy::CreateStructFieldPath
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MirIndexPlan, MirLocalId};
    use runmat_hir::{IndexKind, IndexResultContext};

    #[test]
    fn outer_creation_policy_is_distinct_from_nested_root_segment() {
        let local = MirPlace::Local(MirLocalId(0));
        let indexed = MirPlace::Index(
            Box::new(local),
            MirIndexing {
                kind: IndexKind::Paren,
                plan: MirIndexPlan::Slice,
                components: Vec::new(),
                result_context: IndexResultContext::AssignmentTarget,
                cell_expand_all: false,
            },
        );
        assert_eq!(
            indexed.assignment_creation_policy(false),
            AssignmentCreationPolicy::CreateArrayByIndex
        );
        let nested = MirPlace::Member(Box::new(indexed), MemberName("field".into()));
        assert_eq!(
            nested.assignment_creation_policy(false),
            AssignmentCreationPolicy::CreateStructFieldPath
        );
        assert_eq!(
            nested.assignment_creation_policy(true),
            AssignmentCreationPolicy::ExistingOnly
        );
    }
}
