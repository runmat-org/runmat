use crate::{MirIndexing, MirLocalId, MirOperand};
use runmat_hir::{AssignmentCreationPolicy, BindingId, IndexKind, MemberName};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MirPlace {
    Local(MirLocalId),
    Binding(BindingId),
    Member(Box<MirPlace>, MemberName),
    DynamicMember(Box<MirPlace>, MirOperand),
    Index(Box<MirPlace>, MirIndexing),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MirRootStep {
    Member,
    Index(IndexKind),
}

impl MirPlace {
    /// First operation rooted in the binding, not the outermost operation
    /// of a nested assignment path.
    pub fn first_root_step(&self) -> Option<MirRootStep> {
        match self {
            Self::Local(_) | Self::Binding(_) => None,
            Self::Index(base, indexing) => base
                .first_root_step()
                .or(Some(MirRootStep::Index(indexing.kind))),
            Self::Member(base, _) | Self::DynamicMember(base, _) => {
                base.first_root_step().or(Some(MirRootStep::Member))
            }
        }
    }

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
            nested.first_root_step(),
            Some(MirRootStep::Index(IndexKind::Paren))
        );
        assert_eq!(
            nested.assignment_creation_policy(false),
            AssignmentCreationPolicy::CreateStructFieldPath
        );
        assert_eq!(
            nested.assignment_creation_policy(true),
            AssignmentCreationPolicy::ExistingOnly
        );

        let paren_indexing = MirIndexing {
            kind: IndexKind::Paren,
            plan: MirIndexPlan::Slice,
            components: Vec::new(),
            result_context: IndexResultContext::AssignmentTarget,
            cell_expand_all: false,
        };
        let brace_indexing = MirIndexing {
            kind: IndexKind::Brace,
            ..paren_indexing.clone()
        };
        let brace = MirPlace::Index(
            Box::new(MirPlace::Local(MirLocalId(0))),
            brace_indexing.clone(),
        );
        assert_eq!(
            MirPlace::Member(Box::new(brace.clone()), MemberName("field".into())).first_root_step(),
            Some(MirRootStep::Index(IndexKind::Brace))
        );
        let paren = MirPlace::Index(
            Box::new(MirPlace::Local(MirLocalId(0))),
            paren_indexing.clone(),
        );
        let nested_brace = MirPlace::Index(Box::new(paren), brace_indexing);
        assert_eq!(
            nested_brace.first_root_step(),
            Some(MirRootStep::Index(IndexKind::Paren))
        );
        let nested_paren = MirPlace::Index(Box::new(brace), paren_indexing);
        assert_eq!(
            nested_paren.first_root_step(),
            Some(MirRootStep::Index(IndexKind::Brace))
        );
    }
}
