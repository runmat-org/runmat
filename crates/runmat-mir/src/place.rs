use crate::{MirIndexing, MirLocalId, MirOperand};
use runmat_types::{AssignmentCreationPolicy, BindingId, IndexKind, MemberName};
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
    /// Private selector regions, in root-to-leaf evaluation order. The caller
    /// owns the phase in which the destination is prepared.
    pub fn visit_selector_regions(
        &self,
        mut visitor: impl FnMut(crate::MirExpressionRegionRef<'_>),
    ) {
        fn visit(place: &MirPlace, visitor: &mut dyn FnMut(crate::MirExpressionRegionRef<'_>)) {
            match place {
                MirPlace::Local(_) | MirPlace::Binding(_) => {}
                MirPlace::Member(base, _) | MirPlace::DynamicMember(base, _) => {
                    visit(base, visitor)
                }
                MirPlace::Index(base, indexing) => {
                    visit(base, visitor);
                    indexing.visit_direct_expression_regions_dyn(visitor);
                }
            }
        }
        visit(self, &mut visitor);
    }

    pub fn has_expanding_selectors(&self) -> bool {
        match self {
            Self::Local(_) | Self::Binding(_) => false,
            Self::Member(base, _) | Self::DynamicMember(base, _) => base.has_expanding_selectors(),
            Self::Index(base, indexing) => {
                base.has_expanding_selectors()
                    || indexing.components.iter().any(|component| {
                        matches!(component, crate::MirIndexComponent::Expansion(_))
                    })
            }
        }
    }
    /// Member paths crossing an indexed receiver need one captured destination
    /// even when every selector has statically scalar width.
    pub fn has_indexed_member_path(&self) -> bool {
        fn kinds(place: &MirPlace) -> (bool, bool) {
            match place {
                MirPlace::Local(_) | MirPlace::Binding(_) => (false, false),
                MirPlace::Member(base, _) | MirPlace::DynamicMember(base, _) => {
                    let (_, index) = kinds(base);
                    (true, index)
                }
                MirPlace::Index(base, _) => {
                    let (member, _) = kinds(base);
                    (member, true)
                }
            }
        }
        let (member, index) = kinds(self);
        member && index
    }
    pub fn requires_prepared_assignment(&self) -> bool {
        self.has_expanding_selectors() || self.assignment_path_len() > 1
    }

    fn assignment_path_len(&self) -> usize {
        match self {
            Self::Local(_) | Self::Binding(_) => 0,
            Self::Member(base, _) | Self::DynamicMember(base, _) | Self::Index(base, _) => {
                base.assignment_path_len() + 1
            }
        }
    }

    fn assignment_path_step(&self, index: usize) -> Option<&Self> {
        match self {
            Self::Local(_) | Self::Binding(_) => None,
            Self::Member(base, _) | Self::DynamicMember(base, _) | Self::Index(base, _) => {
                let prefix = base.assignment_path_len();
                if index == prefix {
                    Some(self)
                } else {
                    base.assignment_path_step(index)
                }
            }
        }
    }

    /// Derive a missing receiver's exact container from the unevaluated suffix.
    /// Parentheses preserve a container; leading parentheses before a member
    /// require an empty structure array, not a scalar that would make end=1.
    pub fn assignment_child_creation(
        &self,
        first_step: usize,
        creation: AssignmentCreationPolicy,
    ) -> runmat_types::AssignmentChildCreation {
        use runmat_types::AssignmentChildCreation as Child;
        if creation == AssignmentCreationPolicy::ExistingOnly {
            return Child::ExistingOnly;
        }
        let mut parentheses = false;
        for index in first_step..self.assignment_path_len() {
            match self.assignment_path_step(index) {
                Some(Self::Member(..) | Self::DynamicMember(..)) => {
                    return if parentheses {
                        Child::StructureArray
                    } else {
                        Child::ScalarStructure
                    };
                }
                Some(Self::Index(_, indexing)) if indexing.kind == IndexKind::Brace => {
                    return Child::CellArray;
                }
                Some(Self::Index(..)) => parentheses = true,
                _ => unreachable!("bounded assignment path step"),
            }
        }
        if parentheses {
            Child::NumericArray
        } else {
            Child::ExistingOnly
        }
    }
    pub fn root_local(&self) -> Option<MirLocalId> {
        match self {
            Self::Local(local) => Some(*local),
            Self::Binding(_) => None,
            Self::Member(base, _) | Self::DynamicMember(base, _) | Self::Index(base, _) => {
                base.root_local()
            }
        }
    }

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

    /// Parenthesis projections preserve their container. The first member or
    /// brace operation following those projections determines an absent root's
    /// carrier, without granting conversion of any existing receiver.
    pub fn assignment_root_creation(
        &self,
        creation: AssignmentCreationPolicy,
    ) -> crate::SequenceRootCreation {
        match self.assignment_child_creation(0, creation) {
            runmat_types::AssignmentChildCreation::ExistingOnly => {
                crate::SequenceRootCreation::Ordinary(AssignmentCreationPolicy::ExistingOnly)
            }
            child => crate::SequenceRootCreation::OrdinaryContainer(child),
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
    use runmat_types::{IndexKind, IndexResultContext};

    #[test]
    fn outer_creation_policy_is_distinct_from_nested_root_segment() {
        let local = MirPlace::Local(MirLocalId(0));
        let indexed = MirPlace::Index(
            Box::new(local),
            MirIndexing {
                argument_spans: None,
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
            nested.assignment_root_creation(nested.assignment_creation_policy(false)),
            crate::SequenceRootCreation::OrdinaryContainer(
                runmat_types::AssignmentChildCreation::StructureArray
            )
        );
        assert_eq!(
            nested.assignment_root_creation(AssignmentCreationPolicy::ExistingOnly),
            crate::SequenceRootCreation::Ordinary(AssignmentCreationPolicy::ExistingOnly)
        );
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
            argument_spans: None,
            kind: IndexKind::Paren,
            plan: MirIndexPlan::Slice,
            components: Vec::new(),
            result_context: IndexResultContext::AssignmentTarget,
            cell_expand_all: false,
        };
        let brace_indexing = MirIndexing {
            argument_spans: None,
            kind: IndexKind::Brace,
            ..paren_indexing.clone()
        };
        let brace = MirPlace::Index(
            Box::new(MirPlace::Local(MirLocalId(0))),
            brace_indexing.clone(),
        );
        assert_eq!(
            MirPlace::Member(Box::new(brace.clone()), MemberName("field".into()))
                .assignment_root_creation(AssignmentCreationPolicy::CreateStructFieldPath),
            crate::SequenceRootCreation::OrdinaryContainer(
                runmat_types::AssignmentChildCreation::CellArray
            )
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
        use runmat_types::AssignmentChildCreation as Child;
        let scalar = MirPlace::Member(
            Box::new(MirPlace::Local(MirLocalId(0))),
            MemberName("child".into()),
        );
        assert_eq!(
            scalar.assignment_child_creation(0, AssignmentCreationPolicy::CreateStructFieldPath),
            Child::ScalarStructure
        );
        assert_eq!(
            nested.assignment_child_creation(1, AssignmentCreationPolicy::CreateStructFieldPath),
            Child::ScalarStructure
        );
        assert_eq!(
            nested.assignment_child_creation(0, AssignmentCreationPolicy::ExistingOnly),
            Child::ExistingOnly
        );
        assert_eq!(
            nested_paren.assignment_child_creation(1, AssignmentCreationPolicy::CreateArrayByIndex),
            Child::NumericArray
        );
        assert!(nested_paren.requires_prepared_assignment());
    }
}
