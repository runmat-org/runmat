//! Assignment receiver paths retain assignment contexts even when their
//! selectors have already been captured by phased preparation.
use crate::{MirPlace, MirPreparedAssignment};
use runmat_types::{IndexResultContext, PlaceMutationKind};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MirAssignmentIndexRole {
    Receiver,
    Target,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MirAssignmentIndexContextError {
    pub role: MirAssignmentIndexRole,
    pub expected: IndexResultContext,
    pub actual: IndexResultContext,
}

impl std::fmt::Display for MirAssignmentIndexContextError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "MIR assignment {:?} index requires {:?} context, received {:?}",
            self.role, self.expected, self.actual
        )
    }
}
impl std::error::Error for MirAssignmentIndexContextError {}

impl MirAssignmentIndexContextError {
    pub fn identifier(self) -> &'static str {
        if self.expected == IndexResultContext::DeletionTarget {
            "RunMat:MirDeleteAssignmentContextInvalid"
        } else if self.role == MirAssignmentIndexRole::Target
            && self.actual == IndexResultContext::DeletionTarget
        {
            "RunMat:MirDeletionContextWithoutDeleteInvalid"
        } else {
            "RunMat:MirIndexContextInvalid"
        }
    }
}

impl crate::MirBody {
    pub fn validate_prepared_assignment_index_contexts(
        &self,
    ) -> Result<(), MirAssignmentIndexContextError> {
        for statement in self.blocks.iter().flat_map(|block| &block.statements) {
            let assignment = match &statement.kind {
                crate::MirStmtKind::PrepareFixedAssignment(phase) => &phase.assignment,
                crate::MirStmtKind::PreparedAssign { preparation, .. } => preparation,
                _ => continue,
            };
            assignment.validate_index_contexts()?;
        }
        Ok(())
    }
}

impl MirPreparedAssignment {
    pub fn validate_index_contexts(&self) -> Result<(), MirAssignmentIndexContextError> {
        let mut place = &self.mutation.place;
        let mut role = MirAssignmentIndexRole::Target;
        loop {
            let base = match place {
                MirPlace::Local(_) | MirPlace::Binding(_) => return Ok(()),
                MirPlace::Member(base, _) | MirPlace::DynamicMember(base, _) => base,
                MirPlace::Index(base, indexing) => {
                    let expected = if role == MirAssignmentIndexRole::Target
                        && self.mutation.kind == PlaceMutationKind::Delete
                    {
                        IndexResultContext::DeletionTarget
                    } else {
                        IndexResultContext::AssignmentTarget
                    };
                    // A delete target can retain its delete context through
                    // a receiver read; ordinary assignment cannot acquire it.
                    let delete_receiver = role == MirAssignmentIndexRole::Receiver
                        && self.mutation.kind == PlaceMutationKind::Delete
                        && indexing.result_context == IndexResultContext::DeletionTarget;
                    if indexing.result_context != expected && !delete_receiver {
                        return Err(MirAssignmentIndexContextError {
                            role,
                            expected,
                            actual: indexing.result_context,
                        });
                    }
                    base
                }
            };
            place = base;
            role = MirAssignmentIndexRole::Receiver;
        }
    }
}
