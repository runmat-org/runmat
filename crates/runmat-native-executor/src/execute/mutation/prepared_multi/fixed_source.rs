use std::sync::Arc;

use runmat_mir::{MirPreparedAssignment, MirStmtKind};
use runmat_native_codegen::{NativeFunction, NativeOperation, NativeSitePhase};
use runmat_runtime::native::NativeSiteRequest;

use crate::execute::state::HostState;
use crate::{NativeExecutorError, NativeExecutorResult};

/// Lease on the existing immutable executable metadata, not a copied MIR
/// forest. The complete site and source occurrence join the canonical owner;
/// cloning this receipt only retains that already-owned Arc allocation.
#[derive(Clone)]
pub(super) struct FixedSourceOwner {
    functions: Arc<Vec<NativeFunction>>,
    request: NativeSiteRequest,
    occurrence: runmat_types::StmtId,
    function_index: usize,
    block_index: usize,
    instruction_index: usize,
}

impl FixedSourceOwner {
    pub(super) fn capture(
        state: &HostState,
        expected: &MirPreparedAssignment,
    ) -> NativeExecutorResult<Self> {
        let request = state.current_request.ok_or_else(|| {
            NativeExecutorError::Host("fixed source lease has no current canonical site".into())
        })?;
        if request.phase != runmat_runtime::native::NativeSitePhase::STATEMENT
            || request.function != state.function.portable_id.0
            || request.reserved != 0
        {
            return Err(NativeExecutorError::Host(
                "fixed source lease requires the exact statement site in the active function"
                    .into(),
            ));
        }
        let unavailable = || {
            NativeExecutorError::Host(
                "fixed source lease has no exact immutable statement coordinate".into(),
            )
        };
        // Source IDs are not vector offsets. Resolve once against immutable
        // metadata; subsequent phase borrows use these validated coordinates.
        let function_index = state
            .functions
            .iter()
            .position(|function| function.portable_id.0 == request.function)
            .ok_or_else(unavailable)?;
        let function = &state.functions[function_index];
        let block_index = function
            .blocks
            .iter()
            .position(|block| block.id.0 == request.block)
            .ok_or_else(unavailable)?;
        let instruction_index = function.blocks[block_index]
            .instructions
            .iter()
            .position(|instruction| {
                instruction.site.point.position == request.position
                    && instruction.site.point.block == request.block
                    && instruction.site.ordinal == request.ordinal
                    && instruction.site.phase == NativeSitePhase::Statement
            })
            .ok_or_else(unavailable)?;
        let owner = Self {
            functions: Arc::clone(&state.functions),
            request,
            occurrence: expected.source_statement,
            function_index,
            block_index,
            instruction_index,
        };
        if !owner.matches(expected) {
            return Err(NativeExecutorError::Host(
                "fixed source lease disagrees with immutable executable metadata".into(),
            ));
        }
        Ok(owner)
    }

    pub(super) fn descriptor(&self) -> NativeExecutorResult<&MirPreparedAssignment> {
        self.functions
            .get(self.function_index)
            .filter(|function| function.portable_id.0 == self.request.function)
            .and_then(|function| {
                function
                    .blocks
                    .get(self.block_index)
                    .filter(|block| block.id.0 == self.request.block)
            })
            .and_then(|block| {
                block
                    .instructions
                    .get(self.instruction_index)
                    .filter(|instruction| {
                        instruction.site.point.position == self.request.position
                            && instruction.site.point.block == self.request.block
                            && instruction.site.ordinal == self.request.ordinal
                            && instruction.site.phase == NativeSitePhase::Statement
                    })
            })
            .and_then(|instruction| match &instruction.operation {
                NativeOperation::Statement(MirStmtKind::PrepareFixedAssignment(owner))
                    if owner.assignment.source_statement == self.occurrence
                        && matches!(
                            owner.phase,
                            runmat_mir::MirFixedPreparationPhase::CaptureReceiver
                        ) =>
                {
                    Some(&owner.assignment)
                }
                _ => None,
            })
            .ok_or_else(|| {
                NativeExecutorError::Host("retained fixed source site is unavailable".into())
            })
    }

    pub(super) fn matches(&self, expected: &MirPreparedAssignment) -> bool {
        self.descriptor().is_ok_and(|owner| owner == expected)
    }
}
