//! Exact native occurrences for the existing canonical numerical host bodies.
//! This owner grants neither VM registration nor VM completion authority.
use super::{RuntimeContext, VmFusionHostInstruction, VmFusionHostOperation, VmFusionHostOutputs};
use crate::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::native::NativeSiteRequest;
use crate::numeric_region::service::NativeNumericRegionSelectionAuthority;
use crate::RuntimeError;
use runmat_execution::resource::{ExactShared, RetainedHostAllocationClaim};
use std::{cell::RefCell, sync::Arc};

/// Captured from the verified native operation and canonical fusion binding.
#[derive(Clone, Copy)]
pub struct NativeRegionHostOperation {
    pub site: NativeSiteRequest,
    pub operator: runmat_types::OperatorKind,
    pub operation: VmFusionHostOperation,
    /// Original region-input positions; computed operands remain ordinary
    /// native values. No public local or workspace binding is replaced.
    pub input_substitutions: [Option<usize>; 2],
}

// This is a layout assertion, not an additional program eligibility cap:
// MAX_NODES is the existing verified numeric program limit. Keep the largest
// local transcript array below 32 KiB on every supported target.
const _: () = assert!(
    std::mem::size_of::<[Option<NativeRegionHostOperation>; crate::numeric_region::MAX_NODES]>()
        <= 32 * 1024
);

struct State {
    outputs: Option<VmFusionHostOutputs>,
    next: usize,
    pending: Option<(usize, bool)>,
    invalid: bool,
    retirement_active: bool,
}
struct Backing {
    authority: NativeNumericRegionSelectionAuthority,
    operations: [Option<NativeRegionHostOperation>; crate::numeric_region::MAX_NODES],
    len: usize,
    state: RefCell<State>,
    // ExactShared releases its allocation before dropping this payload.
    _claim: Option<Arc<dyn RetainedHostAllocationClaim>>,
}

#[derive(Clone)]
pub struct NativeRegionHostRoute(ExactShared<Backing>);

fn allocation_error(runtime: &RuntimeContext, source: HostAllocationError) -> RuntimeError {
    host_allocation_runtime_error(
        Some(runtime),
        "RunMat:Placement:NativeHostRegion",
        "native host region",
        source,
    )
}
fn invalid() -> RuntimeError {
    crate::build_runtime_error("native host region occurrence or output authority changed")
        .with_identifier("RunMat:Placement:InvalidDecision")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}

impl NativeRegionHostRoute {
    pub fn retained_control_bytes() -> Option<u64> {
        let bytes = u64::try_from(ExactShared::<Backing>::requested_bytes()).ok()?;
        HostAllocationPlan::retained_metadata_bytes(bytes).ok()
    }
    /// The caller has the admitted native program's nonpublication authority,
    /// its exact native site transcript, and already-funded physical outputs.
    pub fn prepare(
        runtime: &RuntimeContext,
        authority: NativeNumericRegionSelectionAuthority,
        len: usize,
        operations: impl Iterator<Item = NativeRegionHostOperation>,
        outputs: Option<VmFusionHostOutputs>,
    ) -> Result<Self, RuntimeError> {
        if len == 0 || len > crate::numeric_region::MAX_NODES || !authority.is_current(runtime) {
            return Err(invalid());
        }
        let bytes = u64::try_from(ExactShared::<Backing>::requested_bytes())
            .map_err(|_| allocation_error(runtime, HostAllocationError::Overflow))?;
        let retained = HostAllocationPlan::retained_metadata_bytes(bytes)
            .map_err(|error| allocation_error(runtime, error))?;
        let receipt = HostAllocationReceipt::admit(
            Some(runtime),
            HostAllocationPlan::new(0, retained)
                .map_err(|error| allocation_error(runtime, error))?,
        )
        .map_err(|error| allocation_error(runtime, error))?;
        let claim = receipt
            .as_ref()
            .map(|receipt| receipt.claim_retained_metadata(runtime, bytes))
            .transpose()
            .map_err(|error| allocation_error(runtime, error))?;
        let mut slots = [None; crate::numeric_region::MAX_NODES];
        let mut captured = 0;
        for operation in operations {
            let Some(slot) = slots.get_mut(captured).filter(|_| captured < len) else {
                return Err(invalid());
            };
            if operation.site.function != authority.identity().1.function {
                return Err(invalid());
            }
            *slot = Some(operation);
            captured += 1;
        }
        if captured != len {
            return Err(invalid());
        }
        let backing = Backing {
            authority,
            operations: slots,
            len,
            state: RefCell::new(State {
                outputs,
                next: 0,
                pending: None,
                invalid: false,
                retirement_active: false,
            }),
            _claim: claim,
        };
        ExactShared::try_new(backing).map(Self).map_err(|_| {
            allocation_error(
                runtime,
                HostAllocationError::Numeric(
                    runmat_value::ExactHostNumericAllocationError::AllocationFailed,
                ),
            )
        })
    }

    pub fn begin(
        &self,
        runtime: &RuntimeContext,
        site: NativeSiteRequest,
        operator: runmat_types::OperatorKind,
    ) -> Result<(), RuntimeError> {
        if !self.0.get().authority.is_current(runtime) {
            return Err(invalid());
        }
        let mut state = self.0.get().state.borrow_mut();
        let valid = !state.invalid
            && !state.retirement_active
            && self
                .0
                .get()
                .operations
                .get(state.next)
                .and_then(Option::as_ref)
                .is_some_and(|operation| operation.site == site && operation.operator == operator)
            && state.pending.is_none_or(|(index, _)| index == state.next);
        if !valid {
            state.invalid = true;
            return Err(invalid());
        }
        if state.pending.is_none() {
            state.pending = Some((state.next, false));
        }
        Ok(())
    }

    pub fn finish(&self, site: NativeSiteRequest) -> Result<(), RuntimeError> {
        let mut state = self.0.get().state.borrow_mut();
        let Some((index, true)) = state.pending else {
            state.invalid = true;
            return Err(invalid());
        };
        if self.0.get().operations[index].is_none_or(|operation| operation.site != site) {
            state.invalid = true;
            return Err(invalid());
        }
        state.pending = None;
        state.next += 1;
        Ok(())
    }

    pub fn pending_site(&self) -> Option<NativeSiteRequest> {
        let state = self.0.get().state.borrow();
        state
            .pending
            .and_then(|(index, _)| self.0.get().operations[index])
            .map(|operation| operation.site)
    }

    pub fn finish_if_pending(&self, site: NativeSiteRequest) -> Result<(), RuntimeError> {
        let expected = {
            let state = self.0.get().state.borrow();
            self.0
                .get()
                .operations
                .get(state.next)
                .copied()
                .flatten()
                .map(|operation| operation.site)
        };
        if expected == Some(site) {
            return self.finish(site);
        }
        match self.pending_site() {
            Some(pending) if pending == site => self.finish(site),
            Some(_) => Err(invalid()),
            None => Ok(()),
        }
    }

    pub fn input_substitution(&self, site: NativeSiteRequest, operand: usize) -> Option<usize> {
        let state = self.0.get().state.borrow();
        if state.invalid || state.retirement_active {
            return None;
        }
        let (index, _) = state.pending?;
        let operation = self.0.get().operations[index]?;
        (operation.site == site)
            .then(|| {
                operation
                    .input_substitutions
                    .get(operand)
                    .copied()
                    .flatten()
            })
            .flatten()
    }

    fn selected(
        &self,
        runtime: &RuntimeContext,
        instruction: VmFusionHostInstruction,
        operation: Option<VmFusionHostOperation>,
    ) -> bool {
        if !self.0.get().authority.is_current(runtime) {
            return false;
        }
        let state = self.0.get().state.borrow();
        let Some((index, false)) = state.pending else {
            return false;
        };
        let Some(expected) = self.0.get().operations[index] else {
            return false;
        };
        let negation_body = expected.operation
            == VmFusionHostOperation::Intrinsic(runmat_types::OperatorKind::UnaryMinus)
            && operation
                == Some(VmFusionHostOperation::Builtin(
                    runmat_builtins::BuiltinBindingIdentity {
                        builtin: runmat_builtins::TIMES_CATALOG_ENTRY.identity,
                        variant: "default",
                    },
                ))
            && instruction == VmFusionHostInstruction::ElementwiseBinary;
        !state.invalid
            && !state.retirement_active
            && (negation_body
                || (operation == Some(expected.operation)
                    && match expected.operation {
                        VmFusionHostOperation::Builtin(_) => {
                            instruction == VmFusionHostInstruction::ElementwiseBinary
                        }
                        VmFusionHostOperation::Intrinsic(operator) => {
                            instruction == VmFusionHostInstruction::Intrinsic(operator)
                        }
                    }))
    }

    pub(crate) fn allows_host_preparation(
        &self,
        runtime: &RuntimeContext,
        instruction: VmFusionHostInstruction,
    ) -> bool {
        self.selected(
            runtime,
            instruction,
            runtime
                .vm_fusion_selected_binding()
                .map(VmFusionHostOperation::Builtin),
        )
    }

    pub(crate) fn require_binary_host(
        &self,
        runtime: &RuntimeContext,
        operator: runmat_types::OperatorKind,
        binding: runmat_builtins::BuiltinBindingIdentity,
    ) -> Result<(), RuntimeError> {
        if !self.selected(
            runtime,
            VmFusionHostInstruction::ElementwiseBinary,
            Some(VmFusionHostOperation::Builtin(binding)),
        ) {
            return Err(invalid());
        }
        let state = self.0.get().state.borrow();
        let expected = state
            .pending
            .and_then(|(index, _)| self.0.get().operations[index])
            .ok_or_else(invalid)?;
        if expected.operator == operator
            || (expected.operator == runmat_types::OperatorKind::UnaryMinus
                && operator == runmat_types::OperatorKind::ElementwiseMultiply)
        {
            Ok(())
        } else {
            Err(invalid())
        }
    }
    pub(crate) fn require_intrinsic_host(
        &self,
        runtime: &RuntimeContext,
        operator: runmat_types::OperatorKind,
    ) -> Result<(), RuntimeError> {
        self.allows_intrinsic_host(runtime, operator)
            .then_some(())
            .ok_or_else(invalid)
    }
    pub(crate) fn allows_intrinsic_host(
        &self,
        runtime: &RuntimeContext,
        operator: runmat_types::OperatorKind,
    ) -> bool {
        self.selected(
            runtime,
            VmFusionHostInstruction::Intrinsic(operator),
            Some(VmFusionHostOperation::Intrinsic(operator)),
        )
    }

    pub(crate) fn take_numerical_output(
        &self,
        runtime: &RuntimeContext,
        instruction: VmFusionHostInstruction,
        binding: runmat_builtins::BuiltinBindingIdentity,
    ) -> Result<Option<super::preparation::PreparedHostNumericalOutput>, RuntimeError> {
        if !self.selected(
            runtime,
            instruction,
            Some(VmFusionHostOperation::Builtin(binding)),
        ) {
            return Err(invalid());
        }
        let intrinsic_negation = {
            let state = self.0.get().state.borrow();
            state
                .pending
                .and_then(|(index, _)| self.0.get().operations[index])
                .is_some_and(|operation| {
                    operation.operation
                        == VmFusionHostOperation::Intrinsic(runmat_types::OperatorKind::UnaryMinus)
                })
        };
        if intrinsic_negation {
            return self.take_output(runtime, |outputs| {
                outputs.take_intrinsic(
                    runtime,
                    VmFusionHostInstruction::Intrinsic(runmat_types::OperatorKind::UnaryMinus),
                    runmat_types::OperatorKind::UnaryMinus,
                )
            });
        }
        self.take_output(runtime, |outputs| {
            outputs.take_builtin(runtime, instruction, binding)
        })
    }
    pub(crate) fn take_intrinsic_numerical_output(
        &self,
        runtime: &RuntimeContext,
        operator: runmat_types::OperatorKind,
    ) -> Result<Option<super::preparation::PreparedHostNumericalOutput>, RuntimeError> {
        if !self.allows_intrinsic_host(runtime, operator) {
            return Err(invalid());
        }
        self.take_output(runtime, |outputs| {
            outputs.take_intrinsic(
                runtime,
                VmFusionHostInstruction::Intrinsic(operator),
                operator,
            )
        })
    }
    fn take_output(
        &self,
        runtime: &RuntimeContext,
        take: impl FnOnce(
            &mut VmFusionHostOutputs,
        )
            -> Result<super::preparation::PreparedHostNumericalOutput, RuntimeError>,
    ) -> Result<Option<super::preparation::PreparedHostNumericalOutput>, RuntimeError> {
        // Move the physical carrier out before any output construction/drop
        // can reenter Runtime; a missing carrier cannot authorize a sibling.
        let outputs = {
            let mut state = self.0.get().state.borrow_mut();
            if state.retirement_active || state.invalid {
                return Err(invalid());
            }
            state.retirement_active = true;
            state.outputs.take()
        };
        let Some(mut outputs) = outputs else {
            self.0.get().state.borrow_mut().retirement_active = false;
            return match super::host_allocation::HostAllocationReceipt::policy(Some(runtime)) {
                Ok(super::host_allocation::HostAllocationPolicy::Unbounded) => Ok(None),
                _ => Err(invalid()),
            };
        };
        let result = take(&mut outputs);
        let mut state = self.0.get().state.borrow_mut();
        state.retirement_active = false;
        if result.is_err() {
            state.invalid = true;
        }
        state.outputs = Some(outputs);
        result.map(Some)
    }
    pub(crate) fn confirm_host_branch(
        &self,
        runtime: &RuntimeContext,
        instruction: VmFusionHostInstruction,
    ) {
        let valid = self.allows_host_preparation(runtime, instruction);
        self.confirm(valid);
    }
    pub(crate) fn confirm_intrinsic_host(
        &self,
        runtime: &RuntimeContext,
        operator: runmat_types::OperatorKind,
    ) {
        self.confirm(self.allows_intrinsic_host(runtime, operator));
    }
    fn confirm(&self, valid: bool) {
        let mut state = self.0.get().state.borrow_mut();
        match state.pending.as_mut() {
            Some((_, confirmed)) if valid => *confirmed = true,
            _ => state.invalid = true,
        }
    }
    pub fn complete(&self, runtime: &RuntimeContext) -> bool {
        if !self.0.get().authority.is_current(runtime) {
            return false;
        }
        let state = self.0.get().state.borrow();
        !state.invalid
            && !state.retirement_active
            && state.pending.is_none()
            && state.next == self.0.get().len
            && state
                .outputs
                .as_ref()
                .is_none_or(VmFusionHostOutputs::complete)
    }
}
