//! One immutable compiler product shared with queued child VM work.
//!
//! Promotion moves the existing bytecode forest into an exactly claimed
//! control block. Instructions and compiler metadata are not copied; runtime
//! Values remain in their independently rooted interpreter frame.

use std::ops::Deref;

use runmat_execution::resource::ExactShared;
use runmat_runtime::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use runmat_runtime::context::RuntimeContext;
use runmat_runtime::RuntimeError;
use runmat_value::{ExactHostAllocationClaim, ExactHostNumericAllocationError};

use crate::bytecode::Bytecode;

#[derive(Debug)]
struct SharedBytecode {
    bytecode: Option<Bytecode>,
    display_name: String,
    _claim: Option<ExactHostAllocationClaim>,
}

#[derive(Clone, Debug)]
pub struct BytecodeLease(ExactShared<SharedBytecode>);

impl Deref for BytecodeLease {
    type Target = Bytecode;

    fn deref(&self) -> &Self::Target {
        self.0
            .get()
            .bytecode
            .as_ref()
            .expect("retained compiler product published before use")
    }
}

impl BytecodeLease {
    pub fn display_name(&self) -> &str {
        &self.0.get().display_name
    }
}

/// Exactly one of `direct` and `shared` owns the compiler product. Keeping the
/// ordinary product inline avoids allocating a box for every interpreter;
/// promotion moves it into the separately admitted shared generation.
#[derive(Debug)]
pub struct BytecodeOwner {
    direct: Option<Bytecode>,
    shared: Option<BytecodeLease>,
    display_name: String,
}

impl BytecodeOwner {
    pub fn direct(bytecode: Bytecode, display_name: String) -> Self {
        Self {
            direct: Some(bytecode),
            shared: None,
            display_name,
        }
    }

    pub fn shared(lease: BytecodeLease) -> Self {
        Self {
            direct: None,
            shared: Some(lease),
            display_name: String::new(),
        }
    }

    pub fn display_name(&self) -> &str {
        self.shared
            .as_ref()
            .map_or(self.display_name.as_str(), BytecodeLease::display_name)
    }

    pub fn shared_lease(&self) -> Option<BytecodeLease> {
        self.shared.clone()
    }

    pub fn promote_shared(&mut self, context: &RuntimeContext) -> Result<(), RuntimeError> {
        if self.shared.is_some() {
            return Ok(());
        }
        debug_assert!(
            self.direct.is_some(),
            "direct compiler product must be present"
        );
        let bytes = u64::try_from(ExactShared::<SharedBytecode>::requested_bytes())
            .map_err(|_| allocation_error(context, HostAllocationError::Overflow))?;
        // The new shared control block is the only new physical backing.
        // The preexisting bytecode and display-name allocations are moved,
        // not retroactively certified by this claim.
        let retained = HostAllocationPlan::retained_exact_output_bytes(bytes, 1)
            .map_err(|error| allocation_error(context, error))?;
        let plan = HostAllocationPlan::new(0, retained)
            .map_err(|error| allocation_error(context, error))?;
        let receipt = HostAllocationReceipt::admit(Some(context), plan)
            .map_err(|error| allocation_error(context, error))?;
        // Allocate the control block before moving the original Bytecode, so
        // any allocator failure leaves the sole compiler product untouched.
        let mut shared = match &receipt {
            Some(receipt) => receipt.with_output_allocation(context, |permit| {
                let claim =
                    permit.reserve_exact_claim(ExactShared::<SharedBytecode>::requested_bytes())?;
                ExactShared::try_new(SharedBytecode {
                    bytecode: None,
                    display_name: String::new(),
                    _claim: Some(claim),
                })
                .map_err(|_| ExactHostNumericAllocationError::AllocationFailed)
            }),
            None => ExactShared::try_new(SharedBytecode {
                bytecode: None,
                display_name: String::new(),
                _claim: None,
            })
            .map_err(|_| {
                HostAllocationError::Numeric(ExactHostNumericAllocationError::AllocationFailed)
            }),
        }
        .map_err(|error| allocation_error(context, error))?;
        if let Some(receipt) = &receipt {
            receipt
                .checkpoint(context)
                .map_err(|error| allocation_error(context, error))?;
        }
        shared.get_mut_unique().bytecode = self.direct.take();
        std::mem::swap(
            &mut shared.get_mut_unique().display_name,
            &mut self.display_name,
        );
        self.shared = Some(BytecodeLease(shared));
        Ok(())
    }
}

impl Deref for BytecodeOwner {
    type Target = Bytecode;

    fn deref(&self) -> &Self::Target {
        if let Some(lease) = &self.shared {
            lease
        } else {
            self.direct
                .as_ref()
                .expect("direct compiler product moved only during promotion")
        }
    }
}

fn allocation_error(context: &RuntimeContext, error: HostAllocationError) -> RuntimeError {
    runmat_runtime::context::host_allocation::host_allocation_runtime_error(
        Some(context),
        &format!(
            "RunMat:Interpreter:BytecodeOwner:{}",
            error.runtime_identifier_suffix()
        ),
        "retained compiler product admission failed",
        error,
    )
}
