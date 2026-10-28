//! Original owner for the native frame's borrowed ABI root projection.
//! This is metadata only: language payloads stay with their existing owners.

use super::*;
use runmat_runtime::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use runmat_value::{exact_capacity_vec, ExactHostAllocationClaim};
use std::alloc::Layout;

#[derive(Default)]
pub(in crate::execute) struct NativeRoots {
    pub(super) values: Vec<NativeRoot>,
    // Field order retires the physical buffer before refunding its claim.
    claim: Option<ExactHostAllocationClaim>,
}

impl NativeRoots {
    pub(super) fn abi_view(&self) -> NativeRootSet {
        NativeRootSet {
            roots: if self.values.is_empty() {
                std::ptr::null()
            } else {
                self.values.as_ptr()
            },
            count: self.values.len(),
        }
    }

    pub(super) fn reserve(
        &mut self,
        runtime: &RuntimeContext,
        count: usize,
    ) -> NativeExecutorResult<()> {
        u32::try_from(count)
            .map_err(|_| NativeExecutorError::Host("native root cardinality overflow".into()))?;
        if count <= self.values.capacity() {
            return Ok(());
        }
        let error = |cause: HostAllocationError| {
            NativeExecutorError::from(host_allocation_runtime_error(
                Some(runtime),
                "RunMat:Native:RootAdmission",
                "native root metadata admission failed",
                cause,
            ))
        };
        let bytes = Layout::array::<NativeRoot>(count)
            .map_err(|_| error(HostAllocationError::Overflow))?
            .size();
        let retained = HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(bytes).map_err(|_| error(HostAllocationError::Overflow))?,
            1,
        )
        .map_err(error)?;
        let plan = HostAllocationPlan::new(0, retained).map_err(error)?;
        let receipt = HostAllocationReceipt::admit(Some(runtime), plan).map_err(error)?;
        // Charge the replacement while the old block is still physically
        // alive. A denied growth leaves that block and any ABI view unchanged.
        let (values, claim) = match receipt {
            Some(receipt) => receipt
                .with_output_allocation(runtime, |permit| {
                    let claim = permit.reserve_exact_claim(bytes)?;
                    Ok((exact_capacity_vec(count)?, Some(claim)))
                })
                .map_err(error)?,
            None => (
                exact_capacity_vec(count)
                    .map_err(|cause| error(HostAllocationError::Numeric(cause)))?,
                None,
            ),
        };
        // Keep the prior logical projection until the infallible refresh.
        let mut values = values;
        values.extend_from_slice(&self.values);
        let old_values = std::mem::replace(&mut self.values, values);
        let old_claim = std::mem::replace(&mut self.claim, claim);
        drop(old_values);
        drop(old_claim);
        Ok(())
    }
}

impl std::ops::Deref for NativeRoots {
    type Target = [NativeRoot];

    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_view_preserves_empty_and_retained_capacity_invariants() {
        let mut roots = NativeRoots::default();
        roots.abi_view().validate().unwrap();
        roots.values.push(NativeRoot {
            value: NativeValueRef::NULL,
            kind: NativeRootKind::LOCAL,
            slot: 0,
        });
        let view = roots.abi_view();
        view.validate().unwrap();
        assert_eq!(view.roots, roots.values.as_ptr());
        assert_eq!(view.count, 1);
        roots.values.clear();
        assert!(roots.values.capacity() > 0);
        let view = roots.abi_view();
        view.validate().unwrap();
        assert!(view.roots.is_null());
    }
}
