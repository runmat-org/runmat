//! Exact retained owner for a dynamic target's completion service.
//!
//! Existing Base services already have an `Rc` owner. A composite service is
//! newly allocated for one capture, so finite operations instead fund its
//! concrete box and this known-layout shared header before publication.

use std::alloc::Layout;
use std::rc::Rc;

use allocator_api2::boxed::Box as FallibleBox;
use runmat_execution::resource::ExactShared;
use runmat_value::{ExactHostAllocationClaim, ExactHostAllocationPermit};

use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::{RuntimeContext, RuntimeWorkspaceService};
use crate::RuntimeError;

struct AdmittedIssuer {
    service: Box<dyn RuntimeWorkspaceService>,
    // Field order ensures the service box and shared header retire before
    // their respective physical-allocation claims are refunded.
    _service_claim: ExactHostAllocationClaim,
    _header_claim: ExactHostAllocationClaim,
}

#[derive(Clone)]
enum IssuerBacking {
    Existing(Rc<dyn RuntimeWorkspaceService>),
    Admitted(ExactShared<AdmittedIssuer>),
}

#[derive(Clone)]
pub(super) struct DynamicCaptureIssuer(IssuerBacking);

impl DynamicCaptureIssuer {
    pub(super) fn existing(service: Rc<dyn RuntimeWorkspaceService>) -> Self {
        Self(IssuerBacking::Existing(service))
    }

    pub(super) fn service(&self) -> &dyn RuntimeWorkspaceService {
        match &self.0 {
            IssuerBacking::Existing(service) => service.as_ref(),
            IssuerBacking::Admitted(owner) => owner.get().service.as_ref(),
        }
    }

    pub(super) fn prepare<T>(context: &RuntimeContext, service: T) -> Result<Self, RuntimeError>
    where
        T: RuntimeWorkspaceService + 'static,
    {
        let service_bytes = Layout::new::<T>().size();
        let header_bytes = ExactShared::<AdmittedIssuer>::requested_bytes();
        let total = service_bytes
            .checked_add(header_bytes)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| admission_error(context, HostAllocationError::Overflow))?;
        let retained = HostAllocationPlan::retained_exact_output_bytes(total, 2)
            .map_err(|cause| admission_error(context, cause))?;
        let plan = HostAllocationPlan::new(0, retained)
            .map_err(|cause| admission_error(context, cause))?;
        let receipt = HostAllocationReceipt::admit(Some(context), plan)
            .map_err(|cause| admission_error(context, cause))?;
        let Some(receipt) = receipt else {
            return Ok(Self::existing(Rc::new(service)));
        };
        let permit = receipt
            .output_permit(context)
            .map_err(|cause| admission_error(context, cause))?;
        let service_claim = permit
            .reserve_exact_claim(service_bytes)
            .map_err(HostAllocationError::from)
            .map_err(|cause| admission_error(context, cause))?;
        let allocated = FallibleBox::try_new(service).map_err(|_| allocation_failed(context))?;
        let pointer = FallibleBox::into_raw(allocated);
        // SAFETY: allocator-api2's Global allocator uses the same global
        // allocation/deallocation contract as std::Box. This fresh allocation
        // has exactly Layout::new::<T>() and transfers once here.
        let service: Box<T> = unsafe { Box::from_raw(pointer) };
        let service: Box<dyn RuntimeWorkspaceService> = service;
        let header_claim = permit
            .reserve_exact_claim(header_bytes)
            .map_err(HostAllocationError::from)
            .map_err(|cause| admission_error(context, cause))?;
        let owner = ExactShared::try_new(AdmittedIssuer {
            service,
            _service_claim: service_claim,
            _header_claim: header_claim,
        })
        .map_err(|_| allocation_failed(context))?;
        receipt
            .checkpoint(context)
            .map_err(|cause| admission_error(context, cause))?;
        Ok(Self(IssuerBacking::Admitted(owner)))
    }
}

fn allocation_failed(context: &RuntimeContext) -> RuntimeError {
    admission_error(
        context,
        HostAllocationError::Numeric(
            runmat_value::ExactHostNumericAllocationError::AllocationFailed,
        ),
    )
}

fn admission_error(context: &RuntimeContext, cause: HostAllocationError) -> RuntimeError {
    crate::build_runtime_error(format!(
        "dynamic workspace issuer admission failed: {cause}"
    ))
    .with_identifier(format!(
        "{}:{}",
        context.error_namespace(),
        cause.runtime_identifier_suffix()
    ))
    .with_source(cause)
    .build()
}
