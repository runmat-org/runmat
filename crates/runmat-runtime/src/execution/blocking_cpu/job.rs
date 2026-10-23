//! The one typed kernel Box and its original external physical claim.
#[cfg(not(target_arch = "wasm32"))]
use crate::context::{
    host_allocation::{
        HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
        HostAllocationWorkerCheckpoint,
    },
    RuntimeContext,
};
#[cfg(not(target_arch = "wasm32"))]
use crate::execution::ExecutionServiceError;
use runmat_value::ExactHostAllocationClaim;
#[cfg(not(target_arch = "wasm32"))]
use runmat_value::ExactHostAllocationPermit;

trait CpuJobBody: Send {
    fn run(self: Box<Self>);
}

pub struct OwnedCpuJob {
    body: Box<dyn CpuJobBody>,
    claim: Option<ExactHostAllocationClaim>,
}

impl OwnedCpuJob {
    #[cfg(not(target_arch = "wasm32"))]
    pub(in crate::execution) fn prepare_kernel<T, G, F>(
        runtime: Option<&RuntimeContext>,
        send: super::completion::Sender<T, G>,
        cancellation: runmat_execution::cancellation::CancellationToken,
        work: F,
    ) -> Result<Self, ExecutionServiceError>
    where
        T: Send + 'static,
        G: Send + 'static,
        F: FnOnce() -> T + Send + 'static,
    {
        let bytes = std::mem::size_of::<KernelJob<T, G, F>>();
        let retained = HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(bytes).map_err(|_| {
                ExecutionServiceError::HostAllocation(HostAllocationError::Overflow)
            })?,
            1,
        )
        .map_err(ExecutionServiceError::HostAllocation)?;
        let receipt = HostAllocationReceipt::admit(
            runtime,
            HostAllocationPlan::new(0, retained).map_err(ExecutionServiceError::HostAllocation)?,
        )
        .map_err(ExecutionServiceError::HostAllocation)?;
        let checkpoint = receipt
            .as_ref()
            .map(HostAllocationReceipt::worker_checkpoint);
        let claim = match &receipt {
            Some(receipt) => Some(
                receipt
                    .output_permit(runtime.expect("managed job has a context"))
                    .map_err(ExecutionServiceError::HostAllocation)?
                    .reserve_exact_claim(bytes)
                    .map_err(|cause| ExecutionServiceError::HostAllocation(cause.into()))?,
            ),
            None => None,
        };
        let body = try_box(KernelJob {
            work: Some(work),
            cancellation,
            checkpoint,
            send: Some(send),
        })
        .map_err(|body| {
            drop(body);
            ExecutionServiceError::HostAllocation(HostAllocationError::Numeric(
                runmat_value::ExactHostNumericAllocationError::AllocationFailed,
            ))
        })?;
        let job = Self { body, claim };
        if let Some(receipt) = &receipt {
            receipt
                .checkpoint(runtime.expect("managed job has a context"))
                .map_err(ExecutionServiceError::HostAllocation)?;
        }
        Ok(job)
    }

    /// The consumed body's actual Box retires before its original claim,
    /// including unwind from a panicking kernel.
    pub fn run(self) {
        let Self { body, claim } = self;
        body.run();
        drop(claim);
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct KernelJob<T, G, F> {
    work: Option<F>,
    cancellation: runmat_execution::cancellation::CancellationToken,
    checkpoint: Option<HostAllocationWorkerCheckpoint>,
    // The numerical reservation lives in this completion's shared state.
    // Work is taken/dropped before this sender publishes or signals failure.
    send: Option<super::completion::Sender<T, G>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl<T, G, F> KernelJob<T, G, F> {
    fn check(&self) -> Result<(), ExecutionServiceError> {
        if self.cancellation.is_cancelled() {
            return Err(ExecutionServiceError::Cancelled);
        }
        self.checkpoint.as_ref().map_or(Ok(()), |checkpoint| {
            checkpoint
                .check()
                .map_err(ExecutionServiceError::HostAllocation)
        })
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl<T, G, F> CpuJobBody for KernelJob<T, G, F>
where
    T: Send + 'static,
    G: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    fn run(mut self: Box<Self>) {
        let work = self.work.take().expect("one CPU kernel");
        let result = match self.check() {
            Ok(()) => Ok(work()),
            Err(error) => {
                drop(work);
                Err(error)
            }
        };
        let result = match self.check() {
            Ok(()) => result,
            Err(error) => {
                drop(result);
                Err(error)
            }
        };
        self.send.take().expect("one CPU completion").send(result);
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn try_box<F>(value: F) -> Result<Box<F>, F> {
    let layout = std::alloc::Layout::new::<F>();
    if layout.size() == 0 {
        return Ok(Box::new(value));
    }
    // SAFETY: the exact F layout matches the sole Box<F>'s eventual free.
    let pointer = unsafe { std::alloc::alloc(layout) }.cast::<F>();
    if pointer.is_null() {
        return Err(value);
    }
    // SAFETY: fresh aligned storage is initialized exactly once; ownership is
    // immediately transferred to Box. No allocation claim lives inside it.
    unsafe {
        pointer.write(value);
        Ok(Box::from_raw(pointer))
    }
}
