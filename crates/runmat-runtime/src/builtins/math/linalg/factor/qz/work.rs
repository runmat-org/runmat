//! Invocation-owned QZ CPU work.  The native blocking pool owns the scratch
//! reservation until faer's synchronous kernel physically completes, even if
//! the caller stops awaiting after cancellation.

use crate::context::host_allocation::HostAllocationPlan;
use crate::context::host_allocation::{HostAllocationError, HostAllocationReceipt};
use crate::context::RuntimeContext;
use crate::execution::{CpuCooperationBudget, ExecutionServiceError};
use crate::{BuiltinResult, RuntimeError};
use runmat_execution::resource::{Capability, ResourceRequest};
use runmat_execution::value::ElementType;
use runmat_execution::{
    Digest, OwnedBuiltinBinding, OwnedBuiltinDenseOutputContract, OwnedBuiltinWorkRequest,
    OWNED_BUILTIN_VALUE_LIMITS, OWNED_BUILTIN_WORK_SCHEMA_VERSION,
};
use runmat_value::{NumericDType, Value};
use std::collections::BTreeSet;
use std::io::{self, Write};

#[derive(Default)]
struct WireLength(u64);

impl Write for WireLength {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0 = self
            .0
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| io::Error::other("owned builtin wire length overflow"))?;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub(super) struct QzWork<'a> {
    runtime: Option<&'a RuntimeContext>,
    receipt: Option<&'a HostAllocationReceipt>,
    budget: Option<CpuCooperationBudget>,
    name: &'static str,
}

#[derive(Clone, Copy)]
pub(super) enum QzOutputDomain {
    Real,
    Complex,
    Either,
}

#[derive(Clone, Copy)]
pub(super) struct QzOutputRule {
    pub(super) shape: [usize; 2],
    pub(super) dtype: NumericDType,
    pub(super) domain: QzOutputDomain,
}

impl QzOutputRule {
    fn portable_contract(self) -> BuiltinResult<OwnedBuiltinDenseOutputContract> {
        let real = match self.dtype {
            NumericDType::F32 => ElementType::F32,
            NumericDType::F64 => ElementType::F64,
            _ => return Err(ExecutionServiceError::InvalidOutputContract.into_runtime_error()),
        };
        let complex = match self.dtype {
            NumericDType::F32 => ElementType::ComplexF32,
            NumericDType::F64 => ElementType::ComplexF64,
            _ => return Err(ExecutionServiceError::InvalidOutputContract.into_runtime_error()),
        };
        let allowed_element_types = match self.domain {
            QzOutputDomain::Real => vec![real],
            QzOutputDomain::Complex => vec![complex],
            QzOutputDomain::Either => vec![real, complex],
        };
        Ok(OwnedBuiltinDenseOutputContract {
            shape: self
                .shape
                .into_iter()
                .map(|axis| {
                    u64::try_from(axis).map_err(|_| {
                        ExecutionServiceError::InvalidOutputContract.into_runtime_error()
                    })
                })
                .collect::<BuiltinResult<Vec<_>>>()?,
            allowed_element_types,
        })
    }

    fn accepts(self, value: &Value) -> bool {
        match value {
            Value::Tensor(tensor) => {
                tensor.shape() == self.shape.as_slice()
                    && tensor.numeric_dtype() == self.dtype
                    && matches!(self.domain, QzOutputDomain::Real | QzOutputDomain::Either)
            }
            Value::ComplexTensor(tensor) => {
                tensor.shape() == self.shape.as_slice()
                    && tensor.numeric_dtype() == self.dtype
                    && matches!(
                        self.domain,
                        QzOutputDomain::Complex | QzOutputDomain::Either
                    )
            }
            _ => false,
        }
    }
}

impl<'a> QzWork<'a> {
    pub(super) fn new(
        runtime: Option<&'a RuntimeContext>,
        receipt: Option<&'a HostAllocationReceipt>,
        name: &'static str,
    ) -> Self {
        Self {
            runtime,
            receipt,
            budget: runtime.map(RuntimeContext::cpu_cooperation),
            name,
        }
    }

    fn check(&self) -> BuiltinResult<()> {
        if let (Some(runtime), Some(receipt)) = (self.runtime, self.receipt) {
            receipt
                .checkpoint(runtime)
                .map_err(|source| allocation_error(self.name, source))?;
        } else if self.runtime.is_some_and(RuntimeContext::is_cancelled) {
            return Err(ExecutionServiceError::Cancelled.into_runtime_error());
        }
        Ok(())
    }

    pub(super) async fn checkpoint(&mut self, completed_work: u64) -> BuiltinResult<()> {
        self.check()?;
        if let Some(budget) = self.budget.as_mut() {
            budget
                .checkpoint(completed_work)
                .await
                .map_err(ExecutionServiceError::into_runtime_error)?;
        }
        self.check()
    }

    /// Transfer a preflighted, host-only numeric invocation to the installed
    /// browser worker. No worker port means the existing serial/native path.
    /// The exact QZ allocation plan, not a name-derived scheduler guess,
    /// supplies the resource envelope before materialization or execution.
    pub(super) async fn try_owned_browser_worker(
        &mut self,
        arguments: &[&Value],
        requested_outputs: usize,
        explicit_output_count: bool,
        plan: HostAllocationPlan,
        output_rule: QzOutputRule,
    ) -> BuiltinResult<Option<Value>> {
        let Some(runtime) = self.runtime else {
            return Ok(None);
        };
        let Some(service) = runtime.service_ports().owned_builtin_work().cloned() else {
            return Ok(None);
        };
        self.checkpoint(0).await?;
        let binding = OwnedBuiltinBinding {
            name: self.name.into(),
            variant: "default".into(),
            catalog: Digest::from_bytes(runmat_builtins::builtin_catalog_fingerprint()),
        };
        let Some(control_scratch_bytes) = service
            .continuation_control_bytes(&binding)
            .map_err(ExecutionServiceError::into_runtime_error)?
        else {
            // An optional worker without a target-compiled continuation quote
            // cannot admit finite work; serial execution remains available.
            return Ok(None);
        };
        if control_scratch_bytes == 0 {
            return Err(ExecutionServiceError::InvalidOutputContract.into_runtime_error());
        }
        let requested_outputs = u16::try_from(requested_outputs)
            .map_err(|_| ExecutionServiceError::InvalidOutputContract.into_runtime_error())?;
        let output_contracts = if requested_outputs == 0 {
            Vec::new()
        } else {
            let output_contract = output_rule.portable_contract()?;
            output_contract.validate().map_err(|error| {
                ExecutionServiceError::Failed(error.to_string()).into_runtime_error()
            })?;
            vec![output_contract; usize::from(requested_outputs)]
        };
        let transport_plan = crate::execution::value_codec::OwnedInlineTransportPlan::for_values(
            arguments,
            plan.retained_output_bytes(),
            &output_contracts,
        )
        .map_err(|error| ExecutionServiceError::Failed(error.to_string()).into_runtime_error())?;
        let coordinator_transport = transport_plan.coordinator_peak_bytes().map_err(|error| {
            ExecutionServiceError::Failed(error.to_string()).into_runtime_error()
        })?;
        // Admit this separately on the *same* parent managed ledger before
        // the inline codec allocates any payload. Numerical QZ allocations
        // belong to the selected worker, not to a duplicate parent receipt.
        let transport_receipt = HostAllocationReceipt::admit(
            Some(runtime),
            HostAllocationPlan::new(coordinator_transport, 0)
                .map_err(|source| allocation_error(self.name, source))?,
        )
        .map_err(|source| allocation_error(self.name, source))?;
        if let Some(receipt) = transport_receipt.as_ref() {
            receipt
                .checkpoint(runtime)
                .map_err(|source| allocation_error(self.name, source))?;
        }
        let payloads = arguments
            .iter()
            .map(|value| {
                crate::execution::value_codec::encode_inline_value_in_context(value, runtime)
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| {
                ExecutionServiceError::Failed(error.to_string()).into_runtime_error()
            })?;
        runmat_execution::value::ValuePayload::validate_all(&payloads, OWNED_BUILTIN_VALUE_LIMITS)
            .map_err(|error| {
                ExecutionServiceError::Failed(error.to_string()).into_runtime_error()
            })?;
        let output = plan.retained_output_bytes();
        let scratch = plan.scratch_peak_bytes();
        let worker_transport = transport_plan.worker_peak_bytes().map_err(|error| {
            ExecutionServiceError::Failed(error.to_string()).into_runtime_error()
        })?;
        let memory_bytes = scratch
            .checked_add(output)
            .and_then(|bytes| bytes.checked_add(worker_transport))
            .and_then(|bytes| bytes.checked_add(control_scratch_bytes))
            .ok_or_else(|| ExecutionServiceError::InvalidOutputContract.into_runtime_error())?;
        let scratch_bytes = scratch
            .checked_add(worker_transport)
            .and_then(|bytes| bytes.checked_add(control_scratch_bytes))
            .ok_or_else(|| ExecutionServiceError::InvalidOutputContract.into_runtime_error())?;
        let request = OwnedBuiltinWorkRequest {
            schema_version: OWNED_BUILTIN_WORK_SCHEMA_VERSION,
            binding,
            arguments: payloads,
            requested_outputs,
            response: runmat_execution::OwnedBuiltinResponseContract {
                dense: output_contracts,
                max_wire_bytes: transport_plan.output_wire_bound(),
            },
            resources: ResourceRequest {
                cpu_millicores: 1000,
                memory_bytes: memory_bytes.max(1),
                scratch_bytes,
                max_wall_millis: 24 * 60 * 60 * 1000,
                max_artifact_bytes: 0,
                max_egress_bytes: 0,
                max_relay_bytes: 0,
                accelerators: Vec::new(),
                required_capabilities: BTreeSet::from([Capability::BrowserWorker]),
            },
            transport_scratch_bytes: worker_transport,
            control_scratch_bytes,
            assignment: None,
        };
        let mut wire = WireLength::default();
        serde_json::to_writer(&mut wire, &request).map_err(|error| {
            ExecutionServiceError::Failed(error.to_string()).into_runtime_error()
        })?;
        if wire.0 > transport_plan.input_wire_bound() {
            return Err(ExecutionServiceError::InvalidOutputContract.into_runtime_error());
        }
        if let Some(receipt) = transport_receipt.as_ref() {
            receipt
                .checkpoint(runtime)
                .map_err(|source| allocation_error(self.name, source))?;
        }
        let retained = transport_receipt
            .as_ref()
            .map(HostAllocationReceipt::retain_scratch_for_cpu_work);
        let result = service
            .submit(request, retained, runtime.cancellation())
            .await
            .map_err(ExecutionServiceError::into_runtime_error)?;
        self.checkpoint(1).await?;
        if let Some(receipt) = transport_receipt.as_ref() {
            receipt
                .checkpoint(runtime)
                .map_err(|source| allocation_error(self.name, source))?;
        }
        let value = result.with_owned(|sequence| {
            let outputs = sequence.into_output_storage();
            // The QZ-family owner checks every physical output before
            // publication. These numeric values carry no managed GC handles.
            if !outputs.iter().all(|value| output_rule.accepts(value)) {
                return Err(ExecutionServiceError::Infrastructure(
                    "owned QZ worker returned a wrong class, domain, or shape".into(),
                )
                .into_runtime_error());
            }
            if explicit_output_count {
                Ok(Value::OutputList(outputs))
            } else {
                outputs
                    .try_into_first()
                    .map_err(|_| {
                        crate::sequence::sequence_error_to_runtime(
                            runmat_value::ValueSequenceError::SharedOutputStorage,
                        )
                    })?
                    .ok_or_else(|| {
                        ExecutionServiceError::InvalidOutputContract.into_runtime_error()
                    })
            }
        })?;
        if let Some(receipt) = transport_receipt.as_ref() {
            receipt
                .checkpoint(runtime)
                .map_err(|source| allocation_error(self.name, source))?;
        }
        Ok(Some(value))
    }

    pub(super) async fn blocking<T, F>(&mut self, work: F) -> BuiltinResult<T>
    where
        T: Send + 'static,
        F: FnOnce() -> T + Send + 'static,
    {
        self.checkpoint(0).await?;
        let result = if let Some(budget) = self.budget.as_ref() {
            let retained = self
                .receipt
                .map(HostAllocationReceipt::retain_scratch_for_cpu_work);
            let service = self
                .runtime
                .and_then(|runtime| runtime.service_ports().blocking_cpu())
                .map(|service| service.as_ref());
            budget
                .run_owned_blocking_in_context(
                    self.runtime
                        .expect("a cooperation budget has its original runtime"),
                    service,
                    retained,
                    work,
                )
                .await
                .map_err(ExecutionServiceError::into_runtime_error)?
        } else {
            work()
        };
        self.checkpoint(1).await?;
        Ok(result)
    }
}

fn allocation_error(name: &'static str, source: HostAllocationError) -> RuntimeError {
    let mut error = match source {
        HostAllocationError::Cancelled => ExecutionServiceError::Cancelled.into_runtime_error(),
        HostAllocationError::StaleAuthority | HostAllocationError::MissingManagedOwner => {
            ExecutionServiceError::Infrastructure(
                "managed host allocation authority changed".into(),
            )
            .into_runtime_error()
        }
        _ => crate::build_runtime_error(format!("{name}: host allocation failed: {source}"))
            .with_builtin(name)
            .with_identifier("RunMat:Qz:HostAllocation")
            .build(),
    };
    error.source = Some(Box::new(source));
    error
}

pub(super) fn admit_error(name: &'static str, source: HostAllocationError) -> RuntimeError {
    allocation_error(name, source)
}
