//! Worker-side admission for an invocation-owned canonical builtin. The
//! declaration, full catalog revision and exact compiled binding all have to
//! agree before any argument is decoded or implementation is called.

use std::rc::Rc;
use std::sync::Arc;

use runmat_builtins::{
    builtin_catalog_entry_by_name, builtin_catalog_fingerprint, BuiltinOwnedWorkerPolicy,
};
use runmat_execution::{
    OwnedBuiltinBinding, OwnedBuiltinWorkRequest, OwnedBuiltinWorkResponse,
    OWNED_BUILTIN_VALUE_LIMITS,
};

use crate::builtin::{registration, RuntimeBuiltinBinding, RuntimeBuiltinInvocation};
use crate::context::host_effect::{ManagedHostEffectLedger, ManagedHostEffectStageReservation};
use crate::context::RuntimeServiceFuture;
use crate::context::{
    ParallelCapability, RuntimeContext, RuntimeParallelResources, RuntimeParallelService,
    RuntimeServicePorts,
};

/// Browser-owned canonical work. A submitted stage claim remains with the
/// service until the worker's response or physical termination acknowledgment,
/// independently of whether the caller continues awaiting it.
pub trait RuntimeOwnedBuiltinWorkService {
    /// Exact continuation layout issued by the installed worker target. A
    /// different host architecture must not quote its own Rust future size.
    fn continuation_control_bytes(
        &self,
        binding: &OwnedBuiltinBinding,
    ) -> Result<Option<u64>, super::ExecutionServiceError>;

    fn submit(
        &self,
        request: OwnedBuiltinWorkRequest,
        retained_scratch: Option<Arc<ManagedHostEffectStageReservation>>,
        cancellation: runmat_execution::cancellation::CancellationToken,
    ) -> RuntimeServiceFuture<Result<super::RootedValueSequence, super::ExecutionServiceError>>;
}

/// The scheduler has reserved one worker and this request's finite memory
/// envelope before invoking the owned worker. Its numerical allocations use
/// one local managed owner for that same per-job envelope, not an unbounded
/// default RuntimeContext or a newly minted ledger at every allocation.
struct OwnedJobParallel(ManagedHostEffectLedger);

impl RuntimeParallelService for OwnedJobParallel {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }

    fn placement_resources(&self) -> RuntimeParallelResources {
        let snapshot = self.0.snapshot();
        RuntimeParallelResources {
            cpu_millicores_available: 1_000,
            memory_available_bytes: Some(snapshot.free_bytes()),
            epoch: snapshot.epoch,
        }
    }

    fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
        Some(self.0.clone())
    }
}

fn exact_binding_identity(
    requested: &OwnedBuiltinBinding,
) -> Result<RuntimeBuiltinBinding, String> {
    requested.validate().map_err(|error| error.to_string())?;
    if requested.catalog.bytes() != &builtin_catalog_fingerprint() {
        return Err("owned builtin catalog revision differs from this worker".into());
    }
    let entry = builtin_catalog_entry_by_name(&requested.name)
        .filter(|entry| entry.identity.name == requested.name)
        .ok_or_else(|| "owned builtin has no exact canonical catalog identity".to_string())?;
    let declared = entry.bindings.iter().find(|declaration| {
        declaration.variant == requested.variant
            && declaration.owned_worker == BuiltinOwnedWorkerPolicy::PortableHostCpu
    });
    if declared.is_none() {
        return Err("canonical binding does not admit owned host CPU work".into());
    }
    let registrations = registration::registered_runtime_builtin_bindings_by_name(&requested.name);
    if registrations.len() != 1 {
        return Err("owned builtin has no unique compiled binding".into());
    }
    let selected = registrations[0];
    let binding = selected.binding();
    if binding.identity.builtin.name != requested.name
        || binding.identity.variant != requested.variant
        || !selected.registration().is_current()
    {
        return Err("owned builtin binding revision changed".into());
    }
    Ok(binding)
}

pub fn quote_owned_builtin_continuation(
    binding: &OwnedBuiltinBinding,
) -> Result<Option<u64>, String> {
    let selected = exact_binding_identity(binding)?;
    selected
        .continuation_layout
        .map(|layout| layout().map_err(|error| error.to_string()))
        .transpose()
}

fn exact_binding(request: &OwnedBuiltinWorkRequest) -> Result<RuntimeBuiltinBinding, String> {
    request.validate_wire().map_err(|error| error.to_string())?;
    let selected = exact_binding_identity(&request.binding)?;
    let layout = selected
        .continuation_layout
        .ok_or_else(|| "owned builtin has no compiled continuation quote".to_string())?;
    let bytes = layout().map_err(|error| error.to_string())?;
    if bytes == 0 || bytes != request.control_scratch_bytes {
        return Err("owned builtin continuation quote differs from selected worker".into());
    }
    Ok(selected)
}

pub fn admit_owned_builtin_work(request: &OwnedBuiltinWorkRequest) -> Result<(), String> {
    exact_binding(request).map(|_| ())
}

pub async fn execute_owned_builtin_work(
    request: OwnedBuiltinWorkRequest,
) -> OwnedBuiltinWorkResponse {
    let binding = match exact_binding(&request) {
        Ok(binding) => binding,
        Err(message) => return OwnedBuiltinWorkResponse::InfrastructureFailure { message },
    };
    // This worker invocation has no session/source/path authority. Its exact
    // portable arguments, catalog binding and isolated numerical implementation
    // are the whole executable contract.
    let ledger = ManagedHostEffectLedger::new_with_epoch(request.resources.memory_bytes, 1);
    let _transport_stage = match ledger.reserve_stage(request.transport_scratch_bytes) {
        Ok(stage) => stage,
        Err(error) => {
            return OwnedBuiltinWorkResponse::InfrastructureFailure {
                message: format!("owned builtin transport quota was unavailable: {error}"),
            }
        }
    };
    let runtime = RuntimeContext::new(Rc::new(super::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default().with_parallel(Rc::new(OwnedJobParallel(ledger))),
        );
    let mut cooperation = runtime.cpu_cooperation();
    let values = match super::value_codec::decode_inline_values_cooperatively_in_context(
        &request.arguments,
        OWNED_BUILTIN_VALUE_LIMITS,
        None,
        Some(&mut cooperation),
        &runtime,
    )
    .await
    {
        Ok(values) => values,
        Err(error) => {
            return OwnedBuiltinWorkResponse::InfrastructureFailure {
                message: format!("owned builtin input could not be decoded: {error}"),
            }
        }
    };
    let invocation_runtime = runtime.clone();
    let requested_outputs = usize::from(request.requested_outputs);
    let result = values
        .with_owned_output_storage_async(move |values| async move {
            (binding.implementation)(RuntimeBuiltinInvocation::new(
                &values,
                Some(invocation_runtime.with_vm_fusion_selected_binding(Some(binding.identity))),
                Some(requested_outputs),
            ))
            .await
        })
        .await;
    let response = match result {
        Ok(sequence) => {
            let outputs = sequence
                .value()
                .iter()
                .map(|value| super::value_codec::encode_inline_value_in_context(value, &runtime))
                .collect::<Result<Vec<_>, _>>();
            match outputs {
                Ok(outputs) if outputs.len() == usize::from(request.requested_outputs) => {
                    match request.validate_output_values(&outputs) {
                        Ok(()) => OwnedBuiltinWorkResponse::Success { outputs },
                        Err(error) => OwnedBuiltinWorkResponse::InfrastructureFailure {
                            message: format!(
                                "owned builtin output exceeds portable limits: {error}"
                            ),
                        },
                    }
                }
                Ok(_) => OwnedBuiltinWorkResponse::InfrastructureFailure {
                    message: "owned builtin returned the wrong output count".into(),
                },
                Err(error) => OwnedBuiltinWorkResponse::InfrastructureFailure {
                    message: format!("owned builtin output could not be encoded: {error}"),
                },
            }
        }
        Err(error) => match super::encode_runtime_failure(&error) {
            Ok(failure) => OwnedBuiltinWorkResponse::RuntimeFailure { failure },
            Err(message) => OwnedBuiltinWorkResponse::InfrastructureFailure { message },
        },
    };
    if request.validate_response(&response).is_ok() {
        response
    } else {
        OwnedBuiltinWorkResponse::InfrastructureFailure {
            message: "owned builtin response exceeded its admitted transport envelope".into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::host_allocation::{HostAllocationPlan, HostAllocationReceipt};

    #[test]
    fn compiled_owned_binding_quotes_the_worker_continuation() {
        use runmat_execution::resource::{Capability, ResourceRequest};
        use runmat_execution::value::{ElementType, InlineValue, ValuePayload};
        use runmat_execution::{
            Digest, OwnedBuiltinDenseOutputContract, OwnedBuiltinResponseContract,
            OWNED_BUILTIN_FAILURE_WIRE_BYTES, OWNED_BUILTIN_WORK_SCHEMA_VERSION,
        };
        use std::collections::BTreeSet;

        let binding = OwnedBuiltinBinding {
            name: "qz".into(),
            variant: "default".into(),
            catalog: Digest::from_bytes(builtin_catalog_fingerprint()),
        };
        let control = quote_owned_builtin_continuation(&binding)
            .expect("exact compiled binding")
            .expect("compiled continuation quote");
        assert!(control > 0);
        let transport = OWNED_BUILTIN_FAILURE_WIRE_BYTES;
        let scratch = transport.checked_add(control).expect("finite quote");
        let request = OwnedBuiltinWorkRequest {
            schema_version: OWNED_BUILTIN_WORK_SCHEMA_VERSION,
            binding,
            arguments: vec![
                ValuePayload::Inline(Box::new(InlineValue::F64Bits(1.0f64.to_bits()))),
                ValuePayload::Inline(Box::new(InlineValue::F64Bits(1.0f64.to_bits()))),
            ],
            requested_outputs: 1,
            response: OwnedBuiltinResponseContract {
                dense: vec![OwnedBuiltinDenseOutputContract {
                    shape: vec![1, 1],
                    allowed_element_types: vec![ElementType::F64],
                }],
                max_wire_bytes: transport,
            },
            resources: ResourceRequest {
                cpu_millicores: 1000,
                memory_bytes: scratch + 64,
                scratch_bytes: scratch,
                max_wall_millis: 1000,
                max_artifact_bytes: 0,
                max_egress_bytes: 0,
                max_relay_bytes: 0,
                accelerators: Vec::new(),
                required_capabilities: BTreeSet::from([Capability::BrowserWorker]),
            },
            transport_scratch_bytes: transport,
            control_scratch_bytes: control,
            assignment: None,
        };
        admit_owned_builtin_work(&request).expect("real compiled quote admits ordinary work");

        let mut mismatched = request;
        mismatched.control_scratch_bytes += 1;
        mismatched.resources.scratch_bytes += 1;
        mismatched.resources.memory_bytes += 1;
        assert!(admit_owned_builtin_work(&mismatched).is_err());
    }

    #[test]
    fn owned_worker_numerical_receipts_share_its_finite_job_ledger() {
        let ledger = ManagedHostEffectLedger::new_with_epoch(128, 1);
        let runtime = RuntimeContext::new(Rc::new(super::super::RuntimeExecutionService::new()))
            .with_service_ports(
                RuntimeServicePorts::default()
                    .with_parallel(Rc::new(OwnedJobParallel(ledger.clone()))),
            );
        let first =
            HostAllocationReceipt::admit(Some(&runtime), HostAllocationPlan::new(96, 0).unwrap())
                .unwrap()
                .unwrap();
        assert!(HostAllocationReceipt::admit(
            Some(&runtime),
            HostAllocationPlan::new(64, 0).unwrap(),
        )
        .is_err());
        drop(first);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }
}
