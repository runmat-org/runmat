use std::cell::RefCell;
use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;

use crate::execution::RootedValueSequence;
use runmat_types::InteropManifest;
use runmat_value::ValueSequence;

use super::{
    admit_interop_manifest, foreign_error, ForeignAdapterDescriptor, ForeignErrorKind,
    ForeignHandleRegistry, ForeignPlatform, ForeignTelemetryEvent, ForeignTelemetryOutcome,
    ForeignTelemetrySink, InteropAdmissionPlan, NoopForeignTelemetry,
};
use crate::context::{ForeignCall, ForeignCallCompletion, RuntimeContext, RuntimeForeignService};
use crate::RuntimeError;

pub type ForeignAdapterFuture = Pin<Box<dyn Future<Output = ForeignCallCompletion> + 'static>>;

pub(super) fn root_adapter_result(
    result: Result<ValueSequence, RuntimeError>,
) -> Result<RootedValueSequence, RuntimeError> {
    result.and_then(|sequence| {
        RootedValueSequence::new(sequence).map_err(|error| error.into_runtime_error())
    })
}

pub trait ForeignAdapter {
    fn visit_retained_values(
        &self,
        _visitor: &mut dyn FnMut(&runmat_value::Value) -> Option<()>,
    ) -> Option<()> {
        None
    }
    fn descriptor(&self) -> ForeignAdapterDescriptor;
    fn invoke(&self, context: RuntimeContext, call: ForeignCall) -> ForeignAdapterFuture;
    fn is_isolated(&self) -> bool {
        false
    }
}

#[derive(Clone)]
pub struct ForeignRuntime {
    handles: ForeignHandleRegistry,
    adapters: Rc<RefCell<BTreeMap<runmat_types::ForeignAdapterId, Rc<dyn ForeignAdapter>>>>,
    platform: ForeignPlatform,
    telemetry: Rc<dyn ForeignTelemetrySink>,
}

impl std::fmt::Debug for ForeignRuntime {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ForeignRuntime")
            .field("handles", &self.handles)
            .field("adapter_count", &self.adapters.borrow().len())
            .field("platform", &self.platform)
            .finish_non_exhaustive()
    }
}

impl ForeignRuntime {
    pub fn new(platform: ForeignPlatform) -> Self {
        Self::with_telemetry(platform, Rc::new(NoopForeignTelemetry))
    }

    pub fn with_telemetry(
        platform: ForeignPlatform,
        telemetry: Rc<dyn ForeignTelemetrySink>,
    ) -> Self {
        Self {
            handles: ForeignHandleRegistry::default(),
            adapters: Rc::new(RefCell::new(BTreeMap::new())),
            platform,
            telemetry,
        }
    }

    pub fn handles(&self) -> &ForeignHandleRegistry {
        &self.handles
    }

    pub fn execution_host_inventory(&self) -> Vec<runmat_execution::host::ForeignAdapterInventory> {
        self.adapters
            .borrow()
            .values()
            .map(|adapter| {
                let descriptor = adapter.descriptor();
                runmat_execution::host::ForeignAdapterInventory {
                    adapter: descriptor.adapter,
                    version: descriptor.version,
                    capabilities: runmat_types::CapabilitySet(descriptor.capabilities),
                    foreign_capabilities: descriptor.foreign_capabilities,
                    execution_stack: descriptor.execution_stack,
                    supports_wasm: descriptor.supports_wasm,
                    supports_host_bridge: descriptor.supports_host_bridge,
                }
            })
            .collect()
    }

    pub fn register_adapter(&self, adapter: Rc<dyn ForeignAdapter>) -> Result<(), RuntimeError> {
        let descriptor = adapter.descriptor();
        if descriptor.version == 0 {
            return Err(foreign_error(
                ForeignErrorKind::AdapterUnavailable,
                "foreign adapter identity must be non-empty and its version must be non-zero",
            ));
        }
        let mut adapters = self.adapters.borrow_mut();
        if adapters.contains_key(&descriptor.adapter) {
            return Err(foreign_error(
                ForeignErrorKind::AdapterUnavailable,
                format!(
                    "foreign adapter {} is already registered",
                    descriptor.adapter
                ),
            ));
        }
        adapters.insert(descriptor.adapter, adapter);
        Ok(())
    }

    pub fn admit(&self, manifest: &InteropManifest) -> Result<InteropAdmissionPlan, RuntimeError> {
        let adapters = self.adapter_descriptors();
        self.admit_with_descriptors(manifest, &adapters)
    }

    /// Admit against the exact staged MEX catalog without exposing its names
    /// to the live owner before the caller's publication boundary.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn admit_with_prepared_mex_catalog(
        &self,
        manifest: &InteropManifest,
        catalog: &super::PreparedMexArtifactCatalog,
    ) -> Result<InteropAdmissionPlan, RuntimeError> {
        let mut adapters = self.adapter_descriptors();
        if let Some(mex) = adapters.get_mut(runmat_mex::MEX_ADAPTER_ID) {
            mex.artifact_identities = catalog.artifact_identities().cloned().collect();
        }
        self.admit_with_descriptors(manifest, &adapters)
    }

    /// Admit an exact staged replacement inventory. Families without incoming
    /// or prior project-owned artifacts keep their live adapter state intact.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn admit_with_prepared_selected_replacements(
        &self,
        manifest: &InteropManifest,
        native: Option<&super::PreparedNativeProjectReplacement<'_>>,
        mex: &super::PreparedMexArtifactCatalog,
        java: Option<&super::PreparedJavaProjectReplacement<'_>>,
        python: Option<&super::PreparedPythonProjectReplacement<'_>>,
    ) -> Result<InteropAdmissionPlan, RuntimeError> {
        let mut adapters = self.adapter_descriptors();
        if let (Some(descriptor), Some(native)) = (
            adapters.get_mut(runmat_native_ffi::NATIVE_FFI_ADAPTER_ID),
            native,
        ) {
            descriptor.artifact_identities = native.artifact_identities().cloned().collect();
        }
        if let Some(descriptor) = adapters.get_mut(runmat_mex::MEX_ADAPTER_ID) {
            descriptor.artifact_identities = mex.artifact_identities().cloned().collect();
        }
        if let (Some(descriptor), Some(java)) =
            (adapters.get_mut(runmat_java::JAVA_ADAPTER_ID), java)
        {
            descriptor.artifact_identities = java
                .artifact_identities()
                .map(|identity| {
                    runmat_types::ForeignArtifactIdentity::new(identity.to_string()).map_err(
                        |error| foreign_error(ForeignErrorKind::InvalidCall, error.to_string()),
                    )
                })
                .collect::<Result<_, _>>()?;
        }
        if let (Some(descriptor), Some(python)) =
            (adapters.get_mut(runmat_python::PYTHON_ADAPTER_ID), python)
        {
            descriptor.artifact_identities = python.artifact_identities().cloned().collect();
        }
        self.admit_with_descriptors(manifest, &adapters)
    }

    fn adapter_descriptors(
        &self,
    ) -> BTreeMap<runmat_types::ForeignAdapterId, ForeignAdapterDescriptor> {
        self.adapters
            .borrow()
            .values()
            .map(|adapter| {
                let descriptor = adapter.descriptor();
                (descriptor.adapter.clone(), descriptor)
            })
            .collect()
    }

    fn admit_with_descriptors(
        &self,
        manifest: &InteropManifest,
        adapters: &BTreeMap<runmat_types::ForeignAdapterId, ForeignAdapterDescriptor>,
    ) -> Result<InteropAdmissionPlan, RuntimeError> {
        let plan = admit_interop_manifest(manifest, adapters, self.platform)?;
        for adapter in &plan.adapters {
            let isolated = self
                .adapters
                .borrow()
                .get(adapter)
                .is_some_and(|adapter| adapter.is_isolated());
            self.telemetry.record(ForeignTelemetryEvent {
                adapter: adapter.to_string(),
                operation: "manifest_admission".into(),
                capability: None,
                isolated,
                outcome: ForeignTelemetryOutcome::Admitted,
                error_identifier: None,
            });
        }
        Ok(plan)
    }
}

impl RuntimeForeignService for ForeignRuntime {
    fn visit_retained_values(
        &self,
        visitor: &mut dyn FnMut(&runmat_value::Value) -> Option<()>,
    ) -> Option<()> {
        // Release the adapter table borrow before calling injected adapters.
        let adapters = self
            .adapters
            .try_borrow()
            .ok()?
            .values()
            .cloned()
            .collect::<Vec<_>>();
        for adapter in adapters {
            adapter.visit_retained_values(visitor)?;
        }
        Some(())
    }
    fn execution_stack_requirement(&self) -> runmat_types::ExecutionStackRequirement {
        self.adapters
            .borrow()
            .values()
            .map(|adapter| adapter.descriptor().execution_stack)
            .max()
            .unwrap_or(runmat_types::ExecutionStackRequirement::Any)
    }

    fn invoke(&self, context: RuntimeContext, call: ForeignCall) -> ForeignAdapterFuture {
        if let Some(owner) = context.command_profiling() {
            owner.mark_unproven_device_route();
        }
        runmat_accelerate_api::observe_execution_device_effect(
            runmat_accelerate_api::ExecutionDeviceEffect::ForeignExecution,
        );
        let adapter = self.adapters.borrow().get(call.adapter.as_str()).cloned();
        let telemetry = Rc::clone(&self.telemetry);
        Box::pin(async move {
            let adapter = match adapter {
                Some(adapter) => adapter,
                None => {
                    return ForeignCallCompletion::invoked(Err(foreign_error(
                        ForeignErrorKind::AdapterUnavailable,
                        format!("foreign adapter {} is not registered", call.adapter),
                    )))
                }
            };
            if context.is_cancelled() {
                return ForeignCallCompletion::invoked(Err(foreign_error(
                    ForeignErrorKind::CallbackFailed,
                    format!("foreign call {} was cancelled", call.symbol),
                )));
            }
            for argument in &call.arguments {
                if let Err(error) =
                    runmat_value::validate_no_transient_sequence(argument).map_err(|_| {
                        crate::runtime_error::semantic_error(
                            "TransientSequenceNotPortable",
                            "transient output sequences cannot cross a foreign adapter boundary",
                        )
                    })
                {
                    return ForeignCallCompletion::invoked(Err(error));
                }
            }
            let descriptor = adapter.descriptor();
            if descriptor.execution_stack == runmat_types::ExecutionStackRequirement::Process
                && context.execution_stack() != crate::context::RuntimeExecutionStack::Process
            {
                return ForeignCallCompletion::invoked(Err(foreign_error(
                    ForeignErrorKind::ExecutionStackViolation,
                    format!(
                        "foreign adapter {} requires the process thread stack",
                        descriptor.adapter
                    ),
                )));
            }
            let isolated = adapter.is_isolated();
            let operation = call.symbol.clone();
            let result = context.scope(adapter.invoke(context.clone(), call)).await;
            let (outcome, error_identifier) = match result.invoked_result() {
                Some(Ok(_)) => (ForeignTelemetryOutcome::Succeeded, None),
                Some(Err(error)) => (
                    ForeignTelemetryOutcome::Failed,
                    error.identifier().map(str::to_owned),
                ),
                None => (ForeignTelemetryOutcome::Rejected, None),
            };
            telemetry.record(ForeignTelemetryEvent {
                adapter: descriptor.adapter.to_string(),
                operation,
                capability: Some(runmat_types::ForeignCapability::Invoke),
                isolated,
                outcome,
                error_identifier,
            });
            result
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{RuntimeExecutionStack, RuntimeServicePorts};
    use crate::execution::RuntimeExecutionService;
    use runmat_types::{CapabilityRequirement, ExecutionStackRequirement, ForeignCapability};
    use runmat_value::{CellArray, Value};
    use std::cell::Cell;
    use std::collections::BTreeSet;

    struct ProcessStackAdapter;

    impl ForeignAdapter for ProcessStackAdapter {
        fn descriptor(&self) -> ForeignAdapterDescriptor {
            ForeignAdapterDescriptor {
                adapter: runmat_types::ForeignAdapterId::new("process-stack-test").unwrap(),
                version: 1,
                capabilities: BTreeSet::from([CapabilityRequirement::ForeignRuntime]),
                foreign_capabilities: BTreeSet::from([ForeignCapability::Invoke]),
                artifact_identities: BTreeSet::new(),
                supports_wasm: false,
                supports_host_bridge: false,
                execution_stack: ExecutionStackRequirement::Process,
            }
        }

        fn invoke(&self, _context: RuntimeContext, _call: ForeignCall) -> ForeignAdapterFuture {
            Box::pin(async {
                ForeignCallCompletion::invoked(root_adapter_result(
                    ValueSequence::single(runmat_value::Value::Num(1.0))
                        .map_err(|error| RuntimeError::new(error.to_string())),
                ))
            })
        }
    }

    #[test]
    fn process_stack_adapters_reject_segmented_stack_entry() {
        let foreign = Rc::new(ForeignRuntime::new(ForeignPlatform::Native));
        foreign
            .register_adapter(Rc::new(ProcessStackAdapter))
            .expect("register adapter");
        let services = RuntimeServicePorts::default().with_foreign(foreign.clone());
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
            .with_service_ports(services);
        let _stack = runtime.enter_execution_stack(RuntimeExecutionStack::Segmented);
        let error = futures::executor::block_on(foreign.invoke(
            runtime.clone(),
            ForeignCall {
                adapter: "process-stack-test".into(),
                symbol: "invoke".into(),
                arguments: crate::call::RootedFunctionArguments::ordinary(
                    crate::execution::RootedValues::new(Vec::new()).unwrap(),
                ),
                requested_outputs: 1,
            },
        ))
        .into_invoked_result()
        .expect_err("segmented stack must be rejected");

        assert_eq!(
            error.identifier(),
            Some("RunMat:Foreign:ExecutionStackViolation")
        );
    }

    struct RecordingAdapter {
        invoked: Rc<Cell<bool>>,
    }

    impl ForeignAdapter for RecordingAdapter {
        fn descriptor(&self) -> ForeignAdapterDescriptor {
            ForeignAdapterDescriptor {
                adapter: runmat_types::ForeignAdapterId::new("recording-test").unwrap(),
                version: 1,
                capabilities: BTreeSet::from([CapabilityRequirement::ForeignRuntime]),
                foreign_capabilities: BTreeSet::from([ForeignCapability::Invoke]),
                artifact_identities: BTreeSet::new(),
                supports_wasm: false,
                supports_host_bridge: false,
                execution_stack: ExecutionStackRequirement::Any,
            }
        }

        fn invoke(&self, _context: RuntimeContext, _call: ForeignCall) -> ForeignAdapterFuture {
            self.invoked.set(true);
            Box::pin(async {
                ForeignCallCompletion::invoked(root_adapter_result(Ok(ValueSequence::empty())))
            })
        }
    }

    #[test]
    fn nested_transient_sequences_are_rejected_before_adapter_invocation() {
        let invoked = Rc::new(Cell::new(false));
        let foreign = Rc::new(ForeignRuntime::new(ForeignPlatform::Native));
        foreign
            .register_adapter(Rc::new(RecordingAdapter {
                invoked: Rc::clone(&invoked),
            }))
            .expect("register adapter");
        let services = RuntimeServicePorts::default().with_foreign(foreign.clone());
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
            .with_service_ports(services);
        let nested = Value::Cell(
            CellArray::new(vec![Value::output_list(vec![Value::Num(1.0)])], 1, 1)
                .expect("valid cell fixture"),
        );

        let error = futures::executor::block_on(foreign.invoke(
            runtime,
            ForeignCall {
                adapter: "recording-test".into(),
                symbol: "invoke".into(),
                arguments: crate::call::RootedFunctionArguments::ordinary(
                    crate::execution::RootedValues::new(vec![nested]).unwrap(),
                ),
                requested_outputs: 0,
            },
        ))
        .into_invoked_result()
        .expect_err("nested transient sequence must be rejected");

        assert_eq!(
            error.identifier(),
            Some("RunMat:TransientSequenceNotPortable")
        );
        assert!(
            !invoked.get(),
            "adapter must not observe rejected arguments"
        );
    }
}
