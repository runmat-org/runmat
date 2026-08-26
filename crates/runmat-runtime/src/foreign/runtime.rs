use std::cell::RefCell;
use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;

use runmat_types::InteropManifest;
use runmat_value::Value;

use super::{
    admit_interop_manifest, foreign_error, ForeignAdapterDescriptor, ForeignErrorKind,
    ForeignHandleRegistry, ForeignPlatform, ForeignTelemetryEvent, ForeignTelemetryOutcome,
    ForeignTelemetrySink, InteropAdmissionPlan, NoopForeignTelemetry,
};
use crate::context::{ForeignCall, RuntimeContext, RuntimeForeignService};
use crate::RuntimeError;

pub type ForeignAdapterFuture =
    Pin<Box<dyn Future<Output = Result<Value, RuntimeError>> + 'static>>;

pub trait ForeignAdapter {
    fn descriptor(&self) -> ForeignAdapterDescriptor;
    fn invoke(&self, context: RuntimeContext, call: ForeignCall) -> ForeignAdapterFuture;
    fn is_isolated(&self) -> bool {
        false
    }
}

#[derive(Clone)]
pub struct ForeignRuntime {
    handles: ForeignHandleRegistry,
    adapters: Rc<RefCell<BTreeMap<String, Rc<dyn ForeignAdapter>>>>,
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

    pub fn register_adapter(&self, adapter: Rc<dyn ForeignAdapter>) -> Result<(), RuntimeError> {
        let descriptor = adapter.descriptor();
        if descriptor.adapter.trim().is_empty() || descriptor.version == 0 {
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
        let adapters = self
            .adapters
            .borrow()
            .values()
            .map(|adapter| {
                let descriptor = adapter.descriptor();
                (descriptor.adapter.clone(), descriptor)
            })
            .collect::<BTreeMap<_, _>>();
        let plan = admit_interop_manifest(manifest, &adapters, self.platform)?;
        for adapter in &plan.adapters {
            let isolated = self
                .adapters
                .borrow()
                .get(adapter)
                .is_some_and(|adapter| adapter.is_isolated());
            self.telemetry.record(ForeignTelemetryEvent {
                adapter: adapter.clone(),
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
    fn invoke(&self, context: RuntimeContext, call: ForeignCall) -> ForeignAdapterFuture {
        let adapter = self.adapters.borrow().get(&call.adapter).cloned();
        let telemetry = Rc::clone(&self.telemetry);
        Box::pin(async move {
            let adapter = adapter.ok_or_else(|| {
                foreign_error(
                    ForeignErrorKind::AdapterUnavailable,
                    format!("foreign adapter {} is not registered", call.adapter),
                )
            })?;
            if context
                .cancellation()
                .load(std::sync::atomic::Ordering::SeqCst)
            {
                return Err(foreign_error(
                    ForeignErrorKind::CallbackFailed,
                    format!("foreign call {} was cancelled", call.symbol),
                ));
            }
            let descriptor = adapter.descriptor();
            let isolated = adapter.is_isolated();
            let operation = call.symbol.clone();
            let result = context.scope(adapter.invoke(context.clone(), call)).await;
            telemetry.record(ForeignTelemetryEvent {
                adapter: descriptor.adapter,
                operation,
                capability: Some(runmat_types::ForeignCapability::Invoke),
                isolated,
                outcome: if result.is_ok() {
                    ForeignTelemetryOutcome::Succeeded
                } else {
                    ForeignTelemetryOutcome::Failed
                },
                error_identifier: result
                    .as_ref()
                    .err()
                    .and_then(|error| error.identifier().map(str::to_owned)),
            });
            result
        })
    }
}
