use super::*;
use crate::context::host_effect::ManagedHostEffectLedger;
use crate::context::{
    ParallelCapability, RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
};
use std::rc::Rc;
struct Parallel(ManagedHostEffectLedger);
impl RuntimeParallelService for Parallel {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }
    fn placement_resources(&self) -> RuntimeParallelResources {
        let state = self.0.snapshot();
        RuntimeParallelResources {
            memory_available_bytes: Some(state.free_bytes()),
            epoch: state.epoch,
            ..RuntimeParallelResources::default()
        }
    }
    fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
        Some(self.0.clone())
    }
}
fn managed_runtime(ledger: &ManagedHostEffectLedger) -> RuntimeContext {
    RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default().with_parallel(Rc::new(Parallel(ledger.clone()))),
        )
}
#[test]
fn exact_scratch_is_charged_only_while_live_and_owner_generation_is_not_interchangeable() {
    let ledger = ManagedHostEffectLedger::new(65_536);
    let runtime = managed_runtime(&ledger);
    let owner = ValidationWorkspaceOwner::capture(&runtime);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    owner
        .with_f64_scratch(128, |scratch| {
            assert_eq!(scratch.len(), 128);
            assert!(ledger.snapshot().live_claimed_bytes >= 1024);
            scratch[0] = 3.0;
        })
        .unwrap();
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    assert!(owner.same_owner(&owner.clone()));
    runtime.invalidate_host_effect_session_authority();
    let replacement = ValidationWorkspaceOwner::capture(&runtime);
    assert!(!owner.same_owner(&replacement));
    assert_eq!(
        owner.with_f64_scratch(1, |_| ()),
        Err(HostAllocationError::StaleAuthority)
    );
}
#[test]
fn captured_cancellation_and_denial_are_deferred_without_upgrading_from_later_context() {
    use runmat_execution::cancellation::CancellationSource;
    let source = CancellationSource::new();
    let ledger = ManagedHostEffectLedger::new(65_536);
    let runtime = managed_runtime(&ledger).with_cancellation_token(source.token());
    let owner = ValidationWorkspaceOwner::capture(&runtime);
    source.cancel(runmat_execution::CancellationReason::Deadline);
    assert_eq!(
        owner.with_f64_scratch(1, |_| ()),
        Err(HostAllocationError::Cancelled)
    );
    // Existing no-scratch schemas do not acquire a new admission obligation.
    static PLAIN_SCHEMA: std::sync::Once = std::sync::Once::new();
    const CLASS: &str = "runmat.test.NoScratchDomainControl";
    PLAIN_SCHEMA.call_once(|| {
        super::super::register_immutable_value_class(super::super::ImmutableValueClassSchema {
            class_name: CLASS,
            type_identity: "runmat.test.no-scratch-domain-control",
            schema_version: 1,
            allowed_fields: &[],
            required_fields: &[],
            validate: |_| Ok(()),
        })
        .unwrap()
    });
    let plain = runmat_value::UnboundedObjectBuilder::new(CLASS).finish();
    let schema = super::super::immutable_class::schema_for_class(CLASS)
        .unwrap()
        .unwrap();
    assert!(schema.validate_object(&plain, "$", None).is_ok());
    assert!(schema.validate_object(&plain, "$", Some(&owner)).is_ok());
    let fresh = runtime.with_cancellation_token(CancellationSource::new().token());
    let fresh_owner = ValidationWorkspaceOwner::capture(&fresh);
    assert!(!owner.same_owner(&fresh_owner));
    assert_eq!(
        owner.with_f64_scratch(1, |_| ()),
        Err(HostAllocationError::Cancelled)
    );
    fresh.set_error_namespace("OriginalDomains");
    let error = crate::builtins::math::optim::integer_constraint::validation_error(
        Some(&fresh),
        super::super::ValueCodecError::Allocation {
            path: "$".into(),
            source: HostAllocationError::Cancelled,
        },
    );
    assert_eq!(error.identifier(), Some("OriginalDomains:Cancelled"));
    assert!(matches!(
        std::error::Error::source(&error)
            .and_then(|source| source.downcast_ref::<crate::execution::ExecutionServiceError>()),
        Some(crate::execution::ExecutionServiceError::Cancelled)
    ));
    let small = managed_runtime(&ManagedHostEffectLedger::new(0));
    let denied = ValidationWorkspaceOwner::capture(&small);
    assert!(denied.with_f64_scratch(1, |_| ()).is_err());
}
