use std::rc::Rc;

use runmat_execution::LanguageCompatibilityMode;
use runmat_runtime::context::{
    ForeignCall, ForeignCallCompletion, RuntimeContext, RuntimeForeignService, RuntimeLanguageMode,
    RuntimeServiceFuture, RuntimeServicePorts,
};

use super::{selected_unit_runtime, validate_selected_port_count};

struct ForeignFixture;

impl RuntimeForeignService for ForeignFixture {
    fn invoke(
        &self,
        _context: RuntimeContext,
        _call: ForeignCall,
    ) -> RuntimeServiceFuture<ForeignCallCompletion> {
        Box::pin(async {
            ForeignCallCompletion::invoked(
                runmat_runtime::execution::RootedValueSequence::new(
                    runmat_value::ValueSequence::empty(),
                )
                .map_err(|error| error.into_runtime_error()),
            )
        })
    }
}

fn runtime_with_ambient_foreign() -> RuntimeContext {
    RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(RuntimeServicePorts::default().with_foreign(Rc::new(ForeignFixture)))
}

fn declared_interop() -> runmat_types::InteropManifest {
    let mut declared = runmat_types::InteropManifest::empty();
    declared
        .foreign_types
        .push(runmat_types::ForeignRequirement {
            type_identity: runmat_types::ForeignTypeIdentity {
                family: "test".into(),
                name: "fixture".into(),
                version: 1,
            },
            ownership: runmat_types::ForeignOwnership::Owned,
            affinity: runmat_types::ForeignAffinity::AnyThread,
            lifetime: runmat_types::ForeignLifetime::Call,
            capabilities: vec![runmat_types::ForeignCapability::Invoke],
            wasm: runmat_types::WasmInteropPolicy::Reject,
        });
    declared
}

#[test]
fn selected_ports_change_only_the_exact_units_foreign_owner() {
    let base = runtime_with_ambient_foreign();
    let old: Rc<dyn RuntimeForeignService> = Rc::new(ForeignFixture);
    let new: Rc<dyn RuntimeForeignService> = Rc::new(ForeignFixture);
    let ports = [Some(Rc::clone(&old)), Some(Rc::clone(&new))];
    let interop = declared_interop();
    let old_context = selected_unit_runtime(
        &base,
        Some(&ports),
        0,
        &interop,
        LanguageCompatibilityMode::Matlab,
    )
    .unwrap();
    let new_context = selected_unit_runtime(
        &base,
        Some(&ports),
        1,
        &interop,
        LanguageCompatibilityMode::Strict,
    )
    .unwrap();
    assert_eq!(old_context.local_identity(), base.local_identity());
    assert_eq!(new_context.local_identity(), base.local_identity());
    assert!(Rc::ptr_eq(
        old_context.service_ports().foreign().unwrap(),
        &old
    ));
    assert!(Rc::ptr_eq(
        new_context.service_ports().foreign().unwrap(),
        &new
    ));
    assert!(!Rc::ptr_eq(&old, &new));
    assert_eq!(old_context.language_mode(), RuntimeLanguageMode::Matlab);
    assert_eq!(new_context.language_mode(), RuntimeLanguageMode::Strict);
    assert!(Rc::ptr_eq(
        old_context.service_ports().timing().unwrap(),
        base.service_ports().timing().unwrap()
    ));
    assert!(Rc::ptr_eq(
        new_context.service_ports().timing().unwrap(),
        base.service_ports().timing().unwrap()
    ));
}

#[test]
fn absent_unit_port_cannot_inherit_ambient_foreign_service() {
    let base = runtime_with_ambient_foreign();
    let empty = runmat_types::InteropManifest::empty();
    let selected = selected_unit_runtime(
        &base,
        Some(&[None]),
        0,
        &empty,
        LanguageCompatibilityMode::RunMat,
    )
    .unwrap();
    assert!(selected.service_ports().foreign().is_none());
    assert!(base.service_ports().foreign().is_some());

    assert!(selected_unit_runtime(
        &base,
        Some(&[None]),
        0,
        &declared_interop(),
        LanguageCompatibilityMode::RunMat,
    )
    .is_err());
    let extra: Rc<dyn RuntimeForeignService> = Rc::new(ForeignFixture);
    assert!(selected_unit_runtime(
        &base,
        Some(&[Some(extra)]),
        0,
        &empty,
        LanguageCompatibilityMode::RunMat,
    )
    .is_err());
}

#[test]
fn selected_unit_mode_overrides_primary_without_mutating_host_policy() {
    let base = runtime_with_ambient_foreign();
    base.set_language_mode(RuntimeLanguageMode::Matlab);
    base.set_runmat_extensions_enabled(false);
    let empty = runmat_types::InteropManifest::empty();
    let selected =
        selected_unit_runtime(&base, None, 0, &empty, LanguageCompatibilityMode::RunMat).unwrap();
    assert_eq!(selected.language_mode(), RuntimeLanguageMode::RunMat);
    assert!(selected.runmat_extensions_enabled());
    assert_eq!(base.language_mode(), RuntimeLanguageMode::Matlab);
    assert!(!base.runmat_extensions_enabled());
    assert_eq!(selected.local_identity(), base.local_identity());
}

#[test]
fn selected_foreign_ports_require_one_position_per_v3_unit() {
    assert!(validate_selected_port_count(1, Some(&[None])).is_err());
    assert!(validate_selected_port_count(1, Some(&[None, None])).is_ok());
    assert!(validate_selected_port_count(1, None).is_ok());
}
