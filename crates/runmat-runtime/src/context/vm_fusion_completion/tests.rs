use super::*;
use crate::context::{RuntimePlacementService, RuntimeServicePorts};
use crate::execution::RuntimeExecutionService;
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::{
    CancellationReason, Digest, PlacementExecutionAuthorityRevision, PlacementRevision,
    ProgramEnvironment,
};
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[cfg_attr(not(target_arch = "wasm32"), test)]
fn general_numeric_host_owners_confirm_single_double_intrinsics_and_calls() {
    use crate::context::{VmFusionHostInstruction as I, VmFusionHostOperation as H};
    use futures::executor::block_on;
    use runmat_types::OperatorKind;
    use runmat_value::{NumericDType, Tensor, Value};
    runmat_gc::gc_test_context(|| {
        for single in [false, true] {
            for case in 0..12 {
                let service = Rc::new(SessionWitness::default());
                let selected = context(Some(service), Some(program(b"general-host-owner")));
                let installed = VmFusionInstalledRegion::install(
                    &selected,
                    semantic(b"numeric-owner"),
                    0,
                    0,
                    0,
                )
                .unwrap();
                let (instruction, operations, name) = match case {
                    0 => (
                        I::Intrinsic(OperatorKind::UnaryMinus),
                        vec![
                            H::Intrinsic(OperatorKind::UnaryMinus),
                            H::Builtin(runmat_builtins::TIMES_CATALOG_ENTRY.binding_identity(
                                &runmat_builtins::TIMES_CATALOG_ENTRY.bindings[0],
                            )),
                        ],
                        None,
                    ),
                    1 => (
                        I::Intrinsic(OperatorKind::MatrixMultiply),
                        vec![H::Intrinsic(OperatorKind::MatrixMultiply)],
                        None,
                    ),
                    _ => {
                        let name = [
                            "times", "sin", "exp", "tanh", "power", "expm1", "log", "log10",
                            "log1p", "log2",
                        ][case - 2];
                        let entry = runmat_builtins::builtin_catalog_entry_by_name(name).unwrap();
                        (
                            if case == 2 || case == 6 {
                                I::ElementwiseBinary
                            } else {
                                I::ElementwiseCall
                            },
                            vec![H::Builtin(entry.binding_identity(&entry.bindings[0]))],
                            Some(name),
                        )
                    }
                };
                let route = installed
                    .right(&selected)
                    .unwrap()
                    .host_route_operations(&selected, operations)
                    .unwrap();
                route.begin_instruction(0, instruction);
                let runtime = selected
                    .clone()
                    .with_vm_fusion_host_route(Some(route.clone()));
                let runtime = runtime.with_pinned_cancellation();
                assert!(runtime.vm_fusion_host_route().is_some());
                let value = Value::Tensor(if single {
                    Tensor::from_f32(vec![0.25, 0.5], vec![1, 2]).unwrap()
                } else {
                    Tensor::new(vec![0.25, 0.5], vec![1, 2]).unwrap()
                });
                let result = match case {
                    0 => block_on(crate::unary_arithmetic::evaluate_in_context(
                        Some(&runtime),
                        crate::unary_arithmetic::UnaryArithmeticOperation::Minus,
                        value,
                    ))
                    .unwrap(),
                    1 => crate::execution::RootedValue::new(
                        block_on(crate::value_matmul_in_context(
                            &runtime,
                            &Value::Num(4.0),
                            &value,
                        ))
                        .unwrap(),
                    )
                    .unwrap(),
                    _ => {
                        let args = if case == 2 || case == 6 {
                            vec![value, Value::Num(2.0)]
                        } else {
                            vec![value]
                        };
                        block_on(crate::call_builtin_async_in_context(
                            &runtime,
                            name.unwrap(),
                            &args,
                        ))
                        .unwrap()
                    }
                };
                let Value::Tensor(output) = result.value() else {
                    panic!("numeric owner must return its real host tensor");
                };
                assert_eq!(
                    output.numeric_dtype(),
                    if single {
                        NumericDType::F32
                    } else {
                        NumericDType::F64
                    }
                );
                for (actual, x) in output.materialize_f64().iter().zip([0.25f64, 0.5]) {
                    let expected = match case {
                        0 => -x,
                        1 => 4.0 * x,
                        2 => 2.0 * x,
                        3 => x.sin(),
                        4 => x.exp(),
                        5 => x.tanh(),
                        6 => x * x,
                        7 => x.exp_m1(),
                        8 => x.ln(),
                        9 => x.log10(),
                        10 => x.ln_1p(),
                        _ => x.log2(),
                    };
                    assert!((actual - expected).abs() < if single { 1e-6 } else { 1e-12 });
                }
                route.instruction_exit(0);
                assert!(
                    route.complete(),
                    "actual numeric owner did not confirm case={case} single={single}"
                );
            }
        }
    });
}

#[derive(Default)]
struct SessionWitness {
    registrations: Cell<usize>,
    wrong_authority: Cell<bool>,
    invalidations: RefCell<Vec<runmat_execution::PlacementInvalidation>>,
    permits: RefCell<BTreeMap<u64, Arc<TestPermit>>>,
}

struct TestPermit(AtomicBool);

impl RuntimeVmFusionGenerationPermit for TestPermit {
    fn is_current(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

impl SessionWitness {
    fn reset_epoch(&self) {
        for permit in self.permits.borrow().values() {
            permit.0.store(false, Ordering::Release);
        }
    }
}

impl RuntimePlacementService for SessionWitness {
    fn plan(
        &self,
        _: runmat_execution::PlacementPlanRequest,
    ) -> Result<runmat_execution::PlacementDecision, crate::RuntimeError> {
        unreachable!("VM right fixture provides revision identity only")
    }

    fn observe(&self, _: runmat_execution::PlacementFeedback) -> Result<(), crate::RuntimeError> {
        unreachable!("VM right fixture does not publish placement feedback")
    }

    fn invalidate(&self, invalidation: runmat_execution::PlacementInvalidation) {
        // RuntimeContext::with_program_revision invalidates the previous
        // program even for a revision-only placement service. Record that
        // canonical notification without pretending this fixture owns a cache.
        self.invalidations.borrow_mut().push(invalidation);
    }

    fn whole_work_vm_fusion_region_session_revision(
        &self,
        program: Option<ProgramRevision>,
        region: VmFusionRegionRegistration,
        liveness: Weak<()>,
    ) -> Option<VmFusionSessionRegistration> {
        liveness.upgrade()?;
        self.registrations.set(self.registrations.get() + 1);
        let authority = Digest::sha256_serialized(&(
            "vm-right-test-semantic-v2",
            region.semantic_identity().digest(),
        ))
        .unwrap();
        let revision = PlacementRevision {
            program,
            catalog: Digest::sha256(b"vm-right-test-catalog"),
            compiler: Digest::sha256(b"vm-right-test-compiler"),
            authority: if self.wrong_authority.get() {
                PlacementExecutionAuthorityRevision::Provider(authority)
            } else {
                PlacementExecutionAuthorityRevision::RuntimeBindingSession(authority)
            },
            policy: Digest::sha256(b"vm-right-test-policy"),
        };
        let mut permits = self.permits.borrow_mut();
        let permit = permits
            .entry(region.installation())
            .or_insert_with(|| Arc::new(TestPermit(AtomicBool::new(true))));
        if !permit.is_current() {
            *permit = Arc::new(TestPermit(AtomicBool::new(true)));
        }
        Some(VmFusionSessionRegistration::new(revision, permit.clone()))
    }
}

fn semantic(label: &[u8]) -> VmFusionSemanticIdentity {
    VmFusionSemanticIdentity::from_validated_digest(Digest::sha256(label))
}

fn program(label: &[u8]) -> ProgramRevision {
    ProgramRevision::new(
        Digest::sha256(label),
        Digest::sha256(b"vm-right-test-source"),
        ProgramEnvironment::new(
            2,
            2,
            Digest::sha256(b"vm-right-test-runtime"),
            Digest::sha256(b"vm-right-test-builtins"),
            "runmat",
        )
        .unwrap(),
    )
    .unwrap()
}

fn context(
    service: Option<Rc<SessionWitness>>,
    program: Option<ProgramRevision>,
) -> RuntimeContext {
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let runtime = match service {
        Some(service) => {
            runtime.with_service_ports(RuntimeServicePorts::default().with_placement(service))
        }
        None => runtime,
    };
    runtime.with_program_revision(program)
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn installed_right_is_exact_to_context_program_region_and_cancellation_generation() {
    let service = Rc::new(SessionWitness::default());
    let selected = context(Some(service.clone()), Some(program(b"one")));
    let projected = selected.clone();
    let unchanged = projected
        .clone()
        .with_program_revision(Some(program(b"one")));
    let original_revision = selected.program_revision_owner().unwrap();
    assert!(Arc::ptr_eq(
        original_revision,
        projected.program_revision_owner().unwrap()
    ));
    assert!(Arc::ptr_eq(
        original_revision,
        unchanged.program_revision_owner().unwrap()
    ));
    assert!(service.invalidations.borrow().is_empty());
    assert!(VmFusionInstalledRegion::install(&selected, semantic(b"same"), 7, 9, 8).is_none());
    let installed =
        VmFusionInstalledRegion::install(&selected, semantic(b"same"), 7, 8, 9).unwrap();
    let other_installation =
        VmFusionInstalledRegion::install(&selected, semantic(b"same"), 7, 8, 9).unwrap();
    assert!(installed.matches_group(7, 8, 9));
    assert!(!installed.matches_group(8, 8, 9));
    assert!(!installed.matches_group(7, 8, 10));
    assert!(installed.registration != other_installation.registration);
    let right = installed.right(&selected).unwrap();
    let second = other_installation.right(&selected).unwrap();
    let decline = installed.decline_right(&selected).unwrap();
    for retained in [
        installed.program.as_ref(),
        right.program.as_ref(),
        decline.program.as_ref(),
    ] {
        assert!(Arc::ptr_eq(original_revision, retained.unwrap()));
    }
    assert_eq!(
        decline.semantic_identity(&selected),
        Some(semantic(b"same"))
    );
    let registrations_before_join = service.registrations.get();
    assert!(decline.same_installed_region(&right, &selected));
    assert!(!decline.same_installed_region(&second, &selected));
    assert_eq!(service.registrations.get(), registrations_before_join);
    assert_eq!(right.placement_revision(), second.placement_revision());
    assert_eq!(right.registration(), installed.registration);
    let registrations_before_sizing = service.registrations.get();
    assert_eq!(right.held_domain_contributions_for_sizing(), Some(&[][..]));
    assert_eq!(service.registrations.get(), registrations_before_sizing);
    assert!(right.planning_revision(&selected).is_some());
    assert!(service.registrations.get() > registrations_before_sizing);
    assert!(right.completion_lease(&selected).is_some());
    assert!(second.completion_lease(&selected).is_some());

    let foreign = context(Some(service.clone()), Some(program(b"one")));
    assert_ne!(selected.local_identity(), foreign.local_identity());
    assert!(decline.semantic_identity(&foreign).is_none());
    assert!(!decline.same_installed_region(&installed.right(&selected).unwrap(), &foreign));
    assert!(installed.right(&foreign).is_none());
    assert!(installed
        .right(&selected)
        .unwrap()
        .completion_lease(&foreign)
        .is_none());

    let changed_program = selected
        .clone()
        .with_program_revision(Some(program(b"two")));
    assert_eq!(
        *service.invalidations.borrow(),
        vec![runmat_execution::PlacementInvalidation::Program {
            revision: program(b"one"),
        }]
    );
    assert!(installed.right(&changed_program).is_none());
    assert!(installed
        .right(&selected)
        .unwrap()
        .completion_lease(&changed_program)
        .is_none());

    let next_cancellation = CancellationSource::new();
    let changed_generation = selected
        .clone()
        .with_cancellation_token(next_cancellation.token());
    assert!(installed.right(&changed_generation).is_none());
    assert!(installed
        .right(&selected)
        .unwrap()
        .completion_lease(&changed_generation)
        .is_none());
    assert!(service.registrations.get() >= 3);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn deferred_lease_survives_owner_drop_without_minting_new_rights() {
    let service = Rc::new(SessionWitness::default());
    let selected = context(Some(service.clone()), Some(program(b"deferred")));
    let installed =
        VmFusionInstalledRegion::install(&selected, semantic(b"deferred"), 4, 20, 22).unwrap();
    let first = installed.right(&selected).unwrap();
    let second = installed.right(&selected).unwrap();
    let third = installed.right(&selected).unwrap();
    let deferred = first.completion_lease(&selected).unwrap();
    drop(installed);
    assert!(deferred.is_current());
    assert!(second.placement_revision().is_none());
    assert!(third.placement_revision().is_none());
    assert!(second.planning_revision(&selected).is_none());
    assert!(third.planning_revision(&selected).is_none());
    assert!(second.completion_lease(&selected).is_none());
    assert!(third.completion_lease(&selected).is_none());
    drop(deferred);
    assert_eq!(service.registrations.get(), 4);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn missing_stale_or_wrong_session_revision_never_yields_completion_lease() {
    let without_service = context(None, Some(program(b"missing")));
    let installed =
        VmFusionInstalledRegion::install(&without_service, semantic(b"missing"), 1, 0, 1).unwrap();
    let right = installed.right(&without_service).unwrap();
    assert!(right.placement_revision().is_none());
    assert!(right.completion_lease(&without_service).is_none());

    let service = Rc::new(SessionWitness::default());
    let selected = context(Some(service.clone()), Some(program(b"revision")));
    let installed =
        VmFusionInstalledRegion::install(&selected, semantic(b"revision"), 1, 0, 1).unwrap();
    service.wrong_authority.set(true);
    let wrong = installed.right(&selected).unwrap();
    assert!(wrong.placement_revision().is_none());
    assert!(wrong.completion_lease(&selected).is_none());
    service.wrong_authority.set(false);
    let valid = installed.right(&selected).unwrap();
    assert!(valid.placement_revision().is_some());
    let decline = installed.decline_right(&selected).unwrap();
    assert!(decline.same_installed_region(&valid, &selected));
    service.reset_epoch();
    assert!(!decline.same_installed_region(&valid, &selected));
    assert!(valid.completion_lease(&selected).is_none());
    assert!(installed
        .right(&selected)
        .unwrap()
        .placement_revision()
        .is_some());

    let cancelled = CancellationSource::new();
    let cancellable =
        context(Some(service.clone()), None).with_cancellation_token(cancelled.token());
    let installation =
        VmFusionInstalledRegion::install(&cancellable, semantic(b"cancel"), 2, 1, 2).unwrap();
    let right = installation.right(&cancellable).unwrap();
    cancelled.cancel(CancellationReason::User);
    assert!(installation.right(&cancellable).is_none());
    assert!(right.completion_lease(&cancellable).is_none());
    assert!(VmFusionInstalledRegion::install(&cancellable, semantic(b"cancel"), 2, 1, 2).is_none());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn host_route_requires_exact_selected_binding_and_one_instruction_exit() {
    use crate::context::VmFusionHostInstruction;
    use runmat_builtins::{BuiltinBindingIdentity, BuiltinCatalogIdentity};

    let service = Rc::new(SessionWitness::default());
    let selected = context(Some(service.clone()), Some(program(b"host-route")));
    let installed =
        VmFusionInstalledRegion::install(&selected, semantic(b"host-route"), 3, 10, 12).unwrap();
    let exact = BuiltinBindingIdentity {
        builtin: BuiltinCatalogIdentity { name: "plus" },
        variant: "default",
    };
    let foreign = BuiltinBindingIdentity {
        builtin: BuiltinCatalogIdentity { name: "plus" },
        variant: "alternate",
    };
    let terminal = BuiltinBindingIdentity {
        builtin: BuiltinCatalogIdentity { name: "min" },
        variant: "default",
    };
    let right = installed.right(&selected).unwrap();
    let route = right.host_route(&selected, vec![exact, terminal]).unwrap();
    assert!(!route.complete(), "no instruction has executed");
    assert!(!route.contains_pc(9));
    assert!(route.contains_pc(10));
    route.begin_instruction(10, VmFusionHostInstruction::Structural);
    route.instruction_exit(10);
    route.begin_instruction(11, VmFusionHostInstruction::ElementwiseBinary);
    let scoped = selected
        .clone()
        .with_vm_fusion_host_route(Some(route.clone()));
    assert!(route.allows_host_promotion(&scoped));
    let foreign_placement = selected.clone().with_service_ports(
        RuntimeServicePorts::default().with_placement(Rc::new(SessionWitness::default())),
    );
    assert!(!route.current_for(&foreign_placement));
    let next_cancellation = CancellationSource::new();
    let next_generation = selected
        .clone()
        .with_cancellation_token(next_cancellation.token());
    assert!(!route.current_for(&next_generation));
    let wrong = scoped
        .clone()
        .with_vm_fusion_selected_binding(Some(foreign));
    assert!(!route.allows_host_preparation(&wrong, VmFusionHostInstruction::ElementwiseBinary));
    let exact_context = scoped.clone().with_vm_fusion_selected_binding(Some(exact));
    assert!(
        route.allows_host_preparation(&exact_context, VmFusionHostInstruction::ElementwiseBinary)
    );
    let inner = exact_context
        .clone()
        .with_vm_fusion_selected_binding(Some(exact));
    assert!(inner.vm_fusion_host_route().is_none());
    route.confirm_host_branch(&exact_context, VmFusionHostInstruction::ElementwiseBinary);
    route.instruction_exit(11);
    assert!(!route.complete(), "terminal reduction has not executed");
    route.begin_instruction(12, VmFusionHostInstruction::Reduction);
    let terminal_context = scoped.with_vm_fusion_selected_binding(Some(terminal));
    route.confirm_host_branch(&terminal_context, VmFusionHostInstruction::Reduction);
    route.instruction_exit(12);
    assert!(route.complete());
    assert!(right.completion_lease(&selected).is_some());

    let stale = installed.right(&selected).unwrap();
    let stale_route = stale.host_route(&selected, vec![exact]).unwrap();
    stale_route.begin_instruction(10, VmFusionHostInstruction::ElementwiseBinary);
    stale_route.confirm_host_branch(&wrong, VmFusionHostInstruction::ElementwiseBinary);
    stale_route.instruction_exit(10);
    assert!(!stale_route.complete());
    let missing = installed.right(&selected).unwrap();
    let missing_route = missing.host_route(&selected, vec![exact]).unwrap();
    missing_route.begin_instruction(10, VmFusionHostInstruction::ElementwiseBinary);
    missing_route.instruction_exit(10);
    assert!(!missing_route.complete());
    assert!(missing.completion_lease(&selected).is_some());
    let skipped = installed.right(&selected).unwrap();
    let skipped_route = skipped.host_route(&selected, vec![exact]).unwrap();
    skipped_route.begin_instruction(11, VmFusionHostInstruction::ElementwiseBinary);
    skipped_route.confirm_host_branch(&exact_context, VmFusionHostInstruction::ElementwiseBinary);
    skipped_route.instruction_exit(11);
    assert!(
        !skipped_route.complete(),
        "a skipped PC cannot certify a region"
    );
    let no_terminal = installed.right(&selected).unwrap();
    let no_terminal_route = no_terminal
        .host_route(&selected, vec![exact, terminal])
        .unwrap();
    no_terminal_route.begin_instruction(10, VmFusionHostInstruction::Structural);
    no_terminal_route.instruction_exit(10);
    no_terminal_route.begin_instruction(11, VmFusionHostInstruction::ElementwiseBinary);
    no_terminal_route
        .confirm_host_branch(&exact_context, VmFusionHostInstruction::ElementwiseBinary);
    no_terminal_route.instruction_exit(11);
    no_terminal_route.begin_instruction(12, VmFusionHostInstruction::Reduction);
    no_terminal_route.instruction_exit(12);
    assert!(
        !no_terminal_route.complete(),
        "unconfirmed terminal cannot certify a region"
    );
    let unsupported = installed.right(&selected).unwrap();
    let unsupported_route = unsupported.host_route(&selected, vec![exact]).unwrap();
    unsupported_route.begin_instruction(10, VmFusionHostInstruction::Unproved);
    unsupported_route.instruction_exit(10);
    assert!(!unsupported_route.complete());
    service.reset_epoch();
    assert!(!route.complete());
    assert!(!stale_route.complete());
}
