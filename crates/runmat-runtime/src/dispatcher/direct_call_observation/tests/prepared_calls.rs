//! Runtime-only lifecycle probes; the accelerator's provider receipt tests
//! separately certify physical route completion and total construction.
use super::*;
use crate::context::preparation::{
    AdmittedBuiltinPreparation, BuiltinPreparationObservation, PreparedBuiltinOperands,
    PreparedCallCompletion, PreparedCallStatus, PreparedCompletionRight,
};
use crate::context::{RuntimeAccelerationService, RuntimeServiceFuture, RuntimeWorkspaceService};
use runmat_value::Tensor;

struct Ticket {
    binding: runmat_builtins::BuiltinBindingIdentity,
    status: PreparedCallStatus,
    published: Rc<RefCell<Vec<PreparedCallStatus>>>,
}

impl PreparedCallCompletion for Ticket {
    fn binding(&self) -> runmat_builtins::BuiltinBindingIdentity {
        self.binding
    }

    fn status(&self) -> PreparedCallStatus {
        self.status
    }

    fn publish(self: Box<Self>, _outer_started: Instant, elapsed_ns: u64) {
        assert!(elapsed_ns > 0);
        self.published.borrow_mut().push(self.status);
    }
}

struct PreparedGuard {
    right: PreparedCompletionRight,
    published: Rc<RefCell<Vec<PreparedCallStatus>>>,
    fail_cleanup: bool,
}

impl BuiltinPreparationObservation for PreparedGuard {
    fn finish(
        self: Box<Self>,
        body: Result<&Value, &crate::RuntimeError>,
    ) -> Result<(), crate::RuntimeError> {
        let Self {
            right,
            published,
            fail_cleanup,
        } = *self;
        let binding = right.binding();
        let status = if body.is_ok() && !fail_cleanup {
            PreparedCallStatus::Succeeded
        } else {
            PreparedCallStatus::CompletedFailure
        };
        right.deposit(Box::new(Ticket {
            binding,
            status,
            published,
        }));
        if fail_cleanup {
            Err(
                crate::build_runtime_error("prepared cleanup fixture failed")
                    .with_identifier("RunMat:PreparedCleanupFixture")
                    .build(),
            )
        } else {
            Ok(())
        }
    }

    fn abort(self: Box<Self>) {}
}

struct AdmittedService {
    published: Rc<RefCell<Vec<PreparedCallStatus>>>,
    prepared: Rc<Cell<usize>>,
    fail_body: bool,
    fail_cleanup: bool,
}

impl RuntimeAccelerationService for AdmittedService {
    fn prepare_admitted(
        &self,
        _context: RuntimeContext,
        mut request: AdmittedBuiltinPreparation,
    ) -> RuntimeServiceFuture<Result<PreparedBuiltinOperands, crate::RuntimeError>> {
        self.prepared.set(self.prepared.get() + 1);
        let right = request.take_prepared_completion_right();
        let fail_body = self.fail_body;
        let guard = right.map(|right| {
            Box::new(PreparedGuard {
                right,
                published: self.published.clone(),
                fail_cleanup: self.fail_cleanup,
            }) as Box<dyn BuiltinPreparationObservation>
        });
        Box::pin(async move {
            request
                .into_operands()
                .with_unbounded_values(|mut operands| {
                    if fail_body {
                        // Admission used the original numeric argument. This service
                        // fault forces the actual prepared body to return an error.
                        operands[0] = Value::String("body fixture invalid operand".into());
                    }
                    PreparedBuiltinOperands::new(operands, guard)
                })?
        })
    }
}

struct Fence {
    published: Rc<RefCell<Vec<PreparedCallStatus>>>,
    checkpoints: Rc<Cell<usize>>,
    fail: bool,
    on_checkpoint: Option<Box<dyn Fn()>>,
}

impl RuntimeWorkspaceService for Fence {
    fn lookup(&self, _: &str) -> Option<Value> {
        None
    }
    fn snapshot(&self) -> Vec<(String, Value)> {
        Vec::new()
    }
    fn global_names(&self) -> Vec<String> {
        Vec::new()
    }

    fn assign(&self, _: &str, _: Value) -> Result<(), crate::RuntimeError> {
        panic!("prepared-call fence fixture does not support workspace writes")
    }

    fn clear(&self) -> Result<(), crate::RuntimeError> {
        panic!("prepared-call fence fixture does not support workspace writes")
    }

    fn remove(&self, _: &str) -> Result<(), crate::RuntimeError> {
        panic!("prepared-call fence fixture does not support workspace writes")
    }

    fn ensure_effect_authority(&self) -> Result<(), crate::RuntimeError> {
        assert!(
            self.published.borrow().is_empty(),
            "feedback preceded the outer fence"
        );
        self.checkpoints.set(self.checkpoints.get() + 1);
        if let Some(callback) = &self.on_checkpoint {
            callback();
        }
        if self.fail {
            Err(crate::build_runtime_error("workspace fixture revoked")
                .with_identifier("RunMat:Workspace:FixtureRevoked")
                .build())
        } else {
            Ok(())
        }
    }
}

fn reduction_binding(name: &str) -> RuntimeBuiltinBinding {
    crate::builtins::wasm_registry::register_all();
    crate::builtin::runtime_builtin_bindings_by_name(name)[0]
}

struct PreparedRuntime {
    runtime: RuntimeContext,
    binding: RuntimeBuiltinBinding,
    published: Rc<RefCell<Vec<PreparedCallStatus>>>,
    prepared: Rc<Cell<usize>>,
    checkpoints: Rc<Cell<usize>>,
    scoped: Rc<Scoped>,
}

fn prepared_runtime(
    name: &str,
    fail_body: bool,
    fail_cleanup: bool,
    fail_fence: bool,
    on_checkpoint: Option<Box<dyn Fn()>>,
) -> PreparedRuntime {
    let binding = reduction_binding(name);
    let (runtime, scoped, _placement) = harness(binding);
    let published = Rc::new(RefCell::new(Vec::new()));
    let prepared = Rc::new(Cell::new(0));
    let checkpoints = Rc::new(Cell::new(0));
    let ports = runtime
        .service_ports()
        .clone()
        .with_acceleration(Rc::new(AdmittedService {
            published: published.clone(),
            prepared: prepared.clone(),
            fail_body,
            fail_cleanup,
        }))
        .with_workspace(Rc::new(Fence {
            published: published.clone(),
            checkpoints: checkpoints.clone(),
            fail: fail_fence,
            on_checkpoint,
        }));
    PreparedRuntime {
        runtime: runtime.with_service_ports(ports),
        binding,
        published,
        prepared,
        checkpoints,
        scoped,
    }
}

fn arguments() -> [Value; 1] {
    [Value::Tensor(
        Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap(),
    )]
}

fn call_outer_route(
    runtime: &RuntimeContext,
    binding: RuntimeBuiltinBinding,
    route: usize,
) -> Result<crate::execution::RootedValueSequence, crate::RuntimeError> {
    let args = arguments();
    let name = binding.identity.builtin.name;
    match route {
        0 => futures::executor::block_on(
            crate::call_builtin_sequence_async_with_outputs_in_context(runtime, name, &args, 1),
        ),
        1 => futures::executor::block_on(
            crate::dispatcher::try_call_builtin_sequence_async_with_outputs(
                name,
                &args,
                1,
                Some(runtime),
            ),
        )
        .map(|found| found.expect("selected optional binding")),
        2 => futures::executor::block_on(
            crate::dispatcher::call_selected_builtin_async_with_outputs(
                binding,
                &args,
                1,
                Some(runtime.clone()),
            ),
        ),
        _ => unreachable!("three outer builtin routes"),
    }
}

mod lifecycle;
mod outcomes;
