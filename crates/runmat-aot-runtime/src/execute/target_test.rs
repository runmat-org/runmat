//! Test-only selection, input decoding, and managed receiver reporting.
//! The ordinary AOT host still owns the single setup/execution/cleanup path.

use std::{
    rc::Rc,
    time::{Duration, Instant},
};

use runmat_execution::value::ValueLimits;
use runmat_execution_artifact::target_test::{
    NativePairedProductHeader, NativePairedProductRole, NativeTargetTestRequest,
    NativeTargetTestResult, NativeTargetTestStatus, NativeTargetTestUsage,
    NATIVE_TARGET_TEST_PROTOCOL_SCHEMA_VERSION,
};
use runmat_native_executor::{execute::RootedNativeExecution, NativeExecutor};
use runmat_runtime::{context::host_effect::ManagedHostEffectLedger, RuntimeError};
use runmat_types::FunctionId;

#[path = "target_test/failure.rs"]
mod failure;
pub(super) use failure::classify_failure;
use failure::{TargetTestInputDecodeFailure, TargetTestTimeout};

pub(super) struct TestInvocation<'a> {
    pub header: &'a NativePairedProductHeader,
    pub request: &'a NativeTargetTestRequest,
    pub function: FunctionId,
}

impl<'a> TestInvocation<'a> {
    pub fn select(
        header: &'a NativePairedProductHeader,
        request: &'a NativeTargetTestRequest,
        assembly: &runmat_native_codegen::NativeAssembly,
        program: &runmat_native_codegen::aot::AotProgramManifest,
    ) -> Result<Self, String> {
        if header.role != NativePairedProductRole::Test {
            return Err("paired native image is not a test product".into());
        }
        runmat_native_codegen::aot::validate_paired_product_header(header, assembly, program)
            .map_err(|error| format!("paired linked test admission failed: {error}"))?;
        request
            .validate_for(&header.binding)
            .map_err(|error| format!("paired test request is invalid: {error}"))?;
        if !request.resources.accelerators.is_empty()
            || !request.resources.required_capabilities.is_empty()
        {
            return Err("target-test receiver has no admitted accelerator or placement-capability inventory".into());
        }
        // This is a type-level receiver check, never a grant. The linked
        // admission below still requires an exact factory and issuer for each
        // reached service, then installs and rechecks its live port before
        // native execution. Upper bounds are not effect requests.
        if header
            .binding
            .selected_product
            .services
            .iter()
            .any(|service| {
                !matches!(
                    service.kind,
                    runmat_execution::host::NativeDeploymentServiceKind::SourceEvaluation
                        | runmat_execution::host::NativeDeploymentServiceKind::HostedDynamicDispatch
                        | runmat_execution::host::NativeDeploymentServiceKind::Effect(
                            runmat_types::EffectKind::Clock
                                | runmat_types::EffectKind::Network
                                | runmat_types::EffectKind::UserInterface
                                | runmat_types::EffectKind::HostCallback
                                | runmat_types::EffectKind::WorkspaceRead
                                | runmat_types::EffectKind::WorkspaceWrite
                                | runmat_types::EffectKind::EnvironmentRead
                                | runmat_types::EffectKind::EnvironmentWrite
                                | runmat_types::EffectKind::FilesystemRead
                                | runmat_types::EffectKind::FilesystemWrite
                                | runmat_types::EffectKind::Randomness
                        )
                )
            })
        {
            return Err("target-test receiver cannot admit this selected service kind".into());
        }
        let function = header
            .binding
            .test(&request.suite, &request.test)
            .ok_or_else(|| "paired test entry is absent".to_string())?
            .compiled
            .compiled_function;
        Ok(Self {
            header,
            request,
            function,
        })
    }

    pub fn install_resources(
        &self,
        runtime: runmat_runtime::context::RuntimeContext,
    ) -> (
        runmat_runtime::context::RuntimeContext,
        ManagedHostEffectLedger,
    ) {
        let ledger = ManagedHostEffectLedger::new(self.request.resources.memory_bytes);
        let runtime = install_managed_resources(
            runtime,
            ledger.clone(),
            self.request.resources.cpu_millicores,
        );
        (runtime, ledger)
    }

    pub async fn invoke(
        &self,
        executor: Rc<NativeExecutor>,
        program: &runmat_native_codegen::aot::AotProgramManifest,
        requested_outputs: usize,
        runtime: runmat_runtime::context::RuntimeContext,
    ) -> Result<RootedNativeExecution, RuntimeError> {
        let run = async {
            let mut cooperation = runtime.cpu_cooperation();
            let arguments =
                runmat_runtime::execution::value_codec::decode_inline_values_cooperatively_in_context(
                    &self.request.inputs.arguments,
                    ValueLimits::default(),
                    None,
                    Some(&mut cooperation),
                    &runtime,
                )
                .await
                .map_err(|error| {
                    runmat_runtime::build_runtime_error(format!(
                        "target-test input decoding failed: {error}"
                    ))
                    .with_source(TargetTestInputDecodeFailure(error))
                    .build()
                })?;
            crate::program::invoke_rooted(
                executor,
                program,
                self.function,
                arguments,
                requested_outputs,
                runtime,
            )
            .await
        };
        match tokio::time::timeout(
            Duration::from_millis(self.request.resources.max_wall_millis),
            run,
        )
        .await
        {
            Ok(result) => result,
            Err(_) => Err(runmat_runtime::build_runtime_error(
                "target-test execution exceeded its configured wall time",
            )
            .with_source(TargetTestTimeout)
            .build()),
        }
    }

    pub fn result(
        &self,
        completion: (NativeTargetTestStatus, String),
        ledger: &ManagedHostEffectLedger,
        started: Instant,
    ) -> Result<NativeTargetTestResult, String> {
        let (status, mut diagnostic) = completion;
        // Keep text-oriented reporters unambiguous while retaining the exact
        // NUL occurrence as a reproducible printable escape.
        if diagnostic.contains('\0') {
            diagnostic = diagnostic.replace('\0', "\\0");
        }
        if diagnostic.len() > 64 * 1024 {
            let mut boundary = 64 * 1024;
            while !diagnostic.is_char_boundary(boundary) {
                boundary -= 1;
            }
            diagnostic.truncate(boundary);
        }
        // This is the actual installed Runtime managed-claim ledger after
        // terminal cleanup, not RSS or a full-call certificate. The portable
        // input decoder still owns unmanaged backing outside this ledger
        // (L25), and setup before installation is likewise unaccounted.
        let snapshot = ledger.snapshot();
        Ok(NativeTargetTestResult {
            schema_version: NATIVE_TARGET_TEST_PROTOCOL_SCHEMA_VERSION,
            request: self
                .request
                .digest_for(&self.header.binding)
                .map_err(|error| error.to_string())?,
            suite: self.request.suite.clone(),
            test: self.request.test.clone(),
            attempt: self.request.attempt,
            status,
            usage: Some(NativeTargetTestUsage {
                configured_memory_bytes: snapshot.capacity_bytes,
                peak_managed_reserved_bytes: snapshot.peak_reserved_bytes,
                peak_managed_claimed_bytes: snapshot.peak_claimed_bytes,
                elapsed_millis: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
            }),
            diagnostic,
        })
    }
}

struct ManagedTestResources {
    ledger: ManagedHostEffectLedger,
    cpu_millicores: u32,
}

impl runmat_runtime::context::RuntimeParallelService for ManagedTestResources {
    fn supports(&self, _: runmat_runtime::context::ParallelCapability) -> bool {
        false
    }

    fn placement_resources(&self) -> runmat_runtime::context::RuntimeParallelResources {
        let snapshot = self.ledger.snapshot();
        runmat_runtime::context::RuntimeParallelResources {
            cpu_millicores_available: self.cpu_millicores,
            memory_available_bytes: Some(snapshot.free_bytes()),
            epoch: snapshot.epoch,
        }
    }

    fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
        Some(self.ledger.clone())
    }
}

fn install_managed_resources(
    runtime: runmat_runtime::context::RuntimeContext,
    ledger: ManagedHostEffectLedger,
    cpu_millicores: u32,
) -> runmat_runtime::context::RuntimeContext {
    let ports = runtime
        .service_ports()
        .clone()
        .with_parallel(Rc::new(ManagedTestResources {
            ledger,
            cpu_millicores,
        }));
    runtime.with_service_ports(ports)
}
