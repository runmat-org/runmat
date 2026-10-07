use std::rc::{Rc, Weak};
use std::sync::Arc;

use runmat_execution::value::ValuePayload;
use runmat_runtime::context::{
    RuntimeCallableArtifactBundle, RuntimeCallableArtifactCommit, RuntimeCallableArtifactLimits,
    RuntimeCallableArtifactService, RuntimeContext, RuntimePreparedCallableAdmission,
    RuntimeServiceFuture,
};
use runmat_runtime::{build_runtime_error, RuntimeError};
use runmat_value::{BoundFunctionBinding, Value};

use super::{
    CallableArtifactSelectionError, FunctionRegistryGeneration, PreparedGenerationImport,
    SessionSourceEvaluator,
};

#[path = "callable_artifacts/errors.rs"]
mod errors;
use errors::*;
#[path = "callable_artifacts/source_facts.rs"]
mod source_facts;
use source_facts::{ensure_all_origins, exact_origin_resolver};
#[path = "callable_artifacts/dynamic.rs"]
mod dynamic;
pub(in crate::session) use dynamic::admit_dynamic_callable;

pub(super) struct SessionCallableArtifactService {
    evaluator: Weak<SessionSourceEvaluator>,
}

struct PreparedCallableAdmission {
    evaluator: Weak<SessionSourceEvaluator>,
    expected: Rc<FunctionRegistryGeneration>,
    expected_epoch: Rc<()>,
    replacement: PreparedGenerationImport,
    values: runmat_runtime::execution::RootedValues,
}

impl RuntimePreparedCallableAdmission for PreparedCallableAdmission {
    fn values(&self) -> &[Value] {
        self.values.values()
    }

    fn commit(self: Box<Self>) -> Result<RuntimeCallableArtifactCommit, RuntimeError> {
        let PreparedCallableAdmission {
            evaluator,
            expected,
            expected_epoch,
            replacement,
            values,
        } = *self;
        let evaluator = evaluator.upgrade().ok_or_else(|| {
            build_runtime_error("callable artifact session has ended")
                .with_identifier("RunMat:CallableArtifact:SessionEnded")
                .build()
        })?;
        let retired = evaluator.commit_import(&expected, &expected_epoch, replacement)?;
        Ok(RuntimeCallableArtifactCommit::new(
            values,
            Box::new(retired),
        ))
    }
}

impl SessionCallableArtifactService {
    fn evaluator(&self) -> Result<Rc<SessionSourceEvaluator>, RuntimeError> {
        self.evaluator.upgrade().ok_or_else(|| {
            build_runtime_error("callable artifact session has ended")
                .with_identifier("RunMat:CallableArtifact:SessionEnded")
                .build()
        })
    }
}

impl RuntimeCallableArtifactService for SessionCallableArtifactService {
    fn export(
        &self,
        runtime: RuntimeContext,
        values: Vec<Value>,
        limits: RuntimeCallableArtifactLimits,
    ) -> RuntimeServiceFuture<Result<RuntimeCallableArtifactBundle, RuntimeError>> {
        if values.len() > limits.max_values {
            return Box::pin(async move {
                Err(build_runtime_error(format!(
                    "callable_artifact.values exceeds its limit of {}",
                    limits.max_values
                ))
                .with_identifier("RunMat:CallableArtifact:Invalid")
                .build())
            });
        }
        let evaluator = match self.evaluator() {
            Ok(evaluator) => evaluator,
            Err(error) => return Box::pin(async move { Err(error) }),
        };
        if let Err(error) = evaluator.ensure_runtime_owner(&runtime) {
            return Box::pin(async move { Err(error) });
        }
        let values = match runmat_runtime::execution::RootedValues::new(values) {
            Ok(values) => values,
            Err(error) => return Box::pin(async move { Err(error.into_runtime_error()) }),
        };
        let generation = evaluator.current_generation();
        let artifacts = match generation.artifacts() {
            Ok(artifacts) => artifacts,
            Err(error) => return Box::pin(async move { Err(error) }),
        };
        let selected_runtime = runtime.clone();
        Box::pin(runtime.scope(async move {
            let payloads = values
                .values()
                .iter()
                .map(|value| {
                    runmat_runtime::execution::value_codec::encode_inline_value_in_context(
                        value,
                        &selected_runtime,
                    )
                    .map_err(codec_error)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let origins = ValuePayload::binding_origins(&payloads, limits.value_limits)
                .map_err(contract_error)?;
            let unit_bytes = artifacts
                .units_for_origins(&origins, limits.max_units, limits.max_unit_bytes)
                .map_err(artifact_selection_error)?;
            RuntimeCallableArtifactBundle::new(payloads, unit_bytes, limits)
                .map_err(|error| error.into_runtime_error())
        }))
    }

    fn prepare_admit(
        &self,
        runtime: RuntimeContext,
        bundle: RuntimeCallableArtifactBundle,
        limits: RuntimeCallableArtifactLimits,
    ) -> RuntimeServiceFuture<Result<Box<dyn RuntimePreparedCallableAdmission>, RuntimeError>> {
        let evaluator = match self.evaluator() {
            Ok(evaluator) => evaluator,
            Err(error) => return Box::pin(async move { Err(error) }),
        };
        if let Err(error) = evaluator.ensure_runtime_owner(&runtime) {
            return Box::pin(async move { Err(error) });
        }
        let expected = evaluator.current_generation();
        let registry = match expected.registry() {
            Ok(registry) => registry,
            Err(error) => return Box::pin(async move { Err(error) }),
        };
        let existing_artifacts = match expected.artifacts() {
            Ok(artifacts) => artifacts,
            Err(error) => return Box::pin(async move { Err(error) }),
        };
        let existing_owners = match expected.source_owners() {
            Ok(owners) => owners,
            Err(error) => return Box::pin(async move { Err(error) }),
        };
        let expected_epoch = expected.epoch();
        Box::pin(async move {
            let (payloads, unit_bytes) = bundle.into_parts();
            let (payloads, unit_bytes) =
                RuntimeCallableArtifactBundle::new(payloads, unit_bytes, limits)
                    .map_err(|error| error.into_runtime_error())?
                    .into_parts();
            let origins = ValuePayload::binding_origins(&payloads, limits.value_limits)
                .map_err(contract_error)?;
            let mut units = Vec::with_capacity(unit_bytes.len());
            for bytes in &unit_bytes {
                let admitted =
                    runmat_vm::admit_executable_unit(bytes).map_err(executable_admission_error)?;
                if !admitted.registry().functions.values().any(|body| {
                    body.binding_origin
                        .is_some_and(|origin| origins.contains(&origin))
                }) {
                    return Err(build_runtime_error(
                        "callable artifact contains an executable unit not required by its values",
                    )
                    .with_identifier("RunMat:CallableArtifact:UnreferencedExecutable")
                    .build());
                }
                units.push(
                    admitted
                        .bind_runtime_sources(runtime.clone())
                        .map_err(executable_admission_error)?,
                );
            }
            let prepared = runmat_vm::prepare_retained_callable_units(
                &runtime,
                &registry,
                units,
                runmat_vm::RetainedCallableStageLimits {
                    max_units: limits.max_units,
                    max_functions: limits.max_functions,
                    max_instructions: limits.max_instructions,
                },
            )
            .map_err(staging_error)?;
            let exact_facts = Arc::clone(prepared.exact_origin_facts());
            ensure_all_origins(&origins, &exact_facts)?;
            let resolver = exact_origin_resolver(exact_facts);
            let values = runtime
                .clone()
                .scope(async move {
                    let mut cooperation = runtime.cpu_cooperation();
                    runmat_runtime::execution::value_codec::decode_inline_values_cooperatively(
                        &payloads,
                        limits.value_limits,
                        Some(resolver),
                        Some(&mut cooperation),
                    )
                    .await
                    .map_err(cooperative_codec_error)
                })
                .await?;
            let (registry, _, staging_leases, imported_owners) = prepared.into_parts();
            let mut artifacts = existing_artifacts.as_ref().clone();
            for bytes in unit_bytes {
                artifacts
                    .insert_canonical_unit(bytes)
                    .map_err(artifact_index_error)?;
            }
            let mut source_owners = existing_owners.iter().cloned().collect::<Vec<_>>();
            source_owners.extend(imported_owners.into_iter().map(Rc::new));
            let replacement = evaluator.prepare_import(registry, artifacts, source_owners);
            drop(staging_leases);
            Ok(Box::new(PreparedCallableAdmission {
                evaluator: Rc::downgrade(&evaluator),
                expected,
                expected_epoch,
                replacement,
                values,
            }) as Box<dyn RuntimePreparedCallableAdmission>)
        })
    }
}

pub(in crate::session) fn service(
    evaluator: &Rc<SessionSourceEvaluator>,
) -> Rc<dyn RuntimeCallableArtifactService> {
    Rc::new(SessionCallableArtifactService {
        evaluator: Rc::downgrade(evaluator),
    })
}

pub(in crate::session) fn install_bound_invoker(
    runtime: &RuntimeContext,
    evaluator: &Rc<SessionSourceEvaluator>,
) {
    let evaluator = Rc::downgrade(evaluator);
    runtime.set_bound_function_invoker(Some(Rc::new(
        move |runtime, binding: BoundFunctionBinding, arguments, requested_outputs| {
            let prepared = (|| {
                let evaluator = evaluator.upgrade().ok_or_else(|| {
                    build_runtime_error("bound function session has ended")
                        .with_identifier("RunMat:BoundFunction:SessionEnded")
                        .build()
                })?;
                evaluator.ensure_runtime_owner(&runtime)?;
                let generation = evaluator.current_generation();
                let registry = generation.registry()?;
                let selected = registry
                    .owned_view(runmat_hir::FunctionId(binding.function), &binding.authority)
                    .ok_or_else(|| unavailable_binding(binding.function))?;
                Ok::<_, RuntimeError>((generation, selected, arguments))
            })();
            Box::pin(async move {
                let (_generation, selected, arguments) = prepared?;
                runmat_vm::invoke_semantic_function_value_in_context(
                    binding.function,
                    arguments.values(),
                    requested_outputs,
                    &selected,
                    runtime,
                )
                .await
            })
        },
    )));
}

#[cfg(test)]
#[path = "callable_artifacts/tests.rs"]
mod tests;
