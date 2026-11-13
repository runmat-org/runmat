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
pub(in crate::session) use dynamic::{admit_dynamic_callable, DynamicCallableDependency};

enum BoundInvocationKind {
    SuperBody,
    Constructor(runmat_types::ClassMethodOwner),
    Ordinary,
}

struct PreparedBoundInvocation {
    kind: BoundInvocationKind,
    state: BoundInvocationState,
}

struct BoundInvocationState {
    _generation: Rc<FunctionRegistryGeneration>,
    selected: runmat_vm::FunctionRegistry,
    arguments: runmat_runtime::call::RootedFunctionArguments,
    runtime: RuntimeContext,
    selected_product: Option<Rc<runmat_vm::SelectedClassProduct>>,
    // Keep caller closure handles rooted until the invocation has settled.
    _environment_roots: Option<Vec<runmat_gc::ExplicitRoot>>,
    dynamic_admission: Option<runmat_vm::DynamicEvalAdmissionPort>,
}

fn admit_bound_invocation<F>(
    state: BoundInvocationState,
    make_future: impl FnOnce(BoundInvocationState) -> F,
) -> Result<runmat_runtime::user_functions::BoundFunctionFuture, RuntimeError>
where
    F: std::future::Future<
            Output = Result<runmat_runtime::execution::RootedValueSequence, RuntimeError>,
        > + 'static,
{
    let runtime = state.runtime.clone();
    runmat_runtime::execution::admit_executable_sequence_with(&runtime, move || make_future(state))
}

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
                let mut unit = admitted
                    .bind_runtime_sources(runtime.clone())
                    .map_err(executable_admission_error)?;
                unit.attach_local_selected_product().map_err(|message| {
                    executable_admission_error(
                        runmat_vm::ExecutableUnitAdmissionError::SourceBinding(message),
                    )
                })?;
                units.push(unit);
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
            let resolver = exact_origin_resolver(exact_facts, Rc::new(prepared.registry().clone()));
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
            let (registry, _, staging_leases, imported_owners) =
                prepared.into_parts().map_err(staging_error)?;
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
        move |runtime,
              binding: BoundFunctionBinding,
              arguments,
              requested_outputs,
              lexical_environment| {
            let prepared = (|| {
                // The boxed sibling future can be queued without a first poll.
                // Retain the exact caller activation before source selection or
                // constructor admission can allocate or trigger a collection.
                let environment_roots = lexical_environment
                    .as_ref()
                    .map(runmat_runtime::call::lexical::root_environment)
                    .transpose()?;
                let evaluator = evaluator.upgrade().ok_or_else(|| {
                    build_runtime_error("bound function session has ended")
                        .with_identifier("RunMat:BoundFunction:SessionEnded")
                        .build()
                })?;
                evaluator.ensure_runtime_owner(&runtime)?;
                let generation = evaluator.current_generation();
                let registry = generation.registry()?;
                let function = runmat_hir::FunctionId(binding.function);
                let selected = registry
                    .owned_view(function, &binding.authority)
                    .ok_or_else(|| unavailable_binding(binding.function))?;
                // The constructor receiver is poll-local. Capture its exact
                // admitted owner while this caller is still selected, before
                // the boxed VM future can be polled as a sibling process.
                let selected_body = selected
                    .get(function)
                    .ok_or_else(|| unavailable_binding(binding.function))?;
                let dynamic_admission =
                    evaluator.dynamic_admission_port_for_selected_body(selected_body)?;
                let source_evaluation = evaluator
                    .source_evaluation_for_selected_body(dynamic_admission.clone());
                let owner = selected_body.class_method_owner.clone();
                let selected_product = selected_body.selected_class_product.clone();
                let (runtime, selected_method) =
                    selected.prepare_selected_receiver_method(runtime, &binding)?;
                let (runtime, super_body_owner) =
                    runtime.take_retained_super_body_for(&binding)?;
                let super_body = if let Some(expected) = super_body_owner {
                    if owner.as_ref() != Some(&expected)
                        || expected.role != runmat_types::ClassMethodRole::Constructor
                        || expected.is_static
                    {
                        return Err(build_runtime_error(
                            "selected superclass body disagrees with its admitted method owner",
                        )
                        .with_identifier("RunMat:ConstructorOwnerInvalid")
                        .build());
                    }
                    true
                } else {
                    false
                };
                let runtime = if selected_method || super_body {
                    runtime
                } else {
                    runtime.without_selected_object_class()
                };
                #[cfg(not(target_arch = "wasm32"))]
                // A dynamic source loader may already have issued this exact
                // child a path-MEX route. Preserve only that issuance, never
                // ambient session lookup or another selected body's grant.
                let ordinary_native = match runtime.native_resolution_scope() {
                    runmat_runtime::user_functions::NativeResolutionScope::SelectedOrdinary {
                        route, mex, source_owner,
                    } if source_owner.is_bound_in(&runtime)
                        && selected_body.selected_source_owner.as_ref().is_some_and(|owner| owner.same_issuance(source_owner)) =>
                    {
                        Some((Rc::clone(route), Rc::clone(mex), source_owner.clone()))
                    }
                    _ => None,
                };
                let runtime = runtime
                    .without_selected_graph_program()
                    .with_selected_native_route(None);
                let ports = runtime
                    .service_ports()
                    .clone()
                    .with_source_evaluation(source_evaluation);
                let runtime = runtime.with_service_ports(ports);
                let runtime = if owner.as_ref().is_some_and(|owner| {
                    owner.role == runmat_types::ClassMethodRole::Constructor
                }) {
                    runtime
                } else {
                    runtime.with_constructor_receiver_for_method(owner.as_ref())?
                };
                let receipt = selected_body
                    .selected_executable_product
                    .as_ref()
                    .and_then(|product| product.interop_receipt())
                    .cloned();
                let needs_graph = runtime.execution().requires_program_capture();
                let runtime = if needs_graph || receipt.is_some() {
                    let (origin, source_owner) = selected_body
                        .binding_origin
                        .zip(selected_body.selected_source_owner.as_ref())
                        .ok_or_else(|| {
                            build_runtime_error("bound callable has no selected V3 source owner")
                                .with_identifier("RunMat:CallableArtifact:SelectedGraph")
                                .build()
                        })?;
                    let bytes = selected_body
                        .selected_executable_product
                        .as_ref()
                        .filter(|product| product.source_owner().same_issuance(source_owner))
                        .map(|product| Arc::clone(product.canonical_v3()))
                        .ok_or_else(|| {
                            build_runtime_error("bound callable has no selected V3 product")
                                .with_identifier("RunMat:CallableArtifact:SelectedGraph")
                                .build()
                        })?;
                    let primary = super::SelectedGraphPrimary::retained(
                        bytes,
                        source_owner.clone(),
                        selected.clone(),
                        function,
                        origin,
                    )
                    .map_err(|error| {
                        build_runtime_error(error)
                            .with_identifier("RunMat:CallableArtifact:SelectedGraph")
                            .build()
                    })?;
                    let primary = Rc::new(primary);
                    if receipt.is_some() {
                        #[cfg(not(target_arch = "wasm32"))]
                        {
                            evaluator.bind_selected_foreign(
                                runtime,
                                primary,
                                Some(selected_body),
                                receipt,
                                None,
                            )?
                        }
                        #[cfg(target_arch = "wasm32")]
                        {
                            return Err(build_runtime_error(
                                "selected native foreign artifacts cannot execute in this browser host",
                            )
                            .with_identifier("RunMat:SelectedForeignHost")
                            .build());
                        }
                    } else {
                        let ports = runtime.service_ports().clone().without_foreign();
                        runtime
                            .with_service_ports(ports)
                            .with_selected_graph_program(super::SelectedGraphSnapshot::program(
                                primary,
                                &evaluator,
                            ))
                    }
                } else {
                    let ports = runtime.service_ports().clone().without_foreign();
                    runtime.with_service_ports(ports)
                };
                #[cfg(not(target_arch = "wasm32"))]
                let runtime = if let Some((issued_route, mex, source_owner)) = ordinary_native {
                    if !source_owner.is_bound_in(&runtime) {
                        return Err(build_runtime_error("bound ordinary MEX body changed its exact issued source")
                            .with_identifier("RunMat:MEX:SourceAuthority").build());
                    }
                    let route = match runtime.native_resolution_scope() {
                        runmat_runtime::user_functions::NativeResolutionScope::Selected(pinned) =>
                            // Newly selected sealed artifacts retain priority.
                            super::ordinary_native_route(Some(Rc::clone(pinned)), Rc::clone(&mex), source_owner.clone()),
                        runmat_runtime::user_functions::NativeResolutionScope::SelectedClosed => issued_route,
                        _ => return Err(build_runtime_error("bound ordinary MEX body has an unexpected native grant")
                            .with_identifier("RunMat:MEX:SourceAuthority").build()),
                    };
                    runtime.with_ordinary_selected_native_route(route, mex, source_owner)
                } else {
                    runtime
                };
                let runtime = registry.bind_selected_body_sites(
                    runtime,
                    &binding,
                    selected_body,
                )?;
                let kind = if super_body {
                    BoundInvocationKind::SuperBody
                } else if let Some(owner) = owner.filter(|owner| {
                    owner.role == runmat_types::ClassMethodRole::Constructor
                }) {
                    BoundInvocationKind::Constructor(owner)
                } else {
                    BoundInvocationKind::Ordinary
                };
                Ok::<_, RuntimeError>(PreparedBoundInvocation {
                    kind,
                    state: BoundInvocationState {
                        _generation: generation,
                        selected,
                        arguments,
                        runtime,
                        selected_product,
                        _environment_roots: environment_roots,
                        dynamic_admission,
                    },
                })
            })();
            // The caller already requires one boxed UserFunctionFuture. Select
            // its branch before boxing, so an ordinary callback's poll frame
            // does not reserve the constructor branch's future state too.
            let invocation = match prepared {
                Err(error) => Err(error),
                Ok(PreparedBoundInvocation {
                    kind: BoundInvocationKind::SuperBody,
                    state,
                }) => admit_bound_invocation(state, move |state| async move {
                    let BoundInvocationState {
                        _generation,
                        selected,
                        arguments,
                        runtime,
                        selected_product,
                        _environment_roots,
                        dynamic_admission,
                    } = state;
                    // Preserve inherited Derived facts. A compiled direct
                    // constructor can lack them; only the entered local
                    // parent's exact product may supply the missing proof.
                    let runtime = if runtime.selected_object_class().is_none()
                        && matches!(&binding.authority, runmat_value::BoundFunctionAuthority::Local { .. })
                    {
                        selected_product.as_ref().ok_or_else(|| {
                            build_runtime_error("selected superclass body lacks its exact class product")
                                .with_identifier("RunMat:ConstructorOwnerInvalid").build()
                        })?.invocation_runtime_with_binding(runtime, &selected, &binding)?
                    } else {
                        runtime
                    };
                    runmat_vm::invoke_semantic_function_with_rooted_arguments_and_dynamic_admission_in_context(
                        binding.function,
                        arguments,
                        requested_outputs,
                        &selected,
                        runtime,
                        lexical_environment,
                        dynamic_admission,
                    )
                    .await
                }),
                Ok(PreparedBoundInvocation {
                    kind: BoundInvocationKind::Constructor(owner),
                    state,
                }) => {
                    admit_bound_invocation(state, move |state| async move {
                        let BoundInvocationState {
                            _generation,
                            selected,
                            arguments,
                            runtime,
                            selected_product,
                            _environment_roots,
                            dynamic_admission,
                        } = state;
                        let runtime = selected_product
                            .ok_or_else(|| {
                                build_runtime_error(
                                    "selected constructor lacks its exact class product",
                                )
                                .with_identifier("RunMat:ConstructorOwnerInvalid")
                                .build()
                            })?
                            .invocation_runtime_with_binding(runtime, &selected, &binding)?;
                        let argument_count = arguments.supplied_count();
                        let invocation_owner = owner.clone();
                        let admitted_binding = binding.clone();
                        let constructor_dynamic_admission = dynamic_admission.clone();
                        let invoke = runmat_runtime::dispatcher::PreparedConstructorInvoke::new(
                            &runtime,
                            move |selected_runtime: RuntimeContext| async move {
                                    let selected_runtime = selected_runtime
                                        .with_constructor_receiver_for_method(Some(&invocation_owner))?;
                                    runmat_vm::invoke_semantic_function_with_rooted_arguments_and_dynamic_admission_in_context(
                                        binding.function,
                                        arguments,
                                        requested_outputs,
                                        &selected,
                                        selected_runtime,
                                        lexical_environment,
                                        constructor_dynamic_admission,
                                    )
                                    .await
                            },
                        )?;
                        runmat_runtime::dispatcher::call_prepared_constructor_sequence_with_argument_count_in_context(
                            &runtime,
                            owner,
                            admitted_binding,
                            argument_count,
                            requested_outputs,
                            invoke,
                        )
                        .await
                    })
                }
                Ok(PreparedBoundInvocation {
                    kind: BoundInvocationKind::Ordinary,
                    state,
                }) => admit_bound_invocation(state, move |state| async move {
                    let BoundInvocationState {
                        _generation,
                        selected,
                        arguments,
                        runtime,
                        selected_product: _selected_product,
                        _environment_roots,
                        dynamic_admission,
                    } = state;
                    runmat_vm::invoke_semantic_function_with_rooted_arguments_and_dynamic_admission_in_context(
                        binding.function,
                        arguments,
                        requested_outputs,
                        &selected,
                        runtime,
                        lexical_environment,
                        dynamic_admission,
                    )
                    .await
                }),
            };
            invocation
        },
    )));
}

#[cfg(test)]
#[path = "callable_artifacts/tests.rs"]
mod tests;
