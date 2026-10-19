use std::rc::Rc;
use std::sync::Arc;

use runmat_runtime::context::RuntimeContext;
use runmat_runtime::{build_runtime_error, RuntimeError};
use runmat_value::{BoundFunctionBinding, FunctionDefinitionLease};

use super::{artifact_index_error, exact_origin_resolver, staging_error};
use crate::session::source_evaluation::{GenerationRetirement, SessionSourceEvaluator};

/// Exact retained-only admission for one dynamically selected source product.
/// The binding owns its definition graph; the generation owns the canonical
/// envelope and source leases. No display name enters ordinary resolution.
pub(in crate::session) struct DynamicCallableAdmission {
    binding: BoundFunctionBinding,
    super_constructors: Vec<runmat_runtime::context::RetainedSuperConstructorBinding>,
    _retirement: GenerationRetirement,
    _staging_leases: Vec<FunctionDefinitionLease>,
}

/// A source product selected by the dynamic loader as an exact dependency of
/// the entrypoint. Its canonical bytes and runtime-bound owner travel together.
pub(in crate::session) struct DynamicCallableDependency {
    pub(in crate::session) unit_bytes: Arc<[u8]>,
    pub(in crate::session) unit: runmat_vm::RuntimeBoundExecutableUnit,
}

impl DynamicCallableAdmission {
    pub(in crate::session) fn binding(&self) -> &BoundFunctionBinding {
        &self.binding
    }

    pub(in crate::session) fn super_constructors(
        &self,
    ) -> &[runmat_runtime::context::RetainedSuperConstructorBinding] {
        &self.super_constructors
    }
}

pub(in crate::session) fn admit_dynamic_callable(
    evaluator: &Rc<SessionSourceEvaluator>,
    runtime: &RuntimeContext,
    unit_bytes: &[u8],
    entrypoint_origin: runmat_execution::FunctionBindingOrigin,
    unit: runmat_vm::RuntimeBoundExecutableUnit,
    dependencies: Vec<DynamicCallableDependency>,
) -> Result<DynamicCallableAdmission, RuntimeError> {
    evaluator.ensure_runtime_owner(runtime)?;
    let expected = evaluator.current_generation();
    let registry = expected.registry()?;
    let existing_artifacts = expected.artifacts()?;
    let existing_owners = expected.source_owners()?;
    let expected_epoch = expected.epoch();
    let unit_count = dependencies.len() + 1;
    let dependency_bytes = dependencies
        .iter()
        .map(|dependency| Arc::clone(&dependency.unit_bytes))
        .collect::<Vec<_>>();
    let mut units = Vec::with_capacity(unit_count);
    units.push(unit);
    units.extend(dependencies.into_iter().map(|dependency| dependency.unit));
    for (index, selected) in units.iter_mut().enumerate() {
        let expected_bytes = if index == 0 {
            unit_bytes
        } else {
            dependency_bytes[index - 1].as_ref()
        };
        if selected
            .canonical_v3()
            .is_none_or(|bytes| bytes.as_ref() != expected_bytes)
        {
            return Err(
                build_runtime_error("dynamic selected unit changes its admitted V3 bytes")
                    .with_identifier("RunMat:CallableArtifact:SelectedProduct")
                    .build(),
            );
        }
        selected
            .attach_local_selected_product()
            .map_err(|message| {
                build_runtime_error(message)
                    .with_identifier("RunMat:CallableArtifact:SelectedProduct")
                    .build()
            })?;
    }
    let prepared = runmat_vm::prepare_retained_callable_units(
        runtime,
        &registry,
        units,
        runmat_vm::RetainedCallableStageLimits {
            max_units: unit_count,
            max_functions: 65_536,
            max_instructions: 10_000_000,
        },
    )
    .map_err(staging_error)?;
    let exact_facts = exact_origin_resolver(
        Arc::clone(prepared.exact_origin_facts()),
        Rc::new(prepared.registry().clone()),
    );
    let (binding, super_constructors, selected_class) = {
        let binding = runmat_runtime::user_functions::resolve_binding_origin_binding_in_context(
            &exact_facts,
            entrypoint_origin,
            runtime,
        )
        .ok_or_else(|| {
            build_runtime_error("dynamic executable lacks its exact entrypoint definition")
                .with_identifier("RunMat:CallableArtifact:MissingDefinition")
                .build()
        })?;
        let selected = prepared
            .registry()
            .owned_view(runmat_hir::FunctionId(binding.function), &binding.authority)
            .ok_or_else(|| {
                build_runtime_error("dynamic executable lost its exact definition graph")
                    .with_identifier("RunMat:CallableArtifact:IncompleteDefinitionGraph")
                    .build()
            })?;
        let super_targets = selected
            .retained_super_constructor_targets_for(runmat_hir::FunctionId(binding.function))
            .map_err(|error| {
                build_runtime_error(error.to_string())
                    .with_identifier("RunMat:CallableArtifact:IncompleteDefinitionGraph")
                    .with_source(error)
                    .build()
            })?;
        let implicit_targets = prepared
            .retained_implicit_super_constructor_targets_for(
                &selected,
                runmat_hir::FunctionId(binding.function),
            )
            .map_err(|error| {
                build_runtime_error(error.to_string())
                    .with_identifier("RunMat:CallableArtifact:IncompleteDefinitionGraph")
                    .with_source(error)
                    .build()
            })?;
        let selected_class = selected
            .get(runmat_hir::FunctionId(binding.function))
            .and_then(|body| body.class_method_owner.as_ref())
            .filter(|owner| owner.role == runmat_types::ClassMethodRole::Constructor)
            .map(|owner| owner.declaring_class.clone());
        let selected = Rc::new(selected);
        let selected_facts = selected.execution_source_resolver_in(runtime);
        let mut super_constructors = super_targets
            .into_iter()
            .map(|target| {
                let binding =
                    runmat_runtime::user_functions::resolve_binding_origin_binding_in_context(
                        &selected_facts,
                        target.origin,
                        runtime,
                    )
                    .ok_or_else(|| {
                        build_runtime_error(
                            "dynamic executable lacks an exact superclass constructor definition",
                        )
                        .with_identifier("RunMat:CallableArtifact:MissingDefinition")
                        .build()
                    })?;
                runmat_runtime::context::RetainedSuperConstructorBinding::from_admitted_definition(
                    &target.caller_declaration,
                    &target.target_declaration,
                    target.caller_origin,
                    target.origin,
                    target.owner,
                    binding,
                )
            })
            .collect::<Result<Vec<_>, RuntimeError>>()?;
        super_constructors.extend(
            implicit_targets
                .into_iter()
                .map(|target| target.into_binding())
                .collect::<Result<Vec<_>, RuntimeError>>()?,
        );
        (binding, super_constructors, selected_class)
    };
    // Reject an invalid/duplicate projection before committing the staged
    // registry. The same exact bindings are installed on the call context.
    runtime
        .clone()
        .with_retained_super_constructors(super_constructors.clone(), entrypoint_origin)?;
    let selected_product = selected_class
        .as_ref()
        .map(|class| prepared.selected_class_product_for(class))
        .transpose()?;
    let (mut registry, _, staging_leases, imported_owners) =
        prepared.into_parts().map_err(staging_error)?;
    if let Some(product) = selected_product {
        registry
            .attach_selected_class_product(
                runmat_hir::FunctionId(binding.function),
                &binding.authority,
                Rc::new(product),
            )
            .map_err(|message| {
                build_runtime_error(message)
                    .with_identifier("RunMat:CallableArtifact:IncompleteDefinitionGraph")
                    .build()
            })?;
    }
    let mut artifacts = existing_artifacts.as_ref().clone();
    artifacts
        .insert_canonical_unit(unit_bytes.to_vec())
        .map_err(artifact_index_error)?;
    for dependency in dependency_bytes {
        artifacts
            .insert_canonical_unit(dependency.to_vec())
            .map_err(artifact_index_error)?;
    }
    let mut source_owners = existing_owners.iter().cloned().collect::<Vec<_>>();
    source_owners.extend(imported_owners.into_iter().map(Rc::new));
    let replacement = evaluator.prepare_import(registry, artifacts, source_owners);
    let retirement = evaluator.commit_import(&expected, &expected_epoch, replacement)?;
    Ok(DynamicCallableAdmission {
        binding,
        super_constructors,
        _retirement: retirement,
        _staging_leases: staging_leases,
    })
}
