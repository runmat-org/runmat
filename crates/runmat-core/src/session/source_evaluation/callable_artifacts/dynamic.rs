use std::rc::Rc;

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
    _retirement: GenerationRetirement,
    _staging_leases: Vec<FunctionDefinitionLease>,
}

impl DynamicCallableAdmission {
    pub(in crate::session) fn binding(&self) -> &BoundFunctionBinding {
        &self.binding
    }
}

pub(in crate::session) fn admit_dynamic_callable(
    evaluator: &Rc<SessionSourceEvaluator>,
    runtime: &RuntimeContext,
    unit_bytes: &[u8],
    entrypoint_origin: runmat_execution::FunctionBindingOrigin,
    unit: runmat_vm::RuntimeBoundExecutableUnit,
) -> Result<DynamicCallableAdmission, RuntimeError> {
    evaluator.ensure_runtime_owner(runtime)?;
    let expected = evaluator.current_generation();
    let registry = expected.registry()?;
    let existing_artifacts = expected.artifacts()?;
    let existing_owners = expected.source_owners()?;
    let expected_epoch = expected.epoch();

    let prepared = runmat_vm::prepare_retained_callable_units(
        runtime,
        &registry,
        vec![unit],
        runmat_vm::RetainedCallableStageLimits {
            max_units: 1,
            max_functions: 65_536,
            max_instructions: 10_000_000,
        },
    )
    .map_err(staging_error)?;
    let exact_facts = exact_origin_resolver(Rc::clone(prepared.exact_origin_facts()));
    let binding = {
        let _entry = runtime.enter();
        let _facts = runmat_runtime::user_functions::install_execution_source_resolver(Some(
            Rc::clone(&exact_facts),
        ));
        runmat_runtime::user_functions::resolve_binding_origin_binding(entrypoint_origin)
            .ok_or_else(|| {
                build_runtime_error("dynamic executable lacks its exact entrypoint definition")
                    .with_identifier("RunMat:CallableArtifact:MissingDefinition")
                    .build()
            })?
    };
    let (registry, _, staging_leases, imported_owners) = prepared.into_parts();
    let mut artifacts = existing_artifacts.as_ref().clone();
    artifacts
        .insert_canonical_unit(unit_bytes.to_vec())
        .map_err(artifact_index_error)?;
    let mut source_owners = existing_owners.iter().cloned().collect::<Vec<_>>();
    source_owners.extend(imported_owners.into_iter().map(Rc::new));
    let replacement = evaluator.prepare_import(registry, artifacts, source_owners);
    let retirement = evaluator.commit_import(&expected, &expected_epoch, replacement)?;
    Ok(DynamicCallableAdmission {
        binding,
        _retirement: retirement,
        _staging_leases: staging_leases,
    })
}
