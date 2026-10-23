use super::super::super::*;
use std::rc::Rc;

pub(super) async fn selected_constructor_dependencies(
    cache: &super::super::super::dynamic_cache::DynamicFunctionCache,
    selected_unit: &runmat_vm::AdmittedExecutableUnit,
    runtime: &runmat_runtime::context::RuntimeContext,
    owner: &runmat_types::ClassMethodOwner,
    compat: CompatMode,
    top_level_await_enabled: bool,
    project_handoff: Option<&runmat_package::FrozenProjectHandoff>,
) -> Result<
    (
        Vec<super::super::super::source_evaluation::DynamicCallableDependency>,
        HashSet<super::super::super::DynamicDependencyIdentity>,
        Vec<Rc<runmat_vm::AdmittedExecutableUnit>>,
    ),
    RuntimeError,
> {
    let mut parents =
        runmat_vm::unresolved_admitted_class_parents_for(selected_unit, &owner.declaring_class)?;
    let mut origins = HashSet::new();
    let mut dependencies = Vec::with_capacity(parents.len());
    let mut metadata_units = Vec::with_capacity(parents.len());
    let mut cursor = 0;
    while cursor < parents.len() {
        let parent = parents[cursor].clone();
        cursor += 1;
        let selected =
            runmat_runtime::builtins::common::path_search::find_file_with_extensions_with_origin(
                parent.display_name(),
                &[".m"],
                "superclass source resolution",
            )
            .await
            .map_err(|error| {
                error.into_identified_runtime_error(
                    "superclass source resolution",
                    "RunMat:FunctionResolution",
                )
            })?
            .ok_or_else(|| {
                build_runtime_error("selected superclass source is unavailable")
                    .with_identifier("RunMat:CallableArtifact:IncompleteDefinitionGraph")
                    .build()
            })?;
        let path = selected.path().to_path_buf();
        let source_text =
            runmat_runtime::builtins::io::repl_fs::pcode::read_source_text_async(&path)
                .await
                .map_err(|error| {
                    build_runtime_error(format!(
                        "Could not read superclass source '{}': {error}",
                        path.display()
                    ))
                    .with_identifier("RunMat:FunctionSourceRead")
                    .with_source(error)
                    .with_gpu_gather_retry(runmat_runtime::GpuGatherRetry::Never)
                    .build()
                })?;
        let path_name = path.to_string_lossy();
        let project_context = if let Some(handoff) = project_handoff {
            Some(
                super::super::super::compile::project_compilation_context_from_handoff(
                    path_name.as_ref(),
                    handoff,
                ),
            )
            .transpose()
        } else {
            super::super::super::compile::discover_project_compilation_context_async(
                path_name.as_ref(),
            )
            .await
        }
        .map_err(|error| {
            build_runtime_error(format!(
                "Could not construct project context for '{}': {error}",
                path.display()
            ))
            .with_identifier("RunMat:ProjectComposition")
            .with_source(error)
            .with_gpu_gather_retry(runmat_runtime::GpuGatherRetry::Never)
            .build()
        })?;
        let primary_revision = runmat_execution::Digest::sha256(source_text.as_bytes());
        let inventory_revision = project_context
            .as_ref()
            .map(|context| context.dynamic_inventory_revision())
            .unwrap_or_else(|| runmat_execution::Digest::sha256("empty-source-inventory"));
        let cached = cache.get(&path).filter(|entry| {
            entry.selection.matches_selection(
                &primary_revision,
                &inventory_revision,
                selected.origin(),
            ) && super::super::super::dynamic_cache::DynamicFunctionCache::declares_super_class(
                entry, &parent,
            )
        });
        let cached = if let Some(entry) = cached {
            super::super::reusable_dynamic_product(&entry)
                .await
                .then_some(entry)
        } else {
            None
        };
        let (unit_bytes, admitted, source_records, selection) = if let Some(entry) = cached {
            (
                entry.unit_bytes,
                entry.admitted,
                entry.source_records,
                super::super::super::DynamicDependencyIdentity::Callable(entry.entrypoint_origin),
            )
        } else {
            let phase = runmat_runtime::user_functions::DynamicFunctionLoadPhase::RegisteredConstructorSource(parent.clone());
            let product =
                super::compile::compile_dynamic_product(super::compile::DynamicCompileRequest {
                    name: parent.display_name(),
                    path: &path,
                    path_origin: selected.origin(),
                    source_text: &source_text,
                    project_context: project_context.as_ref(),
                    compat,
                    top_level_await_enabled,
                    project_handoff,
                    phase: &phase,
                    dependency_only: true,
                })
                .await?;
            (
                product.unit_bytes,
                product.admitted,
                product.source_records,
                product.selection,
            )
        };
        for ancestor in runmat_vm::unresolved_admitted_class_parents_for(&admitted, &parent)? {
            if !parents.contains(&ancestor) {
                parents.push(ancestor);
            }
        }
        if !origins.insert(selection) {
            continue;
        }
        let bound = admitted
            .as_ref()
            .clone()
            .bind_runtime_sources_with_records(runtime.clone(), source_records.to_vec())
            .map_err(|error| {
                build_runtime_error(error.to_string())
                    .with_identifier("RunMat:ExecutableSourceIdentity")
                    .with_source(error)
                    .build()
            })?;
        metadata_units.push(admitted);
        dependencies.push(
            super::super::super::source_evaluation::DynamicCallableDependency {
                unit_bytes,
                unit: bound,
            },
        );
    }
    Ok((dependencies, origins, metadata_units))
}
