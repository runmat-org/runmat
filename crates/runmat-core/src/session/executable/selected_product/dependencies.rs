//! Reissue a bounded original-product DAG without recursive native stack use.
use std::collections::{HashMap, HashSet};

use runmat_execution::executable::{ExecutableStaticCaller, ExecutableStaticImportKind};
use runmat_hir::FunctionId;
use runmat_runtime::context::RuntimeContext;
use runmat_vm::FunctionRegistry;

use crate::session::source_evaluation::{
    selected_static_imports_with_preselected, SelectedStaticTarget,
};
use crate::{ExecutableUnit, RunError, RunMatSession};

use super::runtime_error;

pub(super) type OriginalDependencyViews = HashMap<*const ExecutableUnit, FunctionRegistry>;

fn error(message: impl Into<String>) -> RunError {
    runtime_error("RunMat:ExecutableSourceIdentity", message)
}

impl RunMatSession {
    pub(super) fn prepare_original_dependency_views(
        &mut self,
        unit: &ExecutableUnit,
        runtime: &RuntimeContext,
    ) -> Result<OriginalDependencyViews, RunError> {
        let mut views = HashMap::new();
        for owner in dependency_order(unit, runtime)? {
            if std::ptr::eq(owner, unit) {
                continue;
            }
            let invocation = owner
                .bind_runtime_sources(
                    runtime
                        .clone()
                        .with_program_revision(Some(owner.revision().program_revision.clone())),
                )
                .map_err(error)?;
            let mut bytecode = owner.runtime_bytecode(&invocation).map_err(error)?;
            self.prepare_executable_graph_with_dependencies(
                owner,
                &invocation,
                &mut bytecode,
                &views,
            )?;
            views.insert(owner as *const ExecutableUnit, bytecode.function_registry);
        }
        Ok(views)
    }
}

fn dependency_order<'a>(
    unit: &'a ExecutableUnit,
    runtime: &RuntimeContext,
) -> Result<Vec<&'a ExecutableUnit>, RunError> {
    let mut states = HashMap::new();
    let mut pending = vec![(unit, false)];
    let mut order = Vec::new();
    while let Some((owner, finish)) = pending.pop() {
        let key = owner as *const ExecutableUnit;
        if finish {
            states.insert(key, true);
            order.push(owner);
            continue;
        }
        match states.get(&key) {
            Some(true) => continue,
            Some(false) => return Err(error("original dependency inventory is cyclic")),
            None => {}
        }
        if states.len() > runmat_execution::EXECUTABLE_DEPENDENCY_MAX_PRODUCTS {
            return Err(error("original dependency inventory exceeds product limit"));
        }
        states.insert(key, false);
        pending.push((owner, true));
        // Already-selected live dependencies keep their original issued views.
        if !usable_original_targets(owner, runtime) {
            pending.extend(
                owner
                    .original_dependencies()
                    .values()
                    .map(|owner| (owner.as_ref(), false)),
            );
        }
    }
    Ok(order)
}

pub(super) fn original_static_targets(
    unit: &ExecutableUnit,
    current: &HashSet<FunctionId>,
    runtime: &RuntimeContext,
    registries: &OriginalDependencyViews,
) -> Result<Vec<SelectedStaticTarget>, RunError> {
    if usable_original_targets(unit, runtime) {
        return Ok(unit.original_static_targets().to_vec());
    }
    let mut preselected = HashMap::new();
    let mut partitioned = HashSet::new();
    for import in unit.original_static_imports() {
        let instructions = match import.caller {
            ExecutableStaticCaller::Script => &unit.bytecode().instructions,
            ExecutableStaticCaller::Function(origin) => {
                let caller = unit
                    .local_function_for_origin(origin)
                    .map_err(error)?
                    .ok_or_else(|| {
                        error("original static caller is absent from its compiler product")
                    })?;
                &unit
                    .functions()
                    .get(caller)
                    .ok_or_else(|| error("original static caller body is absent"))?
                    .instructions
            }
        };
        let pc = usize::try_from(import.site.instruction)
            .map_err(|_| error("original static site exceeds the host index domain"))?;
        let instruction = instructions
            .get(pc)
            .ok_or_else(|| error("original static site is absent from its caller"))?;
        let (_, operand, _) = runmat_vm::instruction_reference_sites(instruction, pc)
            .map_err(error)?
            .into_iter()
            .find(|(site, _, _)| *site == import.site)
            .ok_or_else(|| error("original static reference is absent from its instruction"))?;
        // The origin selects an original admitted product. The numeric operand
        // only indexes this already-proved site in the compiler's selector.
        let mut matching = unit.original_dependencies().values().filter_map(|owner| {
            match owner.local_function_for_origin(import.target) {
                Ok(Some(function)) => Some(Ok((owner, function))),
                Ok(None) => None,
                Err(cause) => Some(Err(error(cause))),
            }
        });
        let (owner, function) = matching
            .next()
            .ok_or_else(|| error("original static target has no exact compiler owner"))??;
        for candidate in matching {
            let (candidate, candidate_function) = candidate?;
            if !std::ptr::eq(owner.as_ref(), candidate.as_ref()) || candidate_function != function {
                return Err(error(
                    "original static target has ambiguous compiler owners",
                ));
            }
        }
        let registry = registries
            .get(&(owner.as_ref() as *const ExecutableUnit))
            .ok_or_else(|| error("original dependency was not prepared before its caller"))?;
        let binding = registry
            .exact_bound_binding(function)
            .ok_or_else(|| error("original dependency lacks its exact bound graph"))?;
        let view = registry
            .owned_view(function, &binding.authority)
            .ok_or_else(|| error("original dependency lacks its owned execution view"))?;
        if let Some((previous, _)) = preselected.get(&operand) {
            if previous != &binding {
                return Err(error(
                    "original static operand has conflicting selected origins",
                ));
            }
        } else {
            preselected.insert(operand, (binding, view));
        }
        if import.kind == ExecutableStaticImportKind::PartitionedDefinition {
            partitioned.insert(operand);
        }
    }
    let selected = selected_static_imports_with_preselected(
        unit,
        &FunctionRegistry::default(),
        current,
        &partitioned,
        &preselected,
    )
    .map_err(error)?;
    if !selected.imports.iter().eq(unit.original_static_imports()) {
        return Err(error(
            "reissued static targets differ from original compiler selection",
        ));
    }
    Ok(selected.targets)
}

fn usable_original_targets(unit: &ExecutableUnit, runtime: &RuntimeContext) -> bool {
    !unit.original_static_targets().is_empty()
        && unit.original_static_targets().iter().all(|target| {
            target.source_owner.is_live() && target.source_owner.is_issued_in(runtime)
        })
}
