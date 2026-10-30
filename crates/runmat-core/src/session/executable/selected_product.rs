//! Direct execution retains the same immutable product that deferred calls use.
use std::collections::HashSet;
use std::rc::Rc;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;

use runmat_runtime::context::RuntimeContext;
use runmat_vm::{Bytecode, SelectedExecutableProduct};

use crate::session::source_evaluation::{
    SelectedGraphCodeSource, SelectedGraphPrimary, SelectedGraphPublication,
    SelectedGraphSelection, SelectedGraphSource,
};
use crate::session::RunMatSession;
use crate::{ExecutableUnit, RunError};

use super::runtime_error;

#[path = "selected_product/dependencies.rs"]
mod dependencies;

impl RunMatSession {
    pub(super) fn prepare_executable_graph(
        &mut self,
        unit: &ExecutableUnit,
        runtime: &RuntimeContext,
        bytecode: &mut Bytecode,
    ) -> Result<SelectedGraphPrimary, RunError> {
        let dependencies = self.prepare_original_dependency_views(unit, runtime)?;
        self.prepare_executable_graph_with_dependencies(unit, runtime, bytecode, &dependencies)
    }

    pub(super) fn prepare_executable_graph_with_dependencies(
        &mut self,
        unit: &ExecutableUnit,
        runtime: &RuntimeContext,
        bytecode: &mut Bytecode,
        dependencies: &dependencies::OriginalDependencyViews,
    ) -> Result<SelectedGraphPrimary, RunError> {
        let error = |message| runtime_error("RunMat:ExecutableSourceIdentity", message);
        let current = unit.compiled_current_body_ids().map_err(error)?;
        // Source leases exist, but no selected executable/static capture has
        // escaped. Retain the exact original-to-issued constructor relation
        // until its immutable declaration product is published below.
        let constructors = unit
            .materialize_runtime_constructor_definitions(
                &mut bytecode.function_registry,
                Some(&mut bytecode.bound_functions),
            )
            .map_err(error)?;
        let targets = dependencies::original_static_targets(unit, &current, runtime, dependencies)?;
        let names = bytecode
            .function_registry
            .names
            .iter()
            .filter(|(_, id)| current.contains(id))
            .map(|(name, id)| (name.clone(), *id))
            .collect();
        // Foreign compiler projections carry this unit's portable source
        // ordinals, not the original dependency's issued runtime references.
        // Execute those definitions only through their retained selected view.
        let mut bodies = std::mem::take(&mut bytecode.function_registry).into_functions();
        bodies.retain(|id, _| current.contains(id));
        let mut registry = runmat_vm::FunctionRegistry::new(bodies);
        registry.names = names;
        for target in &targets {
            registry
                .retain_selected_site_view(&target.view, &target.binding)
                .map_err(|cause| error(cause.to_owned()))?;
        }
        bytecode.function_registry = registry;
        bytecode
            .bound_functions
            .retain(|id, _| current.contains(id));
        #[cfg(not(target_arch = "wasm32"))]
        let prepared = self
            .prepare_selected_project_foreign(&unit.inferred_interop_manifest())
            .map_err(error)?;
        #[cfg(not(target_arch = "wasm32"))]
        let interop = prepared
            .as_ref()
            .map(|prepared| prepared.interop().clone())
            .unwrap_or_else(runmat_types::InteropManifest::empty);
        #[cfg(target_arch = "wasm32")]
        let interop = runmat_types::InteropManifest::empty();
        #[cfg(not(target_arch = "wasm32"))]
        let basis = if interop == runmat_types::InteropManifest::empty() {
            None
        } else {
            prepared
                .as_ref()
                .map(|prepared| prepared.selected_reuse_basis())
                .transpose()
                .map_err(|cause| error(cause.to_string()))?
        };
        #[cfg(target_arch = "wasm32")]
        let basis = None;
        let cached_code = unit.cached_direct_code(&interop);
        let selected = SelectedGraphPrimary::new(
            SelectedGraphSource {
                unit: Rc::new(unit.clone()),
                primary_source: Some(unit.primary_source_id().map_err(error)?),
                code: cached_code
                    .map(SelectedGraphCodeSource::Admitted)
                    .unwrap_or(SelectedGraphCodeSource::Unpublished),
            },
            runtime,
            bytecode.function_registry.clone(),
            SelectedGraphPublication::new(&bytecode.function_registry, &HashSet::new()),
            SelectedGraphSelection {
                current_functions: current,
                interop: interop.clone(),
                static_targets: targets,
            },
        )
        .map_err(error)?;
        let code = unit
            .admit_direct_code(interop, || selected.canonical_v3())
            .map_err(error)?;
        #[cfg(not(target_arch = "wasm32"))]
        let receipt = if code.requires_interop() {
            crate::foreign_artifacts::selected_unit_interop_receipt(
                Arc::clone(code.canonical_v3()),
                selected.source_owner(),
                prepared.as_ref(),
            )
            .map_err(error)?
        } else {
            None
        };
        #[cfg(target_arch = "wasm32")]
        let receipt = None;
        let product = SelectedExecutableProduct::bind_admitted_code(
            code,
            selected.source_owner().clone(),
            basis,
            receipt,
            Some(self.source_evaluator.dynamic_admission_port()),
        )
        .map_err(error)?;
        selected
            .with_current_product(bytecode, product, &constructors)
            .map_err(error)
    }
}
