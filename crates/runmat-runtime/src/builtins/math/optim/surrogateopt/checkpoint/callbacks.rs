//! Callback roles retain either an actual compiled builtin identity or one
//! root of the shared admitted user-code/lexical-capture graph.

use super::{error, with_source};
use crate::{
    builtins::math::optim::common::ScalarCooperation,
    call::descriptor::RepeatableCallable,
    execution::{DeferredGraphProgram, RootedValue},
    BuiltinResult,
};
use runmat_execution::value::{PortableSelectedCode, ValueLimits};
use runmat_types::BuiltinId;
use runmat_value::Value;
mod wire;

pub(in super::super) struct ArchivedCallbacks {
    code: Option<PortableSelectedCode>,
    roles: Vec<Role>,
    outputs: usize,
    plots: usize,
}

enum Role {
    Builtin(BuiltinId),
    GraphRoot(usize),
}
fn portable_values(graph: &runmat_execution::value::PayloadGraph) -> BuiltinResult<()> {
    if graph.nodes.iter().any(|node| {
        matches!(node, runmat_execution::value::GraphNode::Leaf { value }
        if !matches!(value, runmat_execution::value::ValuePayload::Inline(_)))
    }) {
        return Err(error("checkpoint captures require a portable value owner, not an old session or resident service reference"));
    }
    Ok(())
}

pub(in super::super) struct RestoredCallbacks {
    pub source: RootedValue,
    pub objective: RepeatableCallable,
    pub outputs: Vec<RepeatableCallable>,
    pub plots: Vec<RepeatableCallable>,
}

impl ArchivedCallbacks {
    pub(in super::super) fn capture<'a>(
        objective: &'a RepeatableCallable,
        outputs: &'a [RepeatableCallable],
        plots: &'a [RepeatableCallable],
        c: &ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let runtime = c
            .runtime()
            .ok_or_else(|| error("checkpoint callbacks require an original runtime"))?;
        let total = outputs
            .len()
            .checked_add(plots.len())
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| error("checkpoint callback role count overflow"))?;
        let mut descriptors = super::super::public_values::capacity(total)?;
        let mut roles = super::super::public_values::capacity(total)?;
        for selected in std::iter::once(objective).chain(outputs).chain(plots) {
            if let Some(name) = selected.selected_target().selected_builtin_name() {
                roles.push(Role::Builtin(BuiltinId(name.to_owned())));
            } else {
                roles.push(Role::GraphRoot(descriptors.len()));
                descriptors.push(selected.selected_descriptor().clone());
            }
        }
        let code = if descriptors.is_empty() {
            None
        } else {
            let program = runtime
                .selected_graph_program()
                .ok_or_else(|| error("checkpoint has no original selected code producer"))?
                .clone()
                .with_validation_workspace(
                    crate::execution::value_codec::ValidationWorkspaceOwner::capture(runtime),
                );
            let capture = program
                .capture_callables(&descriptors, ValueLimits::default())
                .map_err(|source| {
                    with_source("cannot capture selected checkpoint callbacks", source)
                })?;
            portable_values(&capture.graph)?;
            // Current portable restore has no freshly materialized exact-unit
            // foreign service. Refuse declared interop before file effects;
            // an empty issued policy alone does not require a foreign port.
            for bytes in std::iter::once(&capture.primary_unit).chain(&capture.retained_units) {
                let unit = runmat_execution::ExecutableUnitEnvelope::from_canonical_bytes(bytes)
                    .map_err(|source| with_source("selected checkpoint code is invalid", source))?;
                let interop = &unit.manifest.interop;
                if !interop.adapters.is_empty()
                    || !interop.foreign_types.is_empty()
                    || !interop.adapter_contracts.is_empty()
                {
                    return Err(error(
                        "checkpoint callbacks require a freshly selected foreign-unit projection",
                    ));
                }
            }
            Some(capture.into_portable())
        };
        Ok(Self {
            code,
            roles,
            outputs: outputs.len(),
            plots: plots.len(),
        })
    }

    pub(in super::super) fn restore(
        self,
        c: &ScalarCooperation,
    ) -> BuiltinResult<RestoredCallbacks> {
        let total = self
            .outputs
            .checked_add(self.plots)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| error("checkpoint callback role count overflow"))?;
        if total != self.roles.len() {
            return Err(error("checkpoint callback roles have an invalid extent"));
        }
        let mut graph_roots = 0;
        for role in &self.roles {
            match role {
                Role::Builtin(builtin) => {
                    if runmat_builtins::builtin_catalog_entry_by_name(&builtin.0)
                        .is_none_or(|entry| entry.identity.name != builtin.0)
                    {
                        return Err(error(
                            "checkpoint builtin identity is absent from this runtime",
                        ));
                    }
                }
                Role::GraphRoot(index) => {
                    if *index != graph_roots {
                        return Err(error(
                            "checkpoint callback graph roles are not in source order",
                        ));
                    }
                    graph_roots += 1;
                }
            }
        }
        if self.code.as_ref().map_or(0, |code| code.graph.roots.len()) != graph_roots
            || self.code.is_some() != (graph_roots != 0)
        {
            return Err(error(
                "checkpoint callback roles disagree with the captured roots",
            ));
        }
        let runtime = c
            .runtime()
            .ok_or_else(|| error("checkpoint restoration requires its current original runtime"))?;
        let restored = match self.code {
            Some(code) => {
                portable_values(&code.graph)?;
                let program = runtime.selected_graph_program().ok_or_else(|| {
                    error("checkpoint restoration has no original selected program")
                })?;
                Some(program.restore_callables(runtime, code)?)
            }
            None => None,
        };
        let mut selected = super::super::public_values::capacity(total)?;
        let mut objective_source = None;
        for role in self.roles {
            let (source, callback) = match role {
                Role::Builtin(builtin) => {
                    let source = RootedValue::in_context(
                        Value::string(crate::call::argument_text::copy_str(
                            &builtin.0,
                            Some(runtime),
                        )?),
                        runtime,
                    )?;
                    let callback =
                        DeferredGraphProgram::restore_compiled_builtin(runtime, builtin)?;
                    (source, callback)
                }
                Role::GraphRoot(index) => {
                    let graph = restored
                        .as_ref()
                        .ok_or_else(|| error("checkpoint callback graph is absent"))?;
                    let value = graph
                        .values
                        .values()
                        .get(index)
                        .ok_or_else(|| error("checkpoint callable root is absent"))?;
                    let source = RootedValue::in_context(value.clone(), runtime)?;
                    let callback = graph
                        .program
                        .select_repeatable_callable(runtime, source.clone())?;
                    (source, callback)
                }
            };
            if objective_source.is_none() {
                objective_source = Some(source);
            }
            selected.push(callback);
        }
        let mut selected = selected.into_iter();
        let objective = selected
            .next()
            .ok_or_else(|| error("checkpoint objective is absent"))?;
        let outputs = selected.by_ref().take(self.outputs).collect();
        let plots = selected.collect();
        Ok(RestoredCallbacks {
            source: objective_source
                .ok_or_else(|| error("checkpoint objective source is absent"))?,
            objective,
            outputs,
            plots,
        })
    }
}
