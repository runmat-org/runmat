//! Project compiler-selected static targets into native and script invocation facts.
use std::{collections::HashMap, rc::Rc};

use runmat_execution::executable::{ExecutableStaticCaller, ExecutableStaticImportKind};
use runmat_runtime::context::{RuntimeContext, SelectedGraphFunction};
use runmat_types::CallableIdentity;
use runmat_vm::FunctionRegistry;

use crate::{session::SelectedStaticTarget, ExecutableUnit};

pub(crate) fn bind_function_resolvers(
    runtime: RuntimeContext,
    unit: &ExecutableUnit,
    registry: Rc<FunctionRegistry>,
    targets: Vec<SelectedStaticTarget>,
) -> Result<RuntimeContext, String> {
    let scripts = unit.compiled_script_owner_ids()?;
    let current = unit.compiled_current_body_ids()?;
    let mut script_sites = HashMap::new();
    for target in &targets {
        if target.import.caller == ExecutableStaticCaller::Script
            && script_sites
                .insert(target.import.site, target.binding.clone())
                .is_some()
        {
            return Err("selected script repeats a static reference site".into());
        }
    }
    let runtime = if script_sites.is_empty() {
        runtime
    } else {
        runtime.with_selected_static_site_resolver(Rc::new(move |caller, site| {
            Ok(caller
                .is_none()
                .then(|| script_sites.get(&site).cloned())
                .flatten())
        }))
    };
    Ok(
        runtime.with_selected_graph_function_resolver(Rc::new(move |caller, identity| {
            if !scripts.contains(&caller) && !current.contains(&caller) {
                return None;
            }
            let (function, kind) = match identity {
                CallableIdentity::BoundFunction(function)
                | CallableIdentity::AnonymousFunction(function) => {
                    if let Some(binding) = registry.exact_bound_binding(*function) {
                        return Some(SelectedGraphFunction {
                            spawn_safety: registry
                                .get(*function)?
                                .spawn_contract
                                .as_ref()?
                                .safety()
                                .clone(),
                            binding,
                            capabilities: registry.get(*function)?.capabilities.clone(),
                        });
                    }
                    (*function, ExecutableStaticImportKind::PartitionedDefinition)
                }
                CallableIdentity::ExternalFunction { function, .. } => {
                    (*function, ExecutableStaticImportKind::ExplicitExternal)
                }
                _ => return None,
            };
            let caller = if scripts.contains(&caller) {
                ExecutableStaticCaller::Script
            } else {
                ExecutableStaticCaller::Function(registry.get_current(caller)?.binding_origin?)
            };
            let mut selected = None;
            for target in targets.iter().filter(|target| {
                target.import.caller == caller
                    && target.import.kind == kind
                    && target.function == function
            }) {
                if !target.source_owner.is_live()
                    || !target
                        .binding
                        .authority
                        .admits_local(&target.source_owner.session_lease())
                    || selected
                        .as_ref()
                        .is_some_and(|previous| previous != &target.binding)
                {
                    return None;
                }
                selected = Some(target.binding.clone());
            }
            let binding = selected?;
            let target = targets.iter().find(|target| target.binding == binding)?;
            let body = target
                .view
                .get_owned(function, binding.authority.local_definition()?)?;
            Some(SelectedGraphFunction {
                spawn_safety: body.spawn_contract.as_ref()?.safety().clone(),
                binding,
                capabilities: body.capabilities.clone(),
            })
        })),
    )
}
