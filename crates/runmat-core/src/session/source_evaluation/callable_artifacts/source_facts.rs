use std::collections::HashSet;
use std::rc::Rc;
use std::sync::Arc;

use runmat_execution::FunctionBindingOrigin;
use runmat_runtime::{build_runtime_error, RuntimeError};

pub(super) fn exact_origin_resolver(
    facts: Arc<Vec<runmat_runtime::user_functions::SourceFunctionInfo>>,
    selected: Rc<runmat_vm::FunctionRegistry>,
) -> Rc<runmat_runtime::user_functions::ExecutionSourceFacts> {
    let by_function = Arc::clone(&facts);
    let by_source = Arc::clone(&facts);
    let by_owners = Arc::clone(&facts);
    let by_origin = facts;
    let by_static_sites = Rc::clone(&selected);
    Rc::new(
        runmat_runtime::user_functions::ExecutionSourceFacts::new(
            Rc::new(move |function| {
                by_function
                    .iter()
                    .find(|fact| fact.function == function)
                    .cloned()
            }),
            Rc::new(move |source| {
                by_source
                    .iter()
                    .filter(|fact| fact.source_id == source)
                    .cloned()
                    .collect()
            }),
            Rc::new(move |function| {
                by_owners
                    .iter()
                    .find(|fact| fact.function == function)
                    .and_then(|fact| fact.definition_owners.clone())
            }),
            Rc::new(move |origin| unique_origin(&by_origin, origin)),
        )
        .with_static_sites(Rc::new(move |function, info, owner| {
            let info = info.filter(|info| info.function == function)?;
            by_static_sites.selected_static_site_leases_for_issued_info(info, owner)
        }))
        .with_selected_class(Rc::new(move |binding| {
            selected.selected_class_projection_for_binding(binding)
        })),
    )
}

fn unique_origin(
    facts: &[runmat_runtime::user_functions::SourceFunctionInfo],
    origin: FunctionBindingOrigin,
) -> Option<runmat_runtime::user_functions::SourceFunctionInfo> {
    let mut matches = facts
        .iter()
        .filter(|fact| fact.binding_origin == Some(origin));
    let fact = matches.next()?.clone();
    matches.next().is_none().then_some(fact)
}

pub(super) fn ensure_all_origins(
    origins: &HashSet<FunctionBindingOrigin>,
    facts: &[runmat_runtime::user_functions::SourceFunctionInfo],
) -> Result<(), RuntimeError> {
    if origins
        .iter()
        .all(|origin| unique_origin(facts, *origin).is_some())
    {
        return Ok(());
    }
    Err(
        build_runtime_error("callable artifact lacks one exact bound definition")
            .with_identifier("RunMat:CallableArtifact:MissingDefinition")
            .build(),
    )
}
