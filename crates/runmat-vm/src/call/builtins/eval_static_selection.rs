//! Invocation-local source selection at a proven closure issuance site.

use std::collections::HashMap;

use runmat_hir::FunctionId;
use runmat_runtime::context::RuntimeContext;
use runmat_runtime::user_functions::{
    observe_runtime_name, RuntimeNameCandidate, RuntimeNameFilter, RuntimeNameQuery,
    RuntimeNameRoute,
};
use runmat_runtime::RuntimeError;
use runmat_value::BoundFunctionBinding;

use super::eval_hir_sites::HirUnresolvedSite;
use crate::bytecode::FunctionRegistry;
use crate::interpreter::errors::mex;

pub struct EvalSelectedSource {
    pub alias: FunctionId,
    pub binding: BoundFunctionBinding,
    pub fixed_outputs: usize,
    pub has_varargout: bool,
}

pub struct EvalSelectedExternalSource {
    pub alias: FunctionId,
    pub target: runmat_runtime::user_functions::SelectedExternalSourceTarget,
}

/// A no-execute installed native selection at an exact unresolved HIR site.
/// The identity is only a claim until Core matches the original prepared
/// project scope and seals the selected function-only unit.
pub(crate) struct EvalSelectedNative {
    pub(crate) site: HirUnresolvedSite,
    pub(crate) artifact: runmat_runtime::user_functions::InstalledNativeArtifactFact,
}

#[derive(Default)]
pub(super) struct EvalSelectedSources {
    pub names: HashMap<String, FunctionId>,
    pub sources: Vec<EvalSelectedSource>,
    pub external_sources: Vec<EvalSelectedExternalSource>,
    pub natives: Vec<EvalSelectedNative>,
}

/// These sites came from one HIR function whose closure-creation opcode is
/// executing now. Their display names are not authority: the canonical
/// observer runs after preceding effects and only an exact Source candidate
/// enters the existing no-invoke Handle loader. A non-source route remains
/// dynamic. No unreachable closure performs an observation.
pub(super) async fn select_sites(
    sites: &[HirUnresolvedSite],
    alias_floor: usize,
    runtime: &RuntimeContext,
    inherited: &FunctionRegistry,
    assigned_workspace: &HashMap<String, usize>,
    admission: &crate::interpreter::dynamic_admission::DynamicEvalAdmissionPort,
) -> Result<EvalSelectedSources, RuntimeError> {
    let mut names = sites
        .iter()
        .map(|site| site.name.as_str())
        .collect::<Vec<_>>();
    names.sort();
    names.dedup();
    let reserved = inherited
        .functions
        .keys()
        .map(|function| function.0)
        .max()
        .unwrap_or(0)
        .max(alias_floor);
    let mut next_alias = reserved.checked_add(1).ok_or_else(|| {
        mex(
            "DynamicWorkspaceCompile",
            "eval static source ID domain is exhausted",
        )
    })?;
    let mut selected_names = HashMap::new();
    let mut sources = Vec::new();
    let mut external_sources = Vec::new();
    let mut natives = Vec::new();
    for name in names {
        if assigned_workspace.contains_key(name) {
            continue;
        }
        let observed = runtime
            .scope(observe_runtime_name(
                name,
                RuntimeNameQuery {
                    route: RuntimeNameRoute::DynamicCall,
                    filter: RuntimeNameFilter::Any,
                    all: false,
                },
            ))
            .await
            .map_err(|error| {
                mex(
                    "DynamicWorkspaceSource",
                    &format!("eval source observation failed: {error}"),
                )
            })?;
        if let Some(RuntimeNameCandidate::Semantic {
            function,
            source: Some(source),
            definition_owner: None,
        }) = observed.selected()
        {
            if let Some(origin) = source.binding_origin {
                let target =
                    runmat_runtime::user_functions::selected_external_source_target(origin)
                        .filter(|target| target.function() == *function && target.is_live())
                        .ok_or_else(|| {
                            mex(
                                "DynamicWorkspaceSource",
                                "selected semantic source lacks exact linked target authority",
                            )
                        })?;
                let alias = FunctionId(next_alias);
                next_alias = next_alias.checked_add(1).ok_or_else(|| {
                    mex(
                        "DynamicWorkspaceCompile",
                        "eval static source ID domain is exhausted",
                    )
                })?;
                selected_names.insert(name.to_owned(), alias);
                external_sources.push(EvalSelectedExternalSource { alias, target });
                continue;
            }
        }
        if let Some(RuntimeNameCandidate::Native(native)) = observed.selected() {
            let Some(artifact) = native.installed_artifact.as_ref() else {
                continue;
            };
            if !native.installed || native.origin.is_some() {
                return Err(mex(
                    "DynamicWorkspaceSource",
                    "installed native selection has inconsistent provenance",
                ));
            }
            artifact.interop.validate().map_err(|error| {
                mex(
                    "DynamicWorkspaceSource",
                    &format!("native artifact fact is invalid: {error}"),
                )
            })?;
            if !artifact
                .interop
                .adapters
                .iter()
                .any(|requirement| requirement.artifact_identities.contains(&artifact.identity))
            {
                return Err(mex(
                    "DynamicWorkspaceSource",
                    "installed native fact does not grant its artifact identity",
                ));
            }
            natives.extend(
                sites
                    .iter()
                    .filter(|site| site.name == name)
                    .cloned()
                    .map(|site| EvalSelectedNative {
                        site,
                        artifact: artifact.clone(),
                    }),
            );
            continue;
        }
        let selected = match observed.selected() {
            Some(RuntimeNameCandidate::Semantic {
                function,
                definition_owner: Some(owner),
                ..
            }) => {
                // A warmed source is already semantic, so reloading by name
                // would discard the observation and could select a replacement.
                // Prefer the exact inherited view (including retained private
                // bodies); earlier eval effects may instead have published it
                // in the session's current generation after eval entry.
                let function = FunctionId(*function);
                let binding = inherited.exact_bound_binding(function).filter(|binding| {
                    binding
                        .authority
                        .local_definition()
                        .is_some_and(|lease| lease.matches(owner))
                });
                Some(match binding {
                    Some(binding) => binding,
                    None => admission
                        .host()
                        .selected_semantic_source_binding(runtime, function, owner)?,
                })
            }
            Some(RuntimeNameCandidate::Source(candidate)) => {
                runmat_runtime::user_functions::select_observed_source_binding_in_context(
                    runtime.clone(),
                    name.to_owned(),
                    candidate.clone(),
                )
                .await?
            }
            _ => continue,
        };
        if runtime.search_path().map(|path| path.generation()) != observed.captured_path_revision
            || runmat_runtime::class_registry::class_registry_generation()
                != observed.captured_class_revision
        {
            return Err(mex(
                "DynamicWorkspaceSource",
                "eval source selection changed during precompile admission",
            ));
        }
        let binding = selected.ok_or_else(|| {
            mex(
                "DynamicWorkspaceSource",
                "observed source has no exact selected definition",
            )
        })?;
        let (fixed_outputs, has_varargout) =
            admission.host().selected_source_output_arity(&binding)?;
        let alias = FunctionId(next_alias);
        next_alias = next_alias.checked_add(1).ok_or_else(|| {
            mex(
                "DynamicWorkspaceCompile",
                "eval static source ID domain is exhausted",
            )
        })?;
        selected_names.insert(name.to_owned(), alias);
        sources.push(EvalSelectedSource {
            alias,
            binding,
            fixed_outputs,
            has_varargout,
        });
    }
    Ok(EvalSelectedSources {
        names: selected_names,
        sources,
        external_sources,
        natives,
    })
}
