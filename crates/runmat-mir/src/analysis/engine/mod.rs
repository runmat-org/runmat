mod call_context;
mod call_graph;
mod flow;
mod interprocedural;
#[cfg(test)]
mod numerical_suspension_tests;
mod state;

pub use call_context::{diagnose_call_contexts, CallContextDiagnostics};
pub(crate) use call_graph::lexical_reference_graph;
pub(crate) use state::FlowState;

use runmat_types::{CallableFact, CallableIdentity, DynamicReason, ProgramFunctionId, ValueFact};

use crate::MirAssembly;

use super::{
    AnalysisDependency, AnalysisStore, ClassAnalysis, ClassPropertyAnalysis, FunctionAnalysis,
    FunctionConvergence,
};

fn certify_numerical_suspension(
    store: &mut AnalysisStore,
    evidence: &std::collections::BTreeMap<
        runmat_types::FunctionId,
        flow::NumericalSuspensionEvidence,
    >,
) {
    use std::collections::BTreeSet;
    let mut certified = BTreeSet::new();
    loop {
        let mut changed = false;
        for function in &store.functions {
            let Some(CallableIdentity::BoundFunction(local)) = function.callable.identity.as_ref()
            else {
                continue;
            };
            let Some(proof) = evidence.get(local) else {
                continue;
            };
            if proof.local
                && proof.callees.is_subset(&certified)
                && function.convergence == FunctionConvergence::Exact
                && function.capabilities.0.is_empty()
                && function.effects.0.iter().all(|effect| {
                    matches!(
                        effect,
                        runmat_types::EffectKind::MayThrow | runmat_types::EffectKind::MaySuspend
                    )
                })
            {
                changed |= certified.insert(*local);
            }
        }
        if !changed {
            break;
        }
    }
    for function in &mut store.functions {
        function.cooperative_suspension = matches!(function.callable.identity,
            Some(CallableIdentity::BoundFunction(local)) if certified.contains(&local));
    }
}

pub fn analyze_assembly(assembly: &MirAssembly) -> AnalysisStore {
    let mut store = AnalysisStore {
        diagnostics: super::dataflow::diagnostics_for_assembly(assembly),
        ..AnalysisStore::default()
    };
    let program = interprocedural::analyze_program(assembly);
    store.diagnostics.extend(program.diagnostics.clone());
    store.dependencies = analysis_dependencies(&store);
    store.program_points.clear();
    store.operation_contracts.clear();
    store.fusion_stage_contracts.clear();
    store.regions.clear();
    store.functions.clear();
    store.classes.clear();
    let mut liveness_by_function = std::collections::BTreeMap::new();
    let mut numerical_suspension = std::collections::BTreeMap::new();

    for (function, body) in &assembly.bodies {
        let source = assembly
            .functions
            .get(function)
            .map(|metadata| metadata.source);
        let Ok(function_ordinal) = u32::try_from(function.0) else {
            let span = assembly
                .functions
                .get(function)
                .map_or_else(runmat_hir::Span::default, |metadata| metadata.span);
            let diagnostic = crate::MirDiagnostic::new(
                "RM-MIR0010",
                crate::MirDiagnosticSeverity::Error,
                "function identity exceeds the portable analysis schema",
                span,
            )
            .with_primary_label("this function cannot receive a stable program identity")
            .with_help("split the executable unit before analysis")
            .with_category("analysis-schema");
            store.diagnostics.push(match source {
                Some(source) => diagnostic.with_source(source),
                None => diagnostic,
            });
            continue;
        };
        let program_function = ProgramFunctionId(function_ordinal);
        let flow = flow::analyze_body_with_suspension(
            body,
            program_function,
            program.parameters.get(function).map_or(&[], Vec::as_slice),
            program.captures.get(function).map_or(&[], Vec::as_slice),
            super::constructor_entry::instance_output_local(assembly, body),
            &program.summaries,
        );
        let mut operation_effects = runmat_types::EffectSet::default();
        let mut operation_capabilities = runmat_types::CapabilitySet::default();
        for contract in &flow.operation_contracts {
            operation_effects
                .0
                .extend(contract.effects.0.iter().copied());
            operation_capabilities
                .0
                .extend(contract.capabilities.0.iter().copied());
        }
        let convergence = if !flow.converged || program.nonconverged.contains(function) {
            FunctionConvergence::DynamicRecursion
        } else if flow.widened || program.widened.contains(function) {
            FunctionConvergence::Widened
        } else {
            FunctionConvergence::Exact
        };
        let liveness = super::regions::analyze_liveness(body);
        store.diagnostics.extend(
            flow.diagnostics
                .iter()
                .cloned()
                .map(|diagnostic| match source {
                    Some(source) => diagnostic.with_source(source),
                    None => diagnostic,
                }),
        );
        store
            .fusion_stage_contracts
            .extend(super::fusion_stage::certify_body(
                body,
                program_function,
                &flow.points,
                &program.summaries,
                &liveness,
                convergence,
            ));
        liveness_by_function.insert(*function, liveness);
        store.program_points.extend(flow.points);
        store.operation_contracts.extend(flow.operation_contracts);
        let summary = program
            .summaries
            .get(function)
            .expect("interprocedural summary exists for every body");
        operation_effects
            .0
            .extend(summary.effects.0.iter().copied());
        operation_capabilities
            .0
            .extend(summary.capabilities.0.iter().copied());
        if !flow.converged || program.nonconverged.contains(function) {
            let span = assembly
                .functions
                .get(function)
                .map_or_else(runmat_hir::Span::default, |metadata| metadata.span);
            let diagnostic = crate::MirDiagnostic::new(
                "RM-MIR0011",
                crate::MirDiagnosticSeverity::Warning,
                "static analysis reached its convergence bound",
                span,
            )
            .with_primary_label("facts in this function were conservatively widened")
            .with_help("simplify recursive or cyclic value flow to recover exact facts")
            .with_category("analysis-convergence");
            store.diagnostics.push(match source {
                Some(source) => diagnostic.with_source(source),
                None => diagnostic,
            });
        }
        numerical_suspension.insert(*function, flow.numerical_suspension);
        store.functions.push(FunctionAnalysis {
            function: program_function,
            captures: capture_contract(
                assembly,
                *function,
                program.captures.get(function).map_or(&[], Vec::as_slice),
            ),
            callable: CallableFact {
                identity: Some(CallableIdentity::BoundFunction(*function)),
                capabilities: operation_capabilities.clone(),
                parameters: program
                    .parameters
                    .get(function)
                    .cloned()
                    .unwrap_or_default(),
                parameters_complete: body.abi.varargin.is_none(),
                outputs: summary.outputs.clone(),
                outputs_complete: body.abi.varargout.is_none(),
                variadic_inputs: body.abi.varargin.is_some(),
                variadic_outputs: body.abi.varargout.is_some(),
                captures: program.captures.get(function).cloned().unwrap_or_default(),
                captures_complete: true,
            },
            outputs: summary.outputs.clone(),
            effects: operation_effects,
            cooperative_suspension: false,
            capabilities: operation_capabilities,
            convergence,
        });
    }

    store.program_points.sort_by_key(|point| point.point);
    store
        .operation_contracts
        .sort_by(|left, right| left.site.cmp(&right.site));
    store
        .fusion_stage_contracts
        .sort_by_key(|contract| contract.point);
    store.functions.sort_by_key(|function| function.function);
    certify_numerical_suspension(&mut store, &numerical_suspension);
    store.classes = assembly
        .classes
        .iter()
        .cloned()
        .map(|declaration| {
            let properties = declaration
                .properties
                .iter()
                .map(|property| ClassPropertyAnalysis {
                    property: property.name.clone(),
                    fact: ValueFact::unknown(DynamicReason::RuntimeValue),
                    has_default: property.has_default,
                })
                .collect();
            let methods = declaration
                .methods
                .iter()
                .filter_map(|method| u32::try_from(method.function.0).ok().map(ProgramFunctionId))
                .collect();
            ClassAnalysis {
                declaration,
                properties,
                methods,
            }
        })
        .collect();
    let mut regions = Vec::new();
    for (hir_function, body) in &assembly.bodies {
        let (Some(metadata), Ok(program_function)) = (
            assembly.functions.get(hir_function),
            u32::try_from(hir_function.0).map(ProgramFunctionId),
        ) else {
            continue;
        };
        regions.extend(super::regions::discover_regions(
            body,
            program_function,
            metadata.source,
            &program.summaries,
            &store,
            liveness_by_function
                .get(hir_function)
                .expect("liveness exists for every analyzed body"),
        ));
    }
    regions.sort_by_key(|region| region.contract.id);
    store.regions = regions;
    let (parallel, diagnostics) = super::parallel::analyze_parallel_contracts(assembly, &store);
    store.parallel = parallel;
    store.diagnostics.extend(diagnostics);
    store
}

fn capture_contract(
    assembly: &MirAssembly,
    function: runmat_types::FunctionId,
    values: &[ValueFact],
) -> Vec<super::FunctionCaptureAnalysis> {
    use super::{FunctionCaptureAnalysis, FunctionCaptureSource};
    if values.is_empty() {
        return Vec::new();
    }
    // Ordinary scalar leaves pay no extra lexical traversal. Capturing bodies
    // use the same projection that supplies both VM and Native execution.
    let metadata = assembly.functions.get(&function);
    let projection = assembly.lexical_environment_projection(function).ok();
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let source = match (
                metadata.and_then(|metadata| metadata.captures.get(index)),
                projection.as_ref(),
            ) {
                (Some(capture), Some(projection)) => {
                    if projection
                        .inherited_storage
                        .iter()
                        .any(|entry| entry.binding == capture.binding)
                    {
                        FunctionCaptureSource::CanonicalStorage(capture.binding)
                    } else if projection.inherited.contains(&capture.binding) {
                        FunctionCaptureSource::Lexical(capture.binding)
                    } else if projection.transit.contains(&capture.binding) {
                        // An anonymous snapshot and its descendant's live cell
                        // are distinct authorities, not one certified value.
                        FunctionCaptureSource::Unknown
                    } else {
                        FunctionCaptureSource::ByValue
                    }
                }
                _ => FunctionCaptureSource::Unknown,
            };
            FunctionCaptureAnalysis {
                source,
                value: value.clone(),
            }
        })
        .collect()
}

fn analysis_dependencies(store: &AnalysisStore) -> Vec<AnalysisDependency> {
    vec![
        AnalysisDependency {
            identity: "runmat-types".to_string(),
            revision: format!(
                "{}.{}",
                store.revision.fact_schema_major, store.revision.fact_schema_minor
            ),
            invalidates: vec![runmat_types::InvalidationCause::DependencyChanged],
        },
        AnalysisDependency {
            identity: "runmat-builtins-catalog".to_string(),
            revision: store
                .revision
                .catalog_fingerprint
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect(),
            invalidates: vec![runmat_types::InvalidationCause::CatalogChanged],
        },
    ]
}
