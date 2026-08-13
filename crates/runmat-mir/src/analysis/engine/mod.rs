mod call_graph;
mod flow;
mod interprocedural;
mod state;

pub(crate) use state::FlowState;

use runmat_types::{CallableFact, CallableIdentity, DynamicReason, ProgramFunctionId, ValueFact};

use crate::MirAssembly;

use super::{
    AnalysisDependency, AnalysisStore, ClassAnalysis, ClassPropertyAnalysis, FunctionAnalysis,
    FunctionConvergence,
};

pub fn analyze_assembly(assembly: &MirAssembly) -> AnalysisStore {
    let mut store = AnalysisStore {
        diagnostics: super::dataflow::diagnostics_for_assembly(assembly),
        ..AnalysisStore::default()
    };
    let program = interprocedural::analyze_program(assembly);
    store.diagnostics.extend(program.diagnostics.clone());
    store.dependencies = analysis_dependencies(&store);
    store.program_points.clear();
    store.functions.clear();
    store.classes.clear();
    store.mir_locals.clear();

    for (function, body) in &assembly.bodies {
        let Ok(function_ordinal) = u32::try_from(function.0) else {
            let span = assembly
                .functions
                .get(function)
                .map_or_else(runmat_hir::Span::default, |metadata| metadata.span);
            store.diagnostics.push(
                crate::MirDiagnostic::new(
                    "RM-MIR0010",
                    crate::MirDiagnosticSeverity::Error,
                    "function identity exceeds the portable analysis schema",
                    span,
                )
                .with_primary_label("this function cannot receive a stable program identity")
                .with_help("split the executable unit before analysis")
                .with_category("analysis-schema"),
            );
            continue;
        };
        let program_function = ProgramFunctionId(function_ordinal);
        let flow = flow::analyze_body(
            body,
            program_function,
            program.parameters.get(function).map_or(&[], Vec::as_slice),
            program.captures.get(function).map_or(&[], Vec::as_slice),
            &program.summaries,
        );
        flow::publish_legacy_projection(*function, &flow.final_state, &mut store.mir_locals);
        store.diagnostics.extend(flow.diagnostics.clone());
        store.program_points.extend(flow.points);
        let summary = program
            .summaries
            .get(function)
            .expect("interprocedural summary exists for every body");
        if !flow.converged || program.nonconverged.contains(function) {
            let span = assembly
                .functions
                .get(function)
                .map_or_else(runmat_hir::Span::default, |metadata| metadata.span);
            store.diagnostics.push(
                crate::MirDiagnostic::new(
                    "RM-MIR0011",
                    crate::MirDiagnosticSeverity::Warning,
                    "static analysis reached its convergence bound",
                    span,
                )
                .with_primary_label("facts in this function were conservatively widened")
                .with_help("simplify recursive or cyclic value flow to recover exact facts")
                .with_category("analysis-convergence"),
            );
        }
        store.functions.push(FunctionAnalysis {
            function: program_function,
            callable: CallableFact {
                identity: Some(CallableIdentity::BoundFunction(*function)),
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
            effects: summary.effects.clone(),
            capabilities: summary.capabilities.clone(),
            convergence: if !flow.converged || program.nonconverged.contains(function) {
                FunctionConvergence::DynamicRecursion
            } else if flow.widened || program.widened.contains(function) {
                FunctionConvergence::Widened
            } else {
                FunctionConvergence::Exact
            },
        });
    }

    store.program_points.sort_by_key(|point| point.point);
    store.functions.sort_by_key(|function| function.function);
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
    store
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
