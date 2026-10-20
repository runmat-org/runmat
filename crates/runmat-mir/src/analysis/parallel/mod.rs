mod classify;
mod facts;
mod legality;
mod patterns;
mod region;
mod spmd;

pub(super) use spmd::exports::{spmd_exit_projections, SpmdExit};

use runmat_types::{ParallelManifest, PARALLEL_MANIFEST_SCHEMA_VERSION};

use crate::{MirAssembly, MirDiagnostic};

use super::AnalysisStore;

pub(super) fn analyze_parallel_contracts(
    assembly: &MirAssembly,
    store: &AnalysisStore,
) -> (ParallelManifest, Vec<MirDiagnostic>) {
    let mut manifest = ParallelManifest {
        schema_version: PARALLEL_MANIFEST_SCHEMA_VERSION,
        parfor_regions: Vec::new(),
        spmd_regions: Vec::new(),
        distributed_values: Vec::new(),
        collectives: Vec::new(),
    };
    let mut diagnostics = Vec::new();
    let summaries = store
        .functions
        .iter()
        .filter_map(|analysis| {
            let function = usize::try_from(analysis.function.0)
                .ok()
                .map(runmat_hir::FunctionId)?;
            Some((
                function,
                super::inference::FunctionSummary {
                    outputs: analysis.outputs.clone(),
                    outputs_complete: analysis.callable.outputs_complete,
                    variadic_outputs: analysis.callable.variadic_outputs,
                    effects: analysis.effects.clone(),
                    capabilities: analysis.capabilities.clone(),
                },
            ))
        })
        .collect();

    for (function, body) in &assembly.bodies {
        let source = assembly
            .functions
            .get(function)
            .map(|metadata| metadata.source);
        let Ok(function) = u32::try_from(function.0).map(runmat_types::ProgramFunctionId) else {
            continue;
        };
        let diagnostic_start = diagnostics.len();
        spmd::classify_distributed_body(body, function, store, &mut manifest);
        classify::classify_body(
            body,
            function,
            store,
            &summaries,
            &mut manifest,
            &mut diagnostics,
        );
        spmd::classify_body(
            body,
            function,
            store,
            &summaries,
            &mut manifest,
            &mut diagnostics,
        );
        if let Some(source) = source {
            for diagnostic in &mut diagnostics[diagnostic_start..] {
                diagnostic.set_source(source);
            }
        }
    }

    manifest.parfor_regions.sort_by_key(|contract| contract.id);
    manifest.spmd_regions.sort_by_key(|contract| contract.id);
    manifest
        .distributed_values
        .sort_by_key(|contract| contract.id);
    manifest.collectives.sort_by_key(|contract| contract.id);
    if let Err(error) = manifest.validate() {
        diagnostics.push(
            MirDiagnostic::new(
                "RM-MIR0013",
                crate::MirDiagnosticSeverity::Error,
                format!("invalid parallel execution contract: {}", error.message),
                runmat_hir::Span::default(),
            )
            .with_primary_label(error.path)
            .with_category("parallel-contract"),
        );
    }
    (manifest, diagnostics)
}
