mod classify;
mod facts;
mod region;

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

    for (function, body) in &assembly.bodies {
        let Ok(function) = u32::try_from(function.0).map(runmat_types::ProgramFunctionId) else {
            continue;
        };
        classify::classify_body(body, function, store, &mut manifest, &mut diagnostics);
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
