//! Exact parallel manifest ↔ compiled region verification shared by producer
//! and worker import. Optimizer pure-run regions have a different identity
//! domain even when their numeric ordinals happen to match.

use std::collections::BTreeSet;

use runmat_types::{ParallelManifest, ParallelRegionId, ProgramFunctionId};

use crate::{Bytecode, FunctionRegistry};

pub fn validate_compiled_parallel_contracts(
    bytecode: &Bytecode,
    registry: &FunctionRegistry,
    manifest: &ParallelManifest,
    selected_functions: Option<&BTreeSet<ProgramFunctionId>>,
) -> Result<(), String> {
    manifest
        .validate()
        .map_err(|error| format!("{}: {}", error.path, error.message))?;
    compare(
        manifest.parfor_regions.as_slice(),
        bytecode
            .parfor_regions
            .iter()
            .chain(
                registry
                    .functions
                    .values()
                    .flat_map(|body| body.parfor_regions.iter()),
            )
            .map(|region| &region.contract),
        |contract| contract.id,
        selected_functions,
        "parfor",
    )?;
    compare(
        manifest.spmd_regions.as_slice(),
        bytecode
            .spmd_regions
            .iter()
            .chain(
                registry
                    .functions
                    .values()
                    .flat_map(|body| body.spmd_regions.iter()),
            )
            .map(|region| &region.contract),
        |contract| contract.id,
        selected_functions,
        "SPMD",
    )
}

fn compare<'a, T: Eq + 'a>(
    manifest: &[T],
    compiled: impl Iterator<Item = &'a T>,
    id: impl Fn(&T) -> ParallelRegionId,
    selected_functions: Option<&BTreeSet<ProgramFunctionId>>,
    label: &str,
) -> Result<(), String> {
    let mut compiled = compiled
        .filter(|contract| {
            selected_functions.is_none_or(|functions| functions.contains(&id(contract).0.function))
        })
        .collect::<Vec<_>>();
    compiled.sort_unstable_by_key(|contract| id(contract));
    if compiled
        .windows(2)
        .any(|pair| id(pair[0]) == id(pair[1]) && pair[0] != pair[1])
    {
        return Err(format!(
            "compiled {label} region has conflicting body contracts"
        ));
    }
    // Entrypoint and registered-function views may both carry the very same
    // immutable body. Collapse only exact duplicates, never a conflicting ID.
    compiled.dedup_by(|right, left| id(right) == id(left));
    if compiled.len() != manifest.len()
        || compiled
            .iter()
            .zip(manifest)
            .any(|(actual, declared)| *actual != declared)
    {
        return Err(format!(
            "selected {label} manifest differs from its compiled region contracts"
        ));
    }
    Ok(())
}
