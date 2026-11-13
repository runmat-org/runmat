use std::sync::Arc;

use runmat_runtime::{build_runtime_error, RuntimeError};

use super::super::{
    private_source_owner::CapturedPrivateSourceOwners, DynamicFunctionCacheEntry,
    DynamicFunctionDependency,
};

pub(in crate::session) fn attach_dynamic_function_origins(
    bytecode: &mut runmat_vm::Bytecode,
    revision: &runmat_execution::ProgramRevision,
    source_map: &crate::ExecutableSourceMap,
) -> Result<(), runmat_execution::ContractError> {
    let private_owners = CapturedPrivateSourceOwners::new(source_map);
    for function in bytecode.function_registry.functions.values_mut() {
        let origin = match function.binding_origin {
            Some(origin) => origin,
            None => runmat_execution::FunctionBindingOrigin::from_revision(
                revision,
                function.portable_function,
            )?,
        };
        function.binding_origin = Some(origin);
        function.source_file = function.source_id.and_then(|source_id| {
            let ordinal = source_id.program()?;
            source_map
                .entries()
                .binary_search_by_key(&ordinal, |entry| entry.source_id)
                .ok()
                .and_then(|index| source_map.entries().get(index))
                .and_then(|entry| entry.full_path.clone())
        });
        function.private_source_owner = function.source_id.and_then(|id| private_owners.get(id));
    }
    bytecode.bound_functions = bytecode.function_registry.functions.clone();
    Ok(())
}

pub(in crate::session) fn dynamic_executable_revision(
    source: &crate::ExecutableSource,
    source_map: &crate::ExecutableSourceMap,
    compatibility: runmat_parser::CompatMode,
) -> Result<crate::ExecutableRevision, RuntimeError> {
    fn frame(bytes: &mut Vec<u8>, value: &[u8]) {
        bytes.extend_from_slice(&(value.len() as u64).to_be_bytes());
        bytes.extend_from_slice(value);
    }

    let mut graph = Vec::new();
    frame(&mut graph, b"runmat-dynamic-source-graph-v1");
    frame(&mut graph, source.owner_identity.as_bytes());
    frame(&mut graph, source.relative_path.as_bytes());
    let mut contents = Vec::new();
    frame(&mut contents, b"runmat-dynamic-source-content-v1");
    for entry in source_map.entries() {
        frame(&mut graph, entry.owner_identity.as_bytes());
        frame(&mut graph, entry.relative_path.as_bytes());
        frame(&mut contents, entry.owner_identity.as_bytes());
        frame(&mut contents, entry.relative_path.as_bytes());
        frame(&mut contents, entry.content_revision.to_string().as_bytes());
    }
    let source_digest = runmat_execution::Digest::sha256(&contents);
    let program_revision = runmat_execution::ProgramRevision::new(
        runmat_execution::Digest::sha256(&graph),
        source_digest,
        crate::program_environment(compatibility),
    )
    .map_err(|error| {
        build_runtime_error(error.to_string())
            .with_identifier("RunMat:ExecutableRevision")
            .with_source(error)
            .build()
    })?;
    Ok(crate::ExecutableRevision {
        program_revision,
        source_digest: source_digest.to_string(),
    })
}

/// A closure compiled after preceding runtime effects has a distinct
/// executable revision for its exact body and observed static targets. The
/// source text/revision remains unchanged; this domain contribution records
/// the late selected product basis without manufacturing another source.
pub(in crate::session) fn dynamic_pending_closure_revision(
    mut base: crate::ExecutableRevision,
    local_function: runmat_hir::FunctionId,
    body_fingerprint: &str,
    selected_basis: &[(runmat_hir::FunctionId, runmat_execution::Digest)],
) -> Result<crate::ExecutableRevision, RuntimeError> {
    let mut bytes = Vec::new();
    let frame = |bytes: &mut Vec<u8>, value: &[u8]| {
        bytes.extend_from_slice(&(value.len() as u64).to_be_bytes());
        bytes.extend_from_slice(value);
    };
    frame(&mut bytes, b"runmat-eval-late-closure-v1");
    frame(&mut bytes, &local_function.0.to_be_bytes());
    frame(&mut bytes, body_fingerprint.as_bytes());
    let mut selected_basis = selected_basis.to_vec();
    selected_basis.sort_by_key(|(alias, _)| alias.0);
    for (alias, product) in selected_basis {
        frame(&mut bytes, &alias.0.to_be_bytes());
        frame(&mut bytes, product.to_string().as_bytes());
    }
    let contribution = runmat_execution::DomainContribution::new(
        "eval-late-closure-v1",
        runmat_execution::Digest::sha256(&bytes),
    )
    .map_err(|error| {
        build_runtime_error(error.to_string())
            .with_identifier("RunMat:ExecutableRevision")
            .with_source(error)
            .build()
    })?;
    base.program_revision = base
        .program_revision
        .with_domain_contribution(contribution)
        .map_err(|error| {
            build_runtime_error(error.to_string())
                .with_identifier("RunMat:ExecutableRevision")
                .with_source(error)
                .build()
        })?;
    Ok(base)
}

pub(super) fn dynamic_product_dependencies(
    observed: &[super::super::compile::ObservedSourceRevision],
) -> Arc<[DynamicFunctionDependency]> {
    Arc::from(
        observed
            .iter()
            .map(|source| DynamicFunctionDependency {
                access_path: source.access_path.clone(),
                content_revision: source.content_revision,
                uses_pcode_reader: source.uses_pcode_reader,
            })
            .collect::<Vec<_>>(),
    )
}

pub(super) async fn reusable_dynamic_product(entry: &DynamicFunctionCacheEntry) -> bool {
    let dependencies = entry.dependencies.as_ref();
    for dependency in dependencies {
        let text = if dependency.uses_pcode_reader {
            runmat_runtime::builtins::io::repl_fs::pcode::read_source_text_async(
                &dependency.access_path,
            )
            .await
            .ok()
        } else {
            runmat_filesystem::read_to_string_async(&dependency.access_path)
                .await
                .ok()
        };
        let content_revision = text.map(|text| runmat_execution::Digest::sha256(text.as_bytes()));
        if content_revision != dependency.content_revision {
            return false;
        }
    }
    true
}
