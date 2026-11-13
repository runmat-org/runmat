use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::BundleCallable;

pub(super) struct CompiledImportedFunction {
    pub revision: ProgramRevision,
    pub callable: BundleCallable,
    pub compiled_binding: runmat_analysis_execution::named_callable::CompiledCallableBinding,
    pub function: runmat_mir::analysis::FunctionAnalysis,
    pub executable: runmat_execution::ExecutableUnitEnvelope,
}

pub(super) async fn compile_imported_function(
    source: &runmat_core::CheckedFeaSessionSource,
    module_path: &str,
    symbol: &str,
    expected_source_digest: Option<&str>,
) -> Result<CompiledImportedFunction, String> {
    let frozen = source.frozen_project().ok_or_else(|| {
        "imported FEA functions require an enclosing frozen RunMat project".to_string()
    })?;
    let module = runmat_analysis_language::LogicalSourcePath::new(module_path.to_string())
        .map_err(|error| error.to_string())?;
    let symbol = runmat_analysis_language::SourceSymbol::new(symbol.to_string())
        .map_err(|error| error.to_string())?;
    let expected_path =
        runmat_runtime::analysis::physical_source_path(&source.source_root, &module)
            .map_err(|error| error.to_string())?;
    let definition = runmat_package::resolve_visible_function_definition(
        frozen,
        source.physical_path(),
        &expected_path,
        symbol.as_str(),
    )?;
    let source_id = definition.source_id.as_ref().ok_or_else(|| {
        format!(
            "FEA function `{}` has no frozen source identity",
            definition.qualified_name
        )
    })?;
    let source_bytes = source
        .read_source(&definition.source_path)
        .await
        .map_err(|error| {
            format!(
                "cannot read frozen FEA function source {}: {error}",
                definition.source_path.display()
            )
        })?;
    let source_digest = Digest::sha256(&source_bytes);
    if source_digest.bytes() != source_id.content_digest.bytes() {
        return Err(format!(
            "FEA function `{}` changed after its project graph was frozen",
            definition.qualified_name
        ));
    }
    if let Some(expected) = expected_source_digest {
        let actual = source_digest.to_string();
        let actual = actual
            .strip_prefix("sha256:")
            .expect("execution digests always use the SHA-256 prefix");
        if expected != actual {
            return Err(format!(
                "FEA function `{}` differs from its checked source identity",
                definition.qualified_name
            ));
        }
    }
    let source_text = String::from_utf8(source_bytes).map_err(|_| {
        format!(
            "FEA function source {} is not UTF-8 MATLAB text",
            definition.source_path.display()
        )
    })?;
    let revision = ProgramRevision::new(
        Digest::from_bytes(*frozen.graph_digest().bytes()),
        Digest::from_bytes(*frozen.source_revision().bytes()),
        source.program_environment().clone(),
    )
    .map_err(|error| error.to_string())?;
    let mut session = runmat_core::RunMatSession::with_options(false, false)
        .map_err(|error| error.to_string())?;
    session.set_compat_mode(source.compatibility_mode());
    session
        .install_project_handoff(runmat_package::FrozenProjectHandoff::new(frozen.clone()))
        .await
        .map_err(|error| error.to_string())?;
    let executable_source = runmat_core::ExecutableSource::frozen(
        source_id.package_instance.to_string(),
        source_id.relative_path.as_str(),
        definition.source_path.to_string_lossy(),
        source_text,
    );
    let unit = match source.captured_sources() {
        Some(captured) => {
            session
                .compile_executable_unit_from_snapshot(
                    executable_source,
                    revision.clone(),
                    &runmat_package::FrozenProjectHandoff::new(frozen.clone()),
                    captured,
                )
                .await
        }
        None => {
            session
                .compile_executable_unit(executable_source, Some(revision.clone()))
                .await
        }
    }
    .map_err(|error| format!("cannot compile FEA function: {error}"))?;
    let executable = unit
        // The frozen resolver already selected this exact primary source.
        // Core owns source-entry/body/registry-name selection; package labels
        // must not be reparsed here as executable registry names.
        .portable_envelope()
        .map_err(|error| error.to_string())?;
    let function = unit
        .function_analysis(
            executable
                .manifest
                .identity
                .entrypoint_function
                .ok_or("FEA function product has no entrypoint function")?,
        )
        .cloned()
        .ok_or_else(|| "FEA function has no static function facts".to_string())?;
    let compiled_binding =
        runmat_analysis_execution::named_callable::CompiledCallableBinding::from_selected_identity(
            &executable.manifest.identity,
            &definition.qualified_name,
        )
        .map_err(|error| error.to_string())?;
    Ok(CompiledImportedFunction {
        revision,
        callable: BundleCallable {
            owner_identity: source_id.package_instance.to_string(),
            qualified_name: definition.qualified_name,
            source_digest,
        },
        compiled_binding,
        function,
        executable,
    })
}
