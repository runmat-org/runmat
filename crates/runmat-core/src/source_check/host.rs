use super::*;
use runmat_config::runtime::{HostConfigDocument, HostRuntimeSettings, ResolvedRuntimeConfig};
use runmat_hir::{HirDiagnostic, HirDiagnosticSeverity, Span};
use runmat_package::{DiscoveredSourceSymbols, FrozenProject};
use runmat_static_analysis::project::{
    analyze_project_source, check_report::project_diagnostic, ProjectSource,
};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

/// Capture the original provider/path once. Select it locally during each poll;
/// no process provider mutation or filesystem lock crosses suspension. Selected
/// invocation authority may not be masked by a different snapshot/provider.
pub async fn check_source(request: SourceCheckRequest) -> Result<SourceCheckReport, String> {
    check_source_with_runtime_settings(request, |settings| settings.resolve().map_err(display))
        .await
}

/// Check captured source using the embedding host's canonical settings owner.
/// Core captures and validates the selected project document before calling
/// `resolve`. A native host can add its OS-user/environment/explicit layers
/// without substituting source inputs or relabelling resolved project settings
/// as the user layer. The default browser/SDK checker uses `settings.resolve()`.
pub async fn check_source_with_runtime_settings<F>(
    mut request: SourceCheckRequest,
    resolve: F,
) -> Result<SourceCheckReport, String>
where
    F: FnOnce(HostRuntimeSettings) -> Result<ResolvedRuntimeConfig, String>,
{
    let inputs = request.input_manifest.take();
    let mut paths = vec![request.target.path()];
    paths.extend(request.search_paths.iter().map(PathBuf::as_path));
    let (scope, resolved) =
        runmat_filesystem::CapturedReadScope::capture(&paths, inputs).map_err(display)?;
    let path = resolved[0].clone();
    request.search_paths = resolved.into_iter().skip(1).collect();
    scope
        .run(async {
            let provider = runmat_filesystem::current_provider();
            check_captured(request, path, provider, resolve).await
        })
        .await
        .map_err(display)?
}

async fn check_captured<F>(
    request: SourceCheckRequest,
    path: PathBuf,
    provider: Arc<dyn runmat_filesystem::FsProvider>,
    resolve: F,
) -> Result<SourceCheckReport, String>
where
    F: FnOnce(HostRuntimeSettings) -> Result<ResolvedRuntimeConfig, String>,
{
    let project = runmat_package::discover_frozen_project_from_async(&path, Default::default())
        .await
        .map_err(display)?;
    if let Some(project) = &project {
        runmat_package::FrozenProjectHandoff::new(project.clone())
            .validate()
            .map_err(display)?;
    }
    let project_document = if let Some(project) = &project {
        let source = read(provider.as_ref(), &project.manifest_path).await?;
        let format = runmat_config::document::RunmatConfigFormat::from_path(&project.manifest_path)
            .map_err(display)?;
        let manifest = match format {
            runmat_config::document::RunmatConfigFormat::Toml => {
                runmat_config::project::parse_project_manifest_toml(&source).map_err(display)?
            }
            runmat_config::document::RunmatConfigFormat::Json => {
                runmat_config::project::parse_project_manifest_json(&source).map_err(display)?
            }
        };
        project.validate_root_manifest(&manifest)?;
        Some(HostConfigDocument { source, format })
    } else {
        None
    };
    let (kind, selected) = match &request.target {
        CheckTarget::Script { .. } => ("script", vec![(path.clone(), None)]),
        CheckTarget::Function { function, .. } => {
            if function.is_empty() {
                return Err("check function name must not be empty".into());
            }
            ("function", vec![(path.clone(), Some(function.clone()))])
        }
        CheckTarget::Project { .. } => {
            let project = project
                .as_ref()
                .ok_or("whole-project checking requires a RunMat project manifest")?;
            let mut paths: Vec<_> = project
                .all_sources()
                .filter(|(source, path)| {
                    source.id.package_instance == project.graph.root
                        && path
                            .extension()
                            .and_then(|s| s.to_str())
                            .is_some_and(|s| s.eq_ignore_ascii_case("m"))
                })
                .map(|(_, path)| (path.clone(), None))
                .collect();
            paths.sort();
            paths.dedup();
            ("project", paths)
        }
    };
    for (selected, _) in &selected {
        if !selected
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|s| s.eq_ignore_ascii_case("m"))
        {
            return Err("ordinary source checks require an indexed .m source".into());
        }
    }
    // Validate and retain the requested immutable source before entering the
    // embedding host's resolver. Move those bytes into analysis below rather
    // than reading them twice or falling back to a live workspace version.
    let mut primary_content = match &request.target {
        CheckTarget::Script { .. } | CheckTarget::Function { .. } => {
            Some(captured_content(&path, project.as_ref(), provider.as_ref()).await?)
        }
        CheckTarget::Project { .. } => None,
    };
    let settings = HostRuntimeSettings {
        user_settings_json: request.user_settings_json.clone(),
        project: project_document,
        policy: request.policy,
    };
    let config = resolve(settings)?.config;
    let compat = match config.language.compat {
        LanguageCompatMode::RunMat => crate::CompatMode::RunMat,
        LanguageCompatMode::Matlab => crate::CompatMode::Matlab,
        LanguageCompatMode::Strict => crate::CompatMode::Strict,
    };
    let mut documents = Vec::with_capacity(selected.len());
    for (selected, function) in selected {
        let content = match primary_content.take() {
            Some(content) => content,
            None => captured_content(&selected, project.as_ref(), provider.as_ref()).await?,
        };
        documents.push(
            check_document(
                &selected,
                content,
                function,
                &request,
                project.as_ref(),
                provider.clone(),
                compat,
            )
            .await?,
        );
    }
    let errors = documents.iter().map(|d| d.summary.errors).sum();
    let warnings = documents.iter().map(|d| d.summary.warnings).sum();
    Ok(SourceCheckReport {
        schema_version: 1,
        document_kind: kind,
        path,
        outcome: outcome(errors, warnings, request.deny_warnings),
        language_compatibility: config.language.compat,
        project_revision: project.as_ref().map(FrozenProject::revision),
        documents,
        summary: CheckSummary {
            errors,
            warnings,
            warnings_denied: request.deny_warnings,
        },
    })
}

async fn captured_content(
    path: &Path,
    frozen: Option<&FrozenProject>,
    provider: &dyn runmat_filesystem::FsProvider,
) -> Result<String, String> {
    match frozen {
        Some(project) => Ok(
            crate::ExecutableSource::capture_project_primary(project, path)
                .await?
                .text,
        ),
        None => read(provider, path).await,
    }
}

async fn check_document(
    path: &Path,
    content: String,
    function: Option<String>,
    request: &SourceCheckRequest,
    frozen: Option<&FrozenProject>,
    provider: Arc<dyn runmat_filesystem::FsProvider>,
    compat: crate::CompatMode,
) -> Result<CheckedDocument, String> {
    let directory = path
        .parent()
        .ok_or("checked source has no parent directory")?;
    let mut catalog = match frozen {
        Some(project) => Some(runmat_package::source_symbols_from_frozen(project, path)),
        None => runmat_package::discover_source_symbols_from_source_name_async(
            &path.to_string_lossy(),
            directory,
        )
        .await
        .map_err(display)?,
    };
    for search in &request.search_paths {
        // Roots were resolved against the original captured CWD before await,
        // matching CLI search-path precedence without a later ambient lookup.
        let root = search.clone();
        let index = runmat_config::project::build_loose_source_index_async(&root)
            .await
            .map_err(display)?;
        merge(
            &mut catalog,
            runmat_package::source_symbols_from_index(&index, &root, path, None),
        );
    }
    let mut project = analyze_project_source(
        ProjectSource {
            path: path.to_path_buf(),
            text: content,
        },
        provider.path_namespace(),
        compat,
        catalog.as_ref(),
        |definition| {
            let provider = provider.clone();
            async move { read(provider.as_ref(), &definition.source_path).await }
        },
    )
    .await
    .map_err(display)?;
    if let (Some(function), Some(lowering)) = (&function, &project.analysis.lowering) {
        let assembly = &lowering.assembly;
        let matches = assembly.primary_source.map_or(0, |primary| {
            assembly
                .modules
                .iter()
                .filter(|module| module.source_id == primary)
                .flat_map(|module| {
                    module.top_level_functions.iter().filter_map(move |id| {
                        assembly.functions.iter().find(|function| {
                            function.id == *id
                                && function.module == module.id
                                && function.parent.is_none()
                                && function.enclosing_class.is_none()
                                && matches!(function.kind, runmat_hir::FunctionKind::Named)
                        })
                    })
                })
                .filter(|selected| selected.name.0 == *function)
                .count()
        });
        if matches != 1 {
            project.analysis.extend_diagnostics(
                [HirDiagnostic::new(
                    "RunMat:Check:FunctionNotFound",
                    HirDiagnosticSeverity::Error,
                    format!("primary source does not uniquely declare function `{function}`"),
                    Span { start: 0, end: 0 },
                )
                .with_category("semantic")],
                runmat_static_analysis::frontend::DiagnosticStage::Semantic,
            );
        }
    }
    let errors = project
        .analysis
        .diagnostics
        .iter()
        .filter(|d| d.severity == HirDiagnosticSeverity::Error)
        .count();
    let warnings = project.analysis.warning_count();
    let diagnostics = project
        .analysis
        .diagnostics
        .iter()
        .enumerate()
        .map(|(index, d)| {
            project_diagnostic(d, project.analysis.diagnostic_stage(index), |id| {
                project
                    .source(id)
                    .map(|s| (s.path.as_path(), s.text.as_str()))
            })
        })
        .collect();
    Ok(CheckedDocument {
        path: path.to_path_buf(),
        function,
        outcome: outcome(errors, warnings, request.deny_warnings),
        diagnostics,
        sources: project
            .sources
            .iter()
            .enumerate()
            .map(|(source_id, s)| CheckedSource {
                source_id,
                path: s.path.clone(),
            })
            .collect(),
        analysis: project.analysis.domains,
        resolution: project.analysis.resolution,
        analysis_incomplete_sources: project.incomplete_sources,
        call_context_budget_exhausted: project.call_context_budget_exhausted,
        call_context_precision_withheld: project.call_context_precision_withheld,
        summary: CheckSummary {
            errors,
            warnings,
            warnings_denied: request.deny_warnings,
        },
    })
}

async fn read(provider: &dyn runmat_filesystem::FsProvider, path: &Path) -> Result<String, String> {
    String::from_utf8(provider.read(path).await.map_err(display)?).map_err(display)
}
fn merge(destination: &mut Option<DiscoveredSourceSymbols>, additional: DiscoveredSourceSymbols) {
    if let Some(destination) = destination {
        destination.symbols.extend(additional.symbols);
        for definition in additional.definitions {
            if !destination.definitions.contains(&definition) {
                destination.definitions.push(definition);
            }
        }
    } else {
        *destination = Some(additional);
    }
}
fn display(error: impl std::fmt::Display) -> String {
    error.to_string()
}
