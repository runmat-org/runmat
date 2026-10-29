use anyhow::{Context, Result};
use runmat_config::runtime::RunMatRuntimeConfig;
use runmat_hir::{HirDiagnostic, HirDiagnosticSeverity, Span};
use runmat_static_analysis::project::check_report::{project_diagnostic, source_position};
use runmat_static_analysis::project::{ProjectAnalysis, ProjectSource};
use runmat_types::ProgramSourceId;
use serde_json::json;
use std::path::{Path, PathBuf};

use crate::cli::Cli;
use crate::commands::script::resolve_script_input;
use crate::diagnostics::parser_compat;
use crate::presentation::{self, StreamStyles, Tone};
use crate::AlreadyReportedCliError;

/// Options for `runmat check`, shared by `.m` and `.fea` sources.
pub struct CheckRequest {
    pub root: crate::cli::FeaRootArgs,
    pub json: bool,
    pub verbose: bool,
    pub deny: Vec<String>,
    pub search_paths: Vec<PathBuf>,
}

pub async fn execute_check(
    file: PathBuf,
    cli: &Cli,
    config: &RunMatRuntimeConfig,
    request: CheckRequest,
) -> Result<()> {
    let CheckRequest {
        root,
        json,
        verbose,
        deny,
        search_paths,
    } = request;
    let deny_warnings = parse_denied_levels(&deny)?;
    let file = resolve_script_input(file)?;
    if crate::commands::fea::source::is_fea_source_path(&file) {
        if !search_paths.is_empty() {
            anyhow::bail!("--path adds MATLAB lookup roots; .fea documents import their functions explicitly");
        }
        return check_fea_file(file, cli, config, root, json, deny_warnings).await;
    }
    if !root.is_empty() {
        anyhow::bail!("--study and --sweep select a root of a .fea document; {} is not one", file.display());
    }
    if is_matlab_file(&file) {
        return check_m_file(
            file,
            cli,
            config,
            CheckOutputOptions {
                json,
                verbose,
                deny_warnings,
                search_paths,
            },
        )
        .await;
    }

    anyhow::bail!(
        "runmat check supports .m scripts and .fea documents; got {}",
        file.display()
    )
}

#[derive(Debug)]
struct CheckOutputOptions {
    json: bool,
    verbose: bool,
    deny_warnings: bool,
    search_paths: Vec<PathBuf>,
}

fn parse_denied_levels(levels: &[String]) -> Result<bool> {
    let mut deny_warnings = false;
    for level in levels {
        match level.trim().to_ascii_lowercase().as_str() {
            "warning" | "warnings" => deny_warnings = true,
            other => {
                anyhow::bail!("unsupported diagnostic level `{other}` for -D; expected `warnings`")
            }
        }
    }
    Ok(deny_warnings)
}

/// Checks a `.fea` document with every rule its run applies before meshing (the runtime's
/// checked source carries the static run admission), and resolves `--study`/`--sweep` exactly
/// as `runmat run` does.
async fn check_fea_file(
    path: PathBuf,
    cli: &Cli,
    config: &RunMatRuntimeConfig,
    root: crate::cli::FeaRootArgs,
    json: bool,
    deny_warnings: bool,
) -> Result<()> {
    let checked = crate::commands::fea::source::check_source(&path, cli, config).await?;
    let report = checked.check_result();
    let roots = checked.root_selectors();
    let selection = if report.status.is_valid() {
        crate::commands::fea::command::root_selector(&root)
            .map(|selector| checked.resolve_root(Some(&selector)).map(|_| selector))
            .transpose()
    } else {
        Ok(None)
    };
    let selection_error = selection.as_ref().err().map(ToString::to_string);
    let warnings = report
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.severity == runmat_analysis_language::FeaDiagnosticSeverity::Warning);
    let valid = report.status.is_valid()
        && selection_error.is_none()
        && !(deny_warnings && warnings);
    if json {
        // The full checked-source report (program, source index, previews, diagnostics) plus
        // the root summary and selection; `status` reflects the selection and `-D warnings`.
        let mut payload = serde_json::to_value(&report)?;
        let object = payload
            .as_object_mut()
            .context("FEA check report serializes as an object")?;
        object.insert("kind".into(), json!("fea"));
        object.insert("status".into(), json!(if valid { "valid" } else { "invalid" }));
        object.insert("studies".into(), json!(checked.program.studies.len()));
        object.insert("sweeps".into(), json!(checked.program.sweeps.len()));
        object.insert("roots".into(), json!(roots));
        object.insert(
            "selected".into(),
            json!(selection.as_ref().ok().cloned().flatten()),
        );
        object.insert("selection_error".into(), json!(selection_error));
        println!("{}", serde_json::to_string(&payload)?);
        return if valid {
            Ok(())
        } else {
            Err(AlreadyReportedCliError.into())
        };
    }
    let styles = presentation::stdout();
    let sources = FeaDiagnosticSources::new(&path, &checked.root_module, &checked.source_root);
    let counts = print_fea_diagnostics(&report.diagnostics, &sources, &styles, false);
    if let Some(error) = &selection_error {
        println!("{}", styles.error(format!("error: {error}")));
    }
    if let Ok(Some(selected)) = &selection {
        println!(
            "  {} {} {}",
            styles.label("selected:"),
            selected.kind(),
            styles.identifier(selected.id())
        );
    } else if roots.len() > 1 {
        println!(
            "  {} {}",
            styles.label("runnable roots:"),
            roots
                .iter()
                .map(|root| format!("--{} {}", root.kind(), root.id()))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    println!("{}", counts.summary(&path, &styles));
    if !valid {
        return Err(AlreadyReportedCliError.into());
    }
    Ok(())
}

/// Error and warning totals for one check, printed as the `checked …` summary line.
pub(crate) struct DiagnosticCounts {
    pub errors: usize,
    pub warnings: usize,
}

impl DiagnosticCounts {
    pub(crate) fn summary(&self, path: &Path, styles: &StreamStyles) -> String {
        let summary = format!(
            "checked {}: {} error(s), {} warning(s)",
            path.display(),
            self.errors,
            self.warnings
        );
        if self.errors > 0 {
            styles.error(summary)
        } else if self.warnings > 0 {
            styles.warning(summary)
        } else {
            styles.success(summary)
        }
    }
}

/// Maps the logical module paths in FEA diagnostics to the files the user names on the command
/// line, so located FEA diagnostics read like `.m` diagnostics.
pub(crate) struct FeaDiagnosticSources<'a> {
    entry_path: &'a Path,
    root_module: &'a runmat_analysis_language::LogicalSourcePath,
    source_root: &'a Path,
}

impl<'a> FeaDiagnosticSources<'a> {
    pub(crate) fn new(
        entry_path: &'a Path,
        root_module: &'a runmat_analysis_language::LogicalSourcePath,
        source_root: &'a Path,
    ) -> Self {
        Self {
            entry_path,
            root_module,
            source_root,
        }
    }

    fn display_path(&self, module: &runmat_analysis_language::LogicalSourcePath) -> PathBuf {
        if module == self.root_module {
            return self.entry_path.to_path_buf();
        }
        let physical = self.source_root.join(module.as_str());
        runmat_filesystem::current_dir()
            .ok()
            .and_then(|cwd| physical.strip_prefix(cwd).ok().map(Path::to_path_buf))
            .unwrap_or(physical)
    }

    fn source(&self, module: &runmat_analysis_language::LogicalSourcePath) -> Option<String> {
        let physical = if module == self.root_module {
            self.entry_path.to_path_buf()
        } else {
            self.source_root.join(module.as_str())
        };
        std::fs::read_to_string(physical).ok()
    }
}

/// Prints FEA diagnostics in the same located form `runmat check` uses for `.m` files and returns
/// the error and warning totals.
pub(crate) fn print_fea_diagnostics(
    diagnostics: &[runmat_analysis_language::FeaDiagnostic],
    sources: &FeaDiagnosticSources<'_>,
    styles: &StreamStyles,
    to_stderr: bool,
) -> DiagnosticCounts {
    let mut counts = DiagnosticCounts {
        errors: 0,
        warnings: 0,
    };
    for diagnostic in diagnostics {
        match diagnostic.severity {
            runmat_analysis_language::FeaDiagnosticSeverity::Error => counts.errors += 1,
            runmat_analysis_language::FeaDiagnosticSeverity::Warning => counts.warnings += 1,
        }
        let rendered = render_fea_diagnostic(diagnostic, sources, styles);
        if to_stderr {
            eprintln!("{rendered}");
        } else {
            println!("{rendered}");
        }
    }
    counts
}

fn render_fea_diagnostic(
    diagnostic: &runmat_analysis_language::FeaDiagnostic,
    sources: &FeaDiagnosticSources<'_>,
    styles: &StreamStyles,
) -> String {
    let severity = match diagnostic.severity {
        runmat_analysis_language::FeaDiagnosticSeverity::Error => HirDiagnosticSeverity::Error,
        runmat_analysis_language::FeaDiagnosticSeverity::Warning => HirDiagnosticSeverity::Warning,
    };
    let located = diagnostic.location.as_ref().and_then(|location| {
        sources
            .source(&location.path)
            .map(|source| (sources.display_path(&location.path), source, location.span))
    });
    let mut rendered = match &located {
        Some((path, source, span)) => render_located(
            &LocatedDiagnostic {
                severity: &severity,
                code: &diagnostic.code,
                message: &diagnostic.message,
                span: *span,
                label: diagnostic.label.as_deref(),
            },
            path,
            source,
            styles,
        ),
        None => {
            let (severity, tone) = severity_presentation(&severity);
            format!(
                "{}[{}]: {}",
                styles.paint(tone, severity),
                styles.identifier(&diagnostic.code),
                diagnostic.message
            )
        }
    };
    for related in &diagnostic.related {
        let location = match sources.source(&related.location.path) {
            Some(source) => {
                let (line, column, _, _) = source_line(&source, related.location.span.start);
                format!(
                    "{}:{line}:{column}",
                    sources.display_path(&related.location.path).display()
                )
            }
            None => sources
                .display_path(&related.location.path)
                .display()
                .to_string(),
        };
        rendered.push_str(&format!(
            "\n  {} {} {}: {}",
            styles.muted("="),
            styles.info("related:"),
            styles.path(location),
            related.message
        ));
    }
    rendered
}

async fn check_m_file(
    path: PathBuf,
    cli: &Cli,
    config: &RunMatRuntimeConfig,
    options: CheckOutputOptions,
) -> Result<()> {
    let (source_provider, resolved_primary) = runmat_filesystem::capture_provider_path(&path)
        .context("Failed to capture script filesystem authority")?;
    let path_namespace = source_provider.path_namespace();
    let content = read_project_source(source_provider.as_ref(), &resolved_primary)
        .await
        .with_context(|| format!("Failed to read script file: {}", path.display()))?;
    let cwd = runmat_filesystem::current_dir().context("Failed to read current directory")?;
    let source_name = path.to_string_lossy();
    let mut source_catalog: Option<runmat_package::DiscoveredSourceSymbols> = None;
    let mut catalog_diagnostics = Vec::new();
    match crate::commands::package::resolve_for_source(&path, cli).await {
        Ok(Some(project)) => {
            source_catalog = Some(runmat_package::source_symbols_from_frozen(
                &project.resolved.frozen,
                &path,
            ));
        }
        Ok(None) => {
            match runmat_package::discover_source_symbols_from_source_name_async(&source_name, &cwd)
                .await
            {
                Ok(Some(discovered)) => source_catalog = Some(discovered),
                Ok(None) => {}
                Err(error) => {
                    catalog_diagnostics.push(source_catalog_diagnostic(error.to_string()))
                }
            }
        }
        Err(error) => catalog_diagnostics.push(source_catalog_diagnostic(error.to_string())),
    }

    for search_path in &options.search_paths {
        let root = if path_namespace.is_absolute(search_path) {
            search_path.clone()
        } else {
            cwd.join(search_path)
        };
        match runmat_config::project::build_loose_source_index_async(&root).await {
            Ok(index) => {
                let discovered =
                    runmat_package::source_symbols_from_index(&index, &root, &path, None);
                merge_source_catalog(&mut source_catalog, discovered);
            }
            Err(error) => catalog_diagnostics.push(source_catalog_diagnostic(format!(
                "failed to index explicit search path {}: {error}",
                root.display()
            ))),
        }
    }

    let mut project = runmat_static_analysis::project::analyze_project_source(
        ProjectSource {
            path: resolved_primary,
            text: content,
        },
        path_namespace,
        parser_compat(config.language.compat),
        source_catalog.as_ref(),
        |definition| {
            let provider = source_provider.clone();
            async move {
                read_project_source(provider.as_ref(), &definition.source_path)
                    .await
                    .map_err(|error| error.to_string())
            }
        },
    )
    .await
    .with_context(|| format!("Failed to analyze project source: {}", path.display()))?;
    project.analysis.extend_diagnostics(
        catalog_diagnostics,
        runmat_static_analysis::frontend::DiagnosticStage::Resolution,
    );
    project.analysis.sort_diagnostics(false);
    let analysis = &project.analysis;
    let sources = CheckSources::Project(&project, &path);

    let error_count = analysis
        .diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.severity == HirDiagnosticSeverity::Error)
        .count();
    let warning_count = analysis.warning_count();
    let failed = error_count > 0 || (options.deny_warnings && warning_count > 0);
    let outcome = if failed {
        "failed"
    } else if warning_count > 0 {
        "warnings"
    } else {
        "clean"
    };

    if options.json {
        let diagnostics = analysis
            .diagnostics
            .iter()
            .enumerate()
            .map(|(index, diagnostic)| {
                project_diagnostic(diagnostic, analysis.diagnostic_stage(index), |id| {
                    sources.source(id)
                })
            })
            .collect::<Vec<_>>();
        let payload = json!({
            "schema_version": 1,
            "document_kind": "script",
            "outcome": outcome,
            "path": path,
            "project_revision": analysis.project_revision,
            "analysis": analysis.domains,
            "analysis_incomplete_sources": project.incomplete_sources,
            "call_context_budget_exhausted": project.call_context_budget_exhausted,
            "call_context_precision_withheld": project.call_context_precision_withheld,
            "resolution": analysis.resolution,
            "sources": project_source_paths(&project, &path),
            "diagnostics": diagnostics,
            "summary": {
                "errors": error_count,
                "warnings": warning_count,
                "warnings_denied": options.deny_warnings,
            },
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&payload).context("Failed to serialize check result")?
        );
    } else {
        let styles = presentation::stdout();
        for diagnostic in &analysis.diagnostics {
            println!(
                "{}",
                render_check_diagnostic_with_sources(diagnostic, &sources, &styles)
            );
        }
        if options.verbose {
            println!("analysis: {}", serde_json::to_string(&analysis.domains)?);
            if project.call_context_budget_exhausted {
                println!("analysis partial: call-context analysis budget exhausted");
            }
            if project.call_context_precision_withheld {
                println!("analysis partial: unsupported call-context precision withheld");
            }
            for source in &project.incomplete_sources {
                println!(
                    "analysis partial: companion source not composed: {}",
                    source.display()
                );
            }
        }
        let summary = format!(
            "checked {}: {} error(s), {} warning(s)",
            path.display(),
            error_count,
            warning_count
        );
        if error_count > 0 {
            println!("{}", styles.error(summary));
        } else if warning_count > 0 {
            println!("{}", styles.warning(summary));
        } else {
            println!("{}", styles.success(summary));
        }
    }
    if failed {
        return Err(AlreadyReportedCliError.into());
    }
    Ok(())
}

// Reading remains adapter-owned. Both primary and selected sources use the
// original captured provider; analysis receives only its pure namespace grammar.
async fn read_project_source(
    provider: &dyn runmat_filesystem::FsProvider,
    path: &Path,
) -> std::io::Result<String> {
    let bytes = provider.read(path).await?;
    String::from_utf8(bytes)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error.utf8_error()))
}

fn merge_source_catalog(
    destination: &mut Option<runmat_package::DiscoveredSourceSymbols>,
    mut additional: runmat_package::DiscoveredSourceSymbols,
) {
    if let Some(destination) = destination {
        destination.symbols.extend(additional.symbols);
        for definition in additional.definitions.drain(..) {
            if !destination.definitions.contains(&definition) {
                destination.definitions.push(definition);
            }
        }
    } else {
        *destination = Some(additional);
    }
}

// Project source IDs are positions in the shared owner's source table. Publish
// only display paths, not source text, so resolution spans remain attributable
// even when no diagnostic happens to refer to the same document.
fn project_source_paths(
    project: &ProjectAnalysis,
    primary_display_path: &Path,
) -> Vec<serde_json::Value> {
    project
        .sources
        .iter()
        .enumerate()
        .map(|(source_id, source)| {
            json!({
                "source_id": source_id,
                "path": if source_id == 0 { primary_display_path } else { source.path.as_path() },
            })
        })
        .collect()
}

enum CheckSources<'a> {
    Project(&'a ProjectAnalysis, &'a Path),
    #[cfg(test)]
    Single(&'a Path, &'a str),
}

impl<'a> CheckSources<'a> {
    fn source(&self, id: Option<ProgramSourceId>) -> Option<(&'a Path, &'a str)> {
        match self {
            Self::Project(project, primary_display_path) => project.source(id).map(|source| {
                let path = if id.is_none_or(|id| id == ProgramSourceId(0)) {
                    *primary_display_path
                } else {
                    source.path.as_path()
                };
                (path, source.text.as_str())
            }),
            #[cfg(test)]
            Self::Single(path, text) if id.is_none_or(|id| id == ProgramSourceId(0)) => {
                Some((path, text))
            }
            #[cfg(test)]
            Self::Single(_, _) => None,
        }
    }
}

fn source_catalog_diagnostic(message: String) -> HirDiagnostic {
    HirDiagnostic::new(
        runmat_static_analysis::frontend::DIAGNOSTIC_SOURCE_CATALOG,
        HirDiagnosticSeverity::Error,
        message,
        Span { start: 0, end: 0 },
    )
    .with_primary_label("source lookup could not be constructed")
    .with_category("source-catalog")
}

#[cfg(test)]
fn render_check_diagnostic(diagnostic: &HirDiagnostic, path: &Path, source: &str) -> String {
    render_check_diagnostic_with_styles(diagnostic, path, source, &StreamStyles::plain())
}

#[cfg(test)]
fn render_check_diagnostic_with_styles(
    diagnostic: &HirDiagnostic,
    path: &Path,
    source: &str,
    styles: &StreamStyles,
) -> String {
    render_check_diagnostic_with_sources(diagnostic, &CheckSources::Single(path, source), styles)
}

fn render_check_diagnostic_with_sources(
    diagnostic: &HirDiagnostic,
    sources: &CheckSources<'_>,
    styles: &StreamStyles,
) -> String {
    let (severity, severity_tone) = match diagnostic.severity {
        HirDiagnosticSeverity::Error => ("error", Tone::Error),
        HirDiagnosticSeverity::Warning => ("warning", Tone::Warning),
        HirDiagnosticSeverity::Information => ("info", Tone::Info),
        HirDiagnosticSeverity::Help => ("help", Tone::Help),
    };
    let mut rendered = if let Some((path, source)) = sources.source(diagnostic.primary.source) {
        let (line, column, line_start, line_text) =
            source_line(source, diagnostic.primary.span.start);
        let primary_start = diagnostic.primary.span.start.min(source.len());
        let primary_end = diagnostic
            .primary
            .span
            .end
            .min(source.len())
            .max(primary_start);
        let prefix_width = source[line_start..primary_start].chars().count();
        let span_len = source[primary_start..primary_end]
            .split('\n')
            .next()
            .unwrap_or_default()
            .chars()
            .count()
            .max(1)
            .min(
                line_text
                    .chars()
                    .count()
                    .saturating_sub(prefix_width)
                    .max(1),
            );
        let gutter = line.to_string().len();
        let severity = styles.paint(severity_tone, severity);
        let code = styles.identifier(&diagnostic.code);
        let location = styles.path(format!("{}:{line}:{column}", path.display()));
        let arrow = styles.muted("-->");
        let separator = styles.muted("|");
        let carets = styles.paint(severity_tone, "^".repeat(span_len));
        format!(
        "{severity}[{code}]: {}\n {arrow} {location}\n{:gutter$} {separator}\n{line:>gutter$} {separator} {line_text}\n{:gutter$} {separator} {}{carets}",
        diagnostic.message,
        "",
        "",
        " ".repeat(prefix_width),
    )
    } else {
        format!(
            "{}[{}]: {}\n {} <unavailable source {}>:bytes {}..{}",
            styles.paint(severity_tone, severity),
            styles.identifier(&diagnostic.code),
            diagnostic.message,
            styles.muted("-->"),
            diagnostic.primary.source.unwrap_or(ProgramSourceId(0)).0,
            diagnostic.primary.span.start,
            diagnostic.primary.span.end,
        )
    };
    if let Some(label) = &diagnostic.primary.label {
        rendered.push(' ');
        rendered.push_str(&styles.paint(severity_tone, label));
    }
    for secondary in &diagnostic.secondary {
        let related = styles.info("related:");
        let location = match sources.source(secondary.source) {
            Some((path, source)) => {
                let (line, column) = source_position(source, secondary.span.start);
                styles.path(format!("{}:{line}:{column}", path.display()))
            }
            None => styles.path(format!(
                "<unavailable source {}>:bytes {}..{}",
                secondary.source.unwrap_or(ProgramSourceId(0)).0,
                secondary.span.start,
                secondary.span.end,
            )),
        };
        rendered.push_str(&format!("\n  {} {related} {location}", styles.muted("=")));
        if let Some(label) = &secondary.label {
            rendered.push_str(&format!(": {label}"));
        }
    }
    for note in &diagnostic.notes {
        rendered.push_str(&format!(
            "\n  {} {} {}",
            styles.muted("="),
            styles.muted("note:"),
            note.message
        ));
    }
    if let Some(help) = &diagnostic.help {
        rendered.push_str(&format!(
            "\n  {} {} {help}",
            styles.muted("="),
            styles.help("help:")
        ));
    }
    rendered
}

struct LocatedDiagnostic<'a> {
    severity: &'a HirDiagnosticSeverity,
    code: &'a str,
    message: &'a str,
    span: Span,
    label: Option<&'a str>,
}

fn severity_presentation(severity: &HirDiagnosticSeverity) -> (&'static str, Tone) {
    match severity {
        HirDiagnosticSeverity::Error => ("error", Tone::Error),
        HirDiagnosticSeverity::Warning => ("warning", Tone::Warning),
        HirDiagnosticSeverity::Information => ("info", Tone::Info),
        HirDiagnosticSeverity::Help => ("help", Tone::Help),
    }
}

/// Renders `severity[code]: message`, the `-->` location, and the source line with a caret
/// underline and label. The snippet drops the line's leading indentation; the reported column
/// stays the column in the file.
fn render_located(
    diagnostic: &LocatedDiagnostic<'_>,
    path: &Path,
    source: &str,
    styles: &StreamStyles,
) -> String {
    let (severity, severity_tone) = severity_presentation(diagnostic.severity);
    let (line, column, line_start, line_text) = source_line(source, diagnostic.span.start);
    let primary_start = diagnostic.span.start.min(source.len());
    let primary_end = diagnostic.span.end.min(source.len()).max(primary_start);
    let indentation = line_text.len() - line_text.trim_start().len();
    let snippet = &line_text[indentation..];
    let prefix_width = source[line_start..primary_start]
        .chars()
        .count()
        .saturating_sub(line_text[..indentation].chars().count());
    let span_len = source[primary_start..primary_end]
        .split('\n')
        .next()
        .unwrap_or_default()
        .chars()
        .count()
        .max(1)
        .min(snippet.chars().count().saturating_sub(prefix_width).max(1));
    let gutter = line.to_string().len();
    let severity = styles.paint(severity_tone, severity);
    let code = styles.identifier(diagnostic.code);
    let location = styles.path(format!("{}:{line}:{column}", path.display()));
    let arrow = styles.muted("-->");
    let separator = styles.muted("|");
    let carets = styles.paint(severity_tone, "^".repeat(span_len));
    let mut rendered = format!(
        "{severity}[{code}]: {}\n {arrow} {location}\n{:gutter$} {separator}\n{line:>gutter$} {separator} {snippet}\n{:gutter$} {separator} {}{carets}",
        diagnostic.message,
        "",
        "",
        " ".repeat(prefix_width),
    );
    if let Some(label) = diagnostic.label {
        rendered.push(' ');
        rendered.push_str(&styles.paint(severity_tone, label));
    }
    rendered
}

fn source_line(source: &str, offset: usize) -> (usize, usize, usize, &str) {
    let offset = offset.min(source.len());
    let line_start = source[..offset].rfind('\n').map_or(0, |index| index + 1);
    let line_end = source[offset..]
        .find('\n')
        .map_or(source.len(), |index| offset + index);
    let line = source[..line_start]
        .bytes()
        .filter(|byte| *byte == b'\n')
        .count()
        + 1;
    let column = source[line_start..offset].chars().count() + 1;
    (line, column, line_start, &source[line_start..line_end])
}

fn is_matlab_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("m"))
}

#[cfg(test)]
#[path = "check/project_diagnostic_tests.rs"]
mod project_diagnostic_tests;

#[cfg(test)]
mod script_check_tests {
    use super::*;

    #[test]
    fn denied_warning_level_accepts_singular_and_plural_only() {
        assert!(parse_denied_levels(&["warnings".to_string()]).unwrap());
        assert!(parse_denied_levels(&["warning".to_string()]).unwrap());
        assert!(!parse_denied_levels(&[]).unwrap());
        assert!(parse_denied_levels(&["unknown".to_string()]).is_err());
    }

    #[test]
    fn diagnostic_renderer_uses_source_coordinates_and_context() {
        let source = "x = 1;\ny = missing(x);\n";
        let start = source.find("missing").unwrap();
        let diagnostic = HirDiagnostic::new(
            runmat_static_analysis::frontend::DIAGNOSTIC_UNRESOLVED_FUNCTION,
            HirDiagnosticSeverity::Warning,
            "cannot find function `missing`",
            Span {
                start,
                end: start + "missing".len(),
            },
        )
        .with_primary_label("not defined in this file or project")
        .with_help("define `missing` in the project source roots");

        let rendered = render_check_diagnostic(&diagnostic, Path::new("project/main.m"), source);
        assert!(rendered.starts_with("warning[RM-RES0001]: cannot find function `missing`"));
        assert!(rendered.contains("--> project/main.m:2:5"));
        assert!(rendered.contains("2 | y = missing(x);"));
        assert!(rendered.contains("^^^^^^^ not defined in this file or project"));
        assert!(rendered.contains("= help: define `missing` in the project source roots"));

        let styled = render_check_diagnostic_with_styles(
            &diagnostic,
            Path::new("project/main.m"),
            source,
            &StreamStyles::new(crate::presentation::ColorLevel::Basic),
        );
        assert!(styled.contains("\u{1b}["));
        assert_eq!(strip_ansi(&styled), rendered);
    }

    #[test]
    fn source_coordinates_count_unicode_columns() {
        let source = "% π\nvalue = 1;";
        let offset = source.find("value").unwrap();
        let (line, column, line_start, text) = source_line(source, offset);
        assert_eq!((line, column, line_start, text), (2, 1, 5, "value = 1;"));
    }

    fn strip_ansi(value: &str) -> String {
        let mut output = String::new();
        let mut chars = value.chars().peekable();
        while let Some(character) = chars.next() {
            if character == '\u{1b}' && chars.peek() == Some(&'[') {
                chars.next();
                for next in chars.by_ref() {
                    if next.is_ascii_alphabetic() {
                        break;
                    }
                }
            } else {
                output.push(character);
            }
        }
        output
    }
}
