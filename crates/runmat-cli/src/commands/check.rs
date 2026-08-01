use anyhow::{Context, Result};
use runmat_config::runtime::RunMatRuntimeConfig;
use runmat_hir::{HirDiagnostic, HirDiagnosticSeverity, LoweringContext, Span};
use runmat_runtime::analysis::{
    analysis_plan_study_op, analysis_plan_study_sweep_op, analysis_validate_study_op,
    analysis_validate_study_sweep_op, is_fea_file_path, load_fea_document_from_path_async,
    AnalysisStudyPlanData, AnalysisStudySweepPlanData, AnalysisStudySweepValidateData,
    AnalysisStudyValidateResult, FeaResolvedDocument,
};
use runmat_runtime::operations::{OperationContext, OperationErrorEnvelope};
use serde::Serialize;
use serde_json::json;
use std::path::{Path, PathBuf};

use crate::cli::Cli;
use crate::commands::script::resolve_script_input;
use crate::diagnostics::parser_compat;
use crate::presentation::{self, StreamStyles, Tone};
use crate::AlreadyReportedCliError;

pub async fn execute_check(
    file: PathBuf,
    cli: &Cli,
    config: &RunMatRuntimeConfig,
    json: bool,
    verbose: bool,
    deny: Vec<String>,
    search_paths: Vec<PathBuf>,
) -> Result<()> {
    let deny_warnings = parse_denied_levels(&deny)?;
    let file = resolve_script_input(file)?;
    if is_fea_file_path(&file) {
        return check_fea_file(file, config, json).await;
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

async fn check_fea_file(path: PathBuf, config: &RunMatRuntimeConfig, json: bool) -> Result<()> {
    if !json {
        let styles = presentation::stderr();
        eprintln!(
            "{} {}",
            styles.heading("Checking"),
            styles.path(path.display())
        );
        eprintln!("  {}", styles.muted("loading study document"));
    }
    let document = load_fea_document_from_path_async(&path)
        .await
        .map_err(|err| anyhow::anyhow!("Failed to load FEA file {}: {err}", path.display()))?;
    let context = OperationContext::new(None, None);
    match document {
        FeaResolvedDocument::Study(spec) => {
            if !json {
                let styles = presentation::stderr();
                eprintln!(
                    "  {} {} ({:?}, {:?})",
                    styles.label("study:"),
                    styles.identifier(&spec.study_id),
                    spec.run_kind,
                    spec.backend
                );
                eprintln!(
                    "  {} {} regions, {} meshes",
                    styles.label("geometry:"),
                    spec.geometry.regions.len(),
                    spec.geometry.meshes.len()
                );
                eprintln!("  {}", styles.info("validating"));
            }
            let validation = analysis_validate_study_op(&spec, context.clone())
                .map_err(report_operation_error)?;
            if !validation.data.valid {
                if json {
                    print_payload("study", "invalid", &validation.data, config.runtime.verbose)?;
                } else {
                    print_study_validation_failure(&validation.data);
                }
                return Err(AlreadyReportedCliError.into());
            }
            if !json {
                eprintln!("  {}", presentation::stderr().info("planning"));
            }
            let plan = analysis_plan_study_op(&spec, context).map_err(report_operation_error)?;
            if json {
                print_payload(
                    "study",
                    "valid",
                    &json!({
                        "validation": validation.data,
                        "plan": plan.data,
                    }),
                    config.runtime.verbose,
                )?;
            } else {
                print_study_check_summary(&validation.data, &plan.data, config.runtime.verbose);
            }
        }
        FeaResolvedDocument::Sweep(spec) => {
            if !json {
                let styles = presentation::stderr();
                eprintln!(
                    "  {} {} ({} studies, fail_fast: {})",
                    styles.label("sweep:"),
                    styles.identifier(&spec.sweep_id),
                    spec.studies.len(),
                    spec.fail_fast
                );
                eprintln!("  {}", styles.info("validating"));
            }
            let validation = analysis_validate_study_sweep_op(&spec, context.clone())
                .map_err(report_operation_error)?;
            if !validation.data.valid {
                if json {
                    print_payload("sweep", "invalid", &validation.data, config.runtime.verbose)?;
                } else {
                    print_sweep_validation_failure(&validation.data);
                }
                return Err(AlreadyReportedCliError.into());
            }
            if !json {
                eprintln!("  {}", presentation::stderr().info("planning"));
            }
            let plan =
                analysis_plan_study_sweep_op(&spec, context).map_err(report_operation_error)?;
            if json {
                print_payload(
                    "sweep",
                    "valid",
                    &json!({
                        "validation": validation.data,
                        "plan": plan.data,
                    }),
                    config.runtime.verbose,
                )?;
            } else {
                print_sweep_check_summary(&validation.data, &plan.data);
            }
        }
    }
    Ok(())
}

async fn check_m_file(
    path: PathBuf,
    cli: &Cli,
    config: &RunMatRuntimeConfig,
    options: CheckOutputOptions,
) -> Result<()> {
    let content = runmat_filesystem::read_to_string_async(&path)
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
        let root = if search_path.is_absolute() {
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

    let known_symbols = source_catalog
        .as_ref()
        .map(|catalog| &catalog.symbols)
        .cloned()
        .unwrap_or_default();
    let lowering_context = LoweringContext::empty()
        .with_runmat_extensions_enabled(
            parser_compat(config.language.compat).allows_runmat_extensions(),
        )
        .with_known_project_symbols(&known_symbols);
    let mut analysis = runmat_static_analysis::frontend::analyze_source_with_catalog(
        &content,
        parser_compat(config.language.compat),
        &lowering_context,
        source_catalog.as_ref(),
    );
    analysis.diagnostics.extend(catalog_diagnostics);
    analysis.diagnostics.sort_by_key(|diagnostic| {
        (
            diagnostic.primary.span.start,
            diagnostic.primary.span.end,
            diagnostic.code.clone(),
        )
    });

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
            .map(|diagnostic| CheckDiagnostic::from_hir(diagnostic, &content))
            .collect::<Vec<_>>();
        let payload = json!({
            "schema_version": 1,
            "document_kind": "script",
            "outcome": outcome,
            "path": path,
            "project_revision": analysis.project_revision,
            "analysis": analysis.domains,
            "resolution": analysis.resolution,
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
                render_check_diagnostic_with_styles(diagnostic, &path, &content, &styles)
            );
        }
        if options.verbose {
            println!("analysis: {}", serde_json::to_string(&analysis.domains)?);
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

#[derive(Serialize)]
struct CheckDiagnostic<'a> {
    code: &'a str,
    severity: &'static str,
    message: &'a str,
    category: Option<&'a str>,
    primary: CheckDiagnosticSpan<'a>,
    secondary: Vec<CheckDiagnosticSpan<'a>>,
    notes: Vec<&'a str>,
    help: Option<&'a str>,
    suggestions: Vec<CheckSuggestion<'a>>,
}

#[derive(Serialize)]
struct CheckDiagnosticSpan<'a> {
    start: usize,
    end: usize,
    line: usize,
    column: usize,
    end_line: usize,
    end_column: usize,
    label: Option<&'a str>,
}

#[derive(Serialize)]
struct CheckSuggestion<'a> {
    span: CheckDiagnosticSpan<'a>,
    replacement: &'a str,
    message: &'a str,
}

impl<'a> CheckDiagnostic<'a> {
    fn from_hir(diagnostic: &'a HirDiagnostic, source: &str) -> Self {
        Self {
            code: &diagnostic.code,
            severity: diagnostic_severity_name(&diagnostic.severity),
            message: &diagnostic.message,
            category: diagnostic.category.as_deref(),
            primary: CheckDiagnosticSpan::from_hir(
                diagnostic.primary.span,
                diagnostic.primary.label.as_deref(),
                source,
            ),
            secondary: diagnostic
                .secondary
                .iter()
                .map(|span| CheckDiagnosticSpan::from_hir(span.span, span.label.as_deref(), source))
                .collect(),
            notes: diagnostic
                .notes
                .iter()
                .map(|note| note.message.as_str())
                .collect(),
            help: diagnostic.help.as_deref(),
            suggestions: diagnostic
                .suggestions
                .iter()
                .map(|suggestion| CheckSuggestion {
                    span: CheckDiagnosticSpan::from_hir(suggestion.span, None, source),
                    replacement: &suggestion.replacement,
                    message: &suggestion.message,
                })
                .collect(),
        }
    }
}

impl<'a> CheckDiagnosticSpan<'a> {
    fn from_hir(span: Span, label: Option<&'a str>, source: &str) -> Self {
        let (line, column) = source_position(source, span.start);
        let (end_line, end_column) = source_position(source, span.end);
        Self {
            start: span.start,
            end: span.end,
            line,
            column,
            end_line,
            end_column,
            label,
        }
    }
}

fn diagnostic_severity_name(severity: &HirDiagnosticSeverity) -> &'static str {
    match severity {
        HirDiagnosticSeverity::Error => "error",
        HirDiagnosticSeverity::Warning => "warning",
        HirDiagnosticSeverity::Information => "information",
        HirDiagnosticSeverity::Help => "help",
    }
}

fn source_position(source: &str, offset: usize) -> (usize, usize) {
    let offset = offset.min(source.len());
    let line_start = source[..offset].rfind('\n').map_or(0, |index| index + 1);
    let line = source[..line_start]
        .bytes()
        .filter(|byte| *byte == b'\n')
        .count()
        + 1;
    let column = source[line_start..offset].chars().count() + 1;
    (line, column)
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

fn render_check_diagnostic_with_styles(
    diagnostic: &HirDiagnostic,
    path: &Path,
    source: &str,
    styles: &StreamStyles,
) -> String {
    let (severity, severity_tone) = match diagnostic.severity {
        HirDiagnosticSeverity::Error => ("error", Tone::Error),
        HirDiagnosticSeverity::Warning => ("warning", Tone::Warning),
        HirDiagnosticSeverity::Information => ("info", Tone::Info),
        HirDiagnosticSeverity::Help => ("help", Tone::Help),
    };
    let (line, column, line_start, line_text) = source_line(source, diagnostic.primary.span.start);
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
    let mut rendered = format!(
        "{severity}[{code}]: {}\n {arrow} {location}\n{:gutter$} {separator}\n{line:>gutter$} {separator} {line_text}\n{:gutter$} {separator} {}{carets}",
        diagnostic.message,
        "",
        "",
        " ".repeat(prefix_width),
    );
    if let Some(label) = &diagnostic.primary.label {
        rendered.push(' ');
        rendered.push_str(&styles.paint(severity_tone, label));
    }
    for secondary in &diagnostic.secondary {
        let (secondary_line, secondary_column, _, _) = source_line(source, secondary.span.start);
        let related = styles.info("related:");
        let location = styles.path(format!(
            "{}:{secondary_line}:{secondary_column}",
            path.display()
        ));
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

fn print_study_check_summary(
    validation: &AnalysisStudyValidateResult,
    plan: &AnalysisStudyPlanData,
    verbose: bool,
) {
    let styles = presentation::stdout();
    println!(
        "{} {}",
        styles.success("OK"),
        styles.identifier(&plan.study_id)
    );
    println!("  {} {:?}", styles.label("kind:"), plan.run_kind);
    println!("  {} {:?}", styles.label("backend:"), plan.backend);
    println!(
        "  {} {}",
        styles.label("model:"),
        styles.identifier(&plan.model_id)
    );
    println!(
        "  {} {} ({} issues)",
        styles.label("validation:"),
        styles.success("passed"),
        validation.issues.len()
    );
    println!(
        "  {} {} ({})",
        styles.label("run op:"),
        styles.identifier(&plan.run_operation),
        plan.run_op_version
    );
    println!(
        "  {} {}",
        styles.label("evidence:"),
        styles.path(&plan.evidence_artifact_path)
    );
    if verbose {
        println!("  study fingerprint: {}", plan.study_fingerprint);
        println!("  operations:");
        for operation in &plan.operation_sequence {
            println!("    {operation}");
        }
    }
}

fn print_study_validation_failure(validation: &AnalysisStudyValidateResult) {
    let styles = presentation::stdout();
    println!("{} validation", styles.error("FAILED"));
    println!(
        "  {} {}",
        styles.label("evidence:"),
        styles.path(&validation.evidence_artifact_path)
    );
    for issue in &validation.issues {
        println!("  {}: {}", styles.identifier(&issue.code), issue.message);
    }
}

fn print_sweep_check_summary(
    validation: &AnalysisStudySweepValidateData,
    plan: &AnalysisStudySweepPlanData,
) {
    let styles = presentation::stdout();
    println!(
        "{} {}",
        styles.success("OK"),
        styles.identifier(&plan.sweep_id)
    );
    println!("  studies: {}", plan.study_count);
    println!("  planned: {}", plan.planned_count);
    println!("  failed: {}", plan.failed_count);
    println!(
        "  validation: passed ({} study entries)",
        validation.study_entries.len()
    );
    println!("  evidence: {}", plan.evidence_artifact_path);
    for entry in &plan.plan_entries {
        println!(
            "  {}: {} ({})",
            entry.study_id, entry.run_operation, entry.run_op_version
        );
    }
}

fn print_sweep_validation_failure(validation: &AnalysisStudySweepValidateData) {
    let styles = presentation::stdout();
    println!(
        "{} {}",
        styles.error("FAILED"),
        styles.identifier(&validation.sweep_id)
    );
    println!("  evidence: {}", validation.evidence_artifact_path);
    for entry in &validation.study_entries {
        if entry.valid {
            continue;
        }
        println!("  {}", entry.study_id);
        for issue in &entry.issues {
            println!("    {}: {}", issue.code, issue.message);
        }
    }
}

fn print_payload<T: Serialize>(
    document_kind: &'static str,
    status: &'static str,
    data: &T,
    include_metadata: bool,
) -> Result<()> {
    if include_metadata {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "document_kind": document_kind,
                "status": status,
                "data": data,
            }))
            .context("Failed to serialize check result")?
        );
    } else {
        println!(
            "{}",
            serde_json::to_string_pretty(data).context("Failed to serialize check result")?
        );
    }
    Ok(())
}

fn is_matlab_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("m"))
}

fn report_operation_error(error: OperationErrorEnvelope) -> anyhow::Error {
    match serde_json::to_string_pretty(&error) {
        Ok(payload) => eprintln!("{payload}"),
        Err(_) => eprintln!("{}: {}", error.error_code, error.message.replace('\n', " ")),
    }
    AlreadyReportedCliError.into()
}

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
