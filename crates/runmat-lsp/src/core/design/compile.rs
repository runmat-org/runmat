//! The dialect compilers behind a design document's diagnostics: the part regenerates on the
//! design kernel (OCCT in the native server; the grid test double in the browser build),
//! assemblies resolve through the design tools' check, drawings and circuits
//! report their front ends and ERC.

use super::*;

/// Runs the dialect's compiler: its problems in this file and its projection.
pub(super) fn compile(
    dialect: Dialect,
    path: &str,
    text: &str,
    files: &DesignFiles,
    table: &SymbolTable,
) -> (Vec<Problem>, Option<Value>) {
    match dialect {
        Dialect::Part => compile_part(path, text, files, table),
        Dialect::Assembly => {
            let (mut problems, projection) = cad_front_end(
                runmat_cad::syntax::parse_assembly_in(path, text, &module_scope(path, text, files)),
                path,
                files,
                text,
            );
            problems.extend(assembly_problems(path, text, files));
            (problems, projection)
        }
        Dialect::Drawing => cad_front_end(
            runmat_cad::syntax::parse_drawing_in(path, text, &module_scope(path, text, files)),
            path,
            files,
            text,
        ),
        Dialect::Circuit | Dialect::CircuitLayout => {
            let out = runmat_circuit::runtime::check(path, text, files, &Default::default());
            let problems = out
                .diagnostics
                .iter()
                .map(|d| {
                    let in_file = d.path == path;
                    Problem {
                        span: if in_file {
                            Span {
                                start: byte_offset(text, d.start),
                                end: byte_offset(text, d.end),
                            }
                        } else {
                            Span::default()
                        },
                        severity: severity(&d.severity),
                        code: None,
                        message: if in_file {
                            d.message.clone()
                        } else {
                            format!("{}: {}", d.path, d.message)
                        },
                    }
                })
                .collect();
            (
                problems,
                out.index.and_then(|index| serde_json::to_value(index).ok()),
            )
        }
        Dialect::Fea => (Vec::new(), None),
    }
}

/// The folder scope of `files` with this document among them.
pub(super) fn module_scope(
    path: &str,
    text: &str,
    files: &DesignFiles,
) -> runmat_cad::modules::ModuleScope {
    runmat_cad::modules::ModuleScope::from_files(
        files
            .iter()
            .filter(|(p, _)| p.as_str() != path)
            .map(|(p, t)| (p.as_str(), t.as_str()))
            .chain(std::iter::once((path, text))),
    )
}

pub(super) fn severity(text: &str) -> DiagnosticSeverity {
    match text {
        "warning" => DiagnosticSeverity::WARNING,
        "info" => DiagnosticSeverity::INFORMATION,
        _ => DiagnosticSeverity::ERROR,
    }
}

/// The byte offset of a UTF-16 offset.
pub(super) fn byte_offset(text: &str, utf16: usize) -> usize {
    if text.is_ascii() {
        return utf16.min(text.len());
    }
    let mut units = 0;
    for (byte, ch) in text.char_indices() {
        if units >= utf16 {
            return byte;
        }
        units += ch.len_utf16();
    }
    text.len()
}

/// A part regenerated on the design kernel (OCCT in the native server, as Desktop regenerates);
/// regeneration diagnostics at their sites.
/// A part regenerated for the editor. Values that call `.m` functions resolve through the folder
/// scope but don't run here (the editor has no runtime): their calls stay pending, and neither
/// they nor what depends on them is reported as a problem. The Desktop's compile runs them.
fn compile_part(path: &str, text: &str, files: &DesignFiles, table: &SymbolTable) -> (Vec<Problem>, Option<Value>) {
    use runmat_cad::regen::{regenerate, RegenCache, RegenContext, RegenRequest};
    use runmat_cad::value::m_functions::MFunctions;
    use runmat_cad::value::m_table::{TableRunner, PENDING};
    let parsed = runmat_cad::syntax::parse_part(path, text);
    let mut problems = model_problems(&parsed.diagnostics);
    let Some(document) = parsed.document else {
        return (problems, None);
    };
    let kernel = runmat_design_tools::kernel::make();
    let scope = module_scope(path, text, files);
    let mut all = files.clone();
    all.insert(path.to_owned(), text.to_owned());
    let (runner, pending) = TableRunner::new(Default::default(), "editor");
    let values = MFunctions::new(&scope, &all, path, &runner);
    let ctx = RegenContext {
        kernel: kernel.as_ref(),
        solver: &runmat_cad::sketch_solver::ConstraintSketchSolver,
        units: &runmat_cad::units::StubUnits,
        dynamic: &values,
        lock: None,
        catalog: None,
        policy: Default::default(),
        slots: None,
    };
    let result = regenerate(
        &document,
        &ctx,
        &RegenRequest::default(),
        &mut RegenCache::default(),
    );
    let mut report = result.diagnostics.clone();
    report.attach_spans(&parsed.spans);
    let waiting = !pending.borrow().is_empty();
    for d in &report.diagnostics {
        let cascade = matches!(d.cause, runmat_cad::diagnostics::Cause::Cascade { .. });
        if d.message.contains(PENDING) || (waiting && cascade) {
            continue;
        }
        let span = d
            .span
            .as_ref()
            .map(|s| Span {
                start: s.start as usize,
                end: s.end as usize,
            })
            .or_else(|| site_span(&d.site, table))
            .unwrap_or_default();
        problems.push(Problem {
            span,
            severity: match d.severity {
                runmat_cad::diagnostics::Severity::Error if cascade => DiagnosticSeverity::WARNING,
                runmat_cad::diagnostics::Severity::Error => DiagnosticSeverity::ERROR,
                runmat_cad::diagnostics::Severity::Warning => DiagnosticSeverity::WARNING,
                _ => DiagnosticSeverity::INFORMATION,
            },
            code: serde_json::to_value(d.code)
                .ok()
                .and_then(|v| v.as_str().map(str::to_string)),
            message: d.message.clone(),
        });
    }
    let mut index = runmat_cad::index::build_index(&document, &result, None);
    for row in &mut index.lets {
        if let Some(function) = &row.function {
            row.file = values.resolve(function).ok().map(|f| f.file);
        }
    }
    let projection = serde_json::json!({ "document": &document, "index": &index });
    (problems, Some(projection))
}

/// The declaration a diagnostic's site names, when the span map has no entry for it.
fn site_span(site: &runmat_cad::diagnostics::Site, table: &SymbolTable) -> Option<Span> {
    let key = site.key();
    let base = key.split('#').next().unwrap_or(&key);
    let (kind, name) = base.split_once('.')?;
    table
        .symbols
        .iter()
        .find(|s| s.kind == kind && s.name == name)
        .map(|s| s.name_span)
}

fn model_problems(diagnostics: &[runmat_design_syntax::FeaSyntaxDiagnostic]) -> Vec<Problem> {
    diagnostics
        .iter()
        .map(|d| Problem {
            span: d.span,
            severity: DiagnosticSeverity::ERROR,
            code: Some("design.model".into()),
            message: d.message.clone(),
        })
        .collect()
}

/// An assembly's or drawing's front end: model errors, and each design file it names that
/// isn't open or in the workspace.
fn cad_front_end<T>(
    parsed: runmat_cad::syntax::Parsed<T>,
    path: &str,
    files: &DesignFiles,
    text: &str,
) -> (Vec<Problem>, Option<Value>) {
    let mut problems = model_problems(&parsed.diagnostics);
    let syntax = parse_source(Dialect::for_path(path).unwrap_or(Dialect::Assembly), text);
    let source = syntax.source();
    for node in syntax.syntax().descendants() {
        if node.kind() != runmat_design_syntax::tree::NodeKind::StringLiteral {
            continue;
        }
        let span = node.trimmed_span();
        let Some(inner) = source
            .get(span.start..span.end)
            .and_then(|t| t.strip_prefix('"'))
            .and_then(|t| t.strip_suffix('"'))
        else {
            continue;
        };
        if !(inner.ends_with(".part") || inner.ends_with(".assembly")) {
            continue;
        }
        let resolved = runmat_circuit::runtime::resolve_path(path, inner);
        if !files.contains_key(&resolved)
            && !files
                .keys()
                .any(|k| k.ends_with(&format!("/{}", inner.trim_start_matches("./"))))
        {
            problems.push(Problem {
                span,
                severity: DiagnosticSeverity::WARNING,
                code: Some("cad.file.missing".into()),
                message: format!("{inner} isn't open or in the workspace, so it draws as its box"),
            });
        }
    }
    (problems, None)
}

/// An assembly's resolve diagnostics (mates, patterns, occurrences, configurations), from the
/// design tools' check over the parts it uses. Front-end and missing-file problems are left to
/// [`cad_front_end`], which places them on the `use` line.
fn assembly_problems(path: &str, text: &str, files: &DesignFiles) -> Vec<Problem> {
    let mut all = files.clone();
    all.insert(path.to_owned(), text.to_owned());
    let Ok(out) = runmat_design_tools::check::check(
        &runmat_design_tools::check::CheckInput {
            path: path.to_owned(),
        },
        &all,
    ) else {
        return Vec::new();
    };
    out.diagnostics
        .into_iter()
        .filter(|d| d.code.is_some())
        .map(|d| Problem {
            span: d
                .range
                .as_ref()
                .filter(|r| r.path == path)
                .map(|r| Span {
                    start: byte_offset(text, r.start),
                    end: byte_offset(text, r.end),
                })
                .unwrap_or_default(),
            severity: severity(&d.severity),
            code: d.code,
            message: d.message,
        })
        .collect()
}
