use std::collections::HashSet;

use runmat_hir::testing::{SemanticDiscoveryInput, SemanticTestSource};
use runmat_hir::{HirAssembly, LoweringContext};
use runmat_test::descriptor::{SourceDescriptor, SourceSpan};
use runmat_test::discovery::{
    DiscoveryDiagnostic, DiscoveryDiagnosticSeverity, FrozenTestRunSnapshot, TestDiscovery,
};

use runmat_parser::CompatMode;

/// Discover tests from a caller-frozen source set. This is the shared entry
/// point for native Core/tooling, browser runtime workers, and the WASM LSP:
/// it validates the immutable revision and never resolves or reads sources.
pub fn discover_frozen_tests(
    snapshot: &FrozenTestRunSnapshot,
    compat: CompatMode,
) -> TestDiscovery {
    if let Err(error) = snapshot.validate() {
        return failed_discovery(
            snapshot,
            "RunMat:TestDiscovery:InvalidSnapshot",
            error.to_string(),
            None,
        );
    }

    let known_symbols = snapshot
        .sources
        .iter()
        .filter_map(|source| source.relative_path.rsplit('/').next())
        .filter_map(|file| file.strip_suffix(".m"))
        .map(str::to_owned)
        // These form the canonical semantic suite factory. Runtime
        // compatibility is implemented in the testing layer; discovery must
        // lower `functiontests(localfunctions)` without executing either call
        // and without depending on native inventory registration.
        .chain(std::iter::once("functiontests".to_owned()))
        .collect::<HashSet<_>>();
    let context = LoweringContext::empty()
        .with_runmat_extensions_enabled(compat.allows_runmat_extensions())
        .with_known_project_symbols(&known_symbols);

    let mut assemblies = Vec::<(usize, HirAssembly)>::with_capacity(snapshot.sources.len());
    let mut diagnostics = Vec::new();
    for (source_index, source) in snapshot.sources.iter().enumerate() {
        let program = match runmat_parser::parse_with_options(
            &source.content,
            runmat_parser::ParserOptions::new(compat),
        ) {
            Ok(program) => program,
            Err(error) => {
                diagnostics.push(DiscoveryDiagnostic {
                    code: "RunMat:TestDiscovery:ParseError".into(),
                    message: error.message,
                    severity: DiscoveryDiagnosticSeverity::Error,
                    source: Some(source_at(
                        &source.owner_identity,
                        &source.relative_path,
                        &source.content,
                        error.position,
                        error.position.saturating_add(1),
                    )),
                });
                continue;
            }
        };
        match runmat_hir::lower(&program, &context) {
            Ok(lowering) => assemblies.push((source_index, lowering.assembly)),
            Err(error) => {
                let span = error.span.unwrap_or(runmat_hir::Span { start: 0, end: 0 });
                diagnostics.push(DiscoveryDiagnostic {
                    code: error
                        .identifier
                        .unwrap_or_else(|| "RunMat:TestDiscovery:LoweringError".into()),
                    message: error.message,
                    severity: DiscoveryDiagnosticSeverity::Error,
                    source: Some(source_at(
                        &source.owner_identity,
                        &source.relative_path,
                        &source.content,
                        span.start,
                        span.end,
                    )),
                });
            }
        }
    }

    let valid_sources = assemblies
        .iter()
        .map(|(source_index, assembly)| {
            let source = &snapshot.sources[*source_index];
            SemanticTestSource {
                owner_identity: &source.owner_identity,
                relative_source_identity: &source.relative_path,
                source_text: &source.content,
                assembly,
            }
        })
        .collect::<Vec<_>>();
    let mut discovery = runmat_hir::testing::discover_tests(&SemanticDiscoveryInput {
        program_revision: snapshot.program_revision.clone(),
        sources: &valid_sources,
    });
    discovery.diagnostics.extend(diagnostics);
    discovery.diagnostics.sort_by(|left, right| {
        left.source
            .as_ref()
            .map(|source| (&source.owner_identity, &source.relative_path))
            .cmp(
                &right
                    .source
                    .as_ref()
                    .map(|source| (&source.owner_identity, &source.relative_path)),
            )
            .then(left.code.cmp(&right.code))
            .then(left.message.cmp(&right.message))
    });
    discovery
}

fn failed_discovery(
    snapshot: &FrozenTestRunSnapshot,
    code: &str,
    message: String,
    source: Option<SourceDescriptor>,
) -> TestDiscovery {
    TestDiscovery {
        program_revision: snapshot.program_revision.clone(),
        suites: Vec::new(),
        pending_materialization: Vec::new(),
        diagnostics: vec![DiscoveryDiagnostic {
            code: code.into(),
            message,
            severity: DiscoveryDiagnosticSeverity::Error,
            source,
        }],
    }
}

fn source_at(
    owner_identity: &str,
    relative_path: &str,
    text: &str,
    start: usize,
    end: usize,
) -> SourceDescriptor {
    let start = start.min(text.len());
    let end = end.min(text.len()).max(start);
    let (start_line, start_column) = line_column(text, start);
    let (end_line, end_column) = line_column(text, end);
    SourceDescriptor {
        owner_identity: owner_identity.into(),
        relative_path: relative_path.into(),
        semantic_path: relative_path.into(),
        span: SourceSpan {
            start_byte: start.min(u32::MAX as usize) as u32,
            end_byte: end.min(u32::MAX as usize) as u32,
            start_line,
            start_column,
            end_line,
            end_column,
        },
    }
}

fn line_column(text: &str, offset: usize) -> (u32, u32) {
    let prefix = &text[..offset.min(text.len())];
    let line = prefix.bytes().filter(|byte| *byte == b'\n').count() as u32 + 1;
    let column = prefix
        .rsplit_once('\n')
        .map_or(prefix.len(), |(_, tail)| tail.len()) as u32
        + 1;
    (line, column)
}
