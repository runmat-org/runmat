//! Project checking composes only statically reached source units. Source bytes
//! and filesystem/revision admission remain explicit adapter inputs.
pub mod composition;
pub mod check_report;
mod reachability;

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::future::Future;
use std::path::PathBuf;

use runmat_filesystem::FsPathNamespace;
use runmat_hir::LoweringContext;
use runmat_package::{DiscoveredSourceSymbols, ProjectSymbolDefinition};
use runmat_parser::{CompatMode, ParserOptions};
use runmat_types::ProgramSourceId;

use crate::frontend::{self, AnalysisCompleteness, FrontendAnalysis};

#[derive(Debug)]
pub struct ProjectSource {
    /// An adapter-resolved absolute path in its explicitly supplied provider
    /// namespace, not a new filesystem lookup authority.
    pub path: PathBuf,
    pub text: String,
}

pub struct ProjectAnalysis {
    pub analysis: FrontendAnalysis,
    /// Source ordinals index this table. Text is retained once, not per call.
    pub sources: Vec<ProjectSource>,
    /// Reached units whose lexical composition is not yet representable.
    pub incomplete_sources: Vec<PathBuf>,
    pub call_context_budget_exhausted: bool,
    pub call_context_precision_withheld: bool,
}

impl ProjectAnalysis {
    pub fn source(&self, source: Option<ProgramSourceId>) -> Option<&ProjectSource> {
        self.sources
            .get(source.map_or(0, |source| source.0 as usize))
    }
}

#[derive(Debug)]
pub enum ProjectAnalysisError {
    InvalidPrimaryPath,
    InvalidSelectedPath { path: PathBuf },
    SourceRead { path: PathBuf, message: String },
    RevisionChanged { path: PathBuf },
    SourceOrdinalOverflow,
}

impl fmt::Display for ProjectAnalysisError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPrimaryPath => {
                formatter.write_str("project primary path must be adapter-resolved and absolute")
            }
            Self::InvalidSelectedPath { path } => write!(
                formatter,
                "selected project source path is not absolute: {}",
                path.display()
            ),
            Self::SourceRead { path, message } => write!(
                formatter,
                "cannot read selected source {}: {message}",
                path.display()
            ),
            Self::RevisionChanged { path } => write!(
                formatter,
                "selected source revision changed: {}",
                path.display()
            ),
            Self::SourceOrdinalOverflow => {
                formatter.write_str("project source ordinal exceeds the analysis schema")
            }
        }
    }
}

impl std::error::Error for ProjectAnalysisError {}

/// Analyze the source-local frontend plus the statically reached project closure.
/// The loader reads only definitions selected by this shared planner. It cannot
/// nominate additional symbols or change the selected catalog's precedence.
/// The namespace is the original provider's path grammar only; this frontend
/// never captures a provider, reads source bytes or consults ambient CWD.
pub async fn analyze_project_source<F, Fut>(
    primary: ProjectSource,
    namespace: FsPathNamespace,
    compat: CompatMode,
    catalog: Option<&DiscoveredSourceSymbols>,
    mut load: F,
) -> Result<ProjectAnalysis, ProjectAnalysisError>
where
    F: FnMut(ProjectSymbolDefinition) -> Fut,
    Fut: Future<Output = Result<String, String>>,
{
    if !namespace.is_absolute(&primary.path) {
        return Err(ProjectAnalysisError::InvalidPrimaryPath);
    }
    let symbols = catalog
        .map(|catalog| catalog.symbols.clone())
        .unwrap_or_default();
    let project_aliases = composition::project_symbol_aliases(catalog);
    let mut sources = vec![primary];
    let options = ParserOptions::new(compat);
    let mut program = match runmat_parser::parse_with_options(&sources[0].text, options) {
        Ok(program) => program,
        Err(_) => {
            let analysis = frontend::analyze_source_with_catalog(
                &sources[0].text,
                compat,
                &LoweringContext::empty()
                    .with_known_project_symbols(&symbols)
                    .with_runmat_extensions_enabled(compat.allows_runmat_extensions()),
                catalog,
            );
            return Ok(ProjectAnalysis {
                analysis,
                sources,
                incomplete_sources: Vec::new(),
                call_context_budget_exhausted: false,
                call_context_precision_withheld: false,
            });
        }
    };
    let mut statement_sources = vec![ProgramSourceId(0); program.body.len()];
    let mut seen = HashSet::from([reachability::path_key(&sources[0].path)]);
    let mut private_owners = HashMap::new();
    let mut private_aliases: HashMap<String, HashMap<String, String>> = HashMap::new();
    let mut incomplete_sources = Vec::new();
    let primary_directory = reachability::path_key(
        sources[0]
            .path
            .parent()
            .unwrap_or_else(|| std::path::Path::new("")),
    );
    let mut scope_paths = HashMap::from([(String::new(), HashSet::from([primary_directory]))]);
    let mut private_scope_paths = HashMap::new();
    loop {
        let pending = {
            let context = LoweringContext::empty()
                .with_known_project_symbols(&symbols)
                .with_project_symbol_aliases(&project_aliases)
                .with_private_functions(&private_owners, &private_aliases)
                .with_top_level_sources(&statement_sources)
                .with_primary_source(ProgramSourceId(0))
                .with_runmat_extensions_enabled(compat.allows_runmat_extensions());
            // Closure discovery needs identities and typed environment effects,
            // not a fixed-point analysis or executable bytecode per frontier.
            // A lowering failure is handled by the canonical final frontend.
            runmat_hir::lower(&program, &context)
                .ok()
                .and_then(|lowering| {
                    runmat_mir::lowering::lower_assembly(&lowering.assembly)
                        .ok()
                        .map(|mir| {
                            reachability::pending_sources(
                                &lowering,
                                &mir,
                                &sources,
                                catalog,
                                &seen,
                                &mut incomplete_sources,
                            )
                        })
                })
                .unwrap_or_default()
        };
        if pending.is_empty() {
            if !incomplete_sources.is_empty() {
                // Do not publish even generic cross-file facts from an assembly
                // whose lexical binding cannot be represented faithfully.
                let mut analysis = frontend::analyze_source_with_catalog(
                    &sources[0].text,
                    compat,
                    &LoweringContext::empty()
                        .with_known_project_symbols(&symbols)
                        .with_runmat_extensions_enabled(compat.allows_runmat_extensions()),
                    catalog,
                );
                analysis.domains.name_resolution = AnalysisCompleteness::Partial;
                analysis.domains.types = AnalysisCompleteness::Partial;
                analysis.domains.shapes = AnalysisCompleteness::Partial;
                return Ok(ProjectAnalysis {
                    analysis,
                    sources,
                    incomplete_sources,
                    call_context_budget_exhausted: false,
                    call_context_precision_withheld: true,
                });
            }
            let context = LoweringContext::empty()
                .with_known_project_symbols(&symbols)
                .with_project_symbol_aliases(&project_aliases)
                .with_private_functions(&private_owners, &private_aliases)
                .with_top_level_sources(&statement_sources)
                .with_primary_source(ProgramSourceId(0))
                .with_runmat_extensions_enabled(compat.allows_runmat_extensions());
            let mut analysis = frontend::analyze_program_with_catalog(&program, &context, catalog);
            let mut call_context_budget_exhausted = false;
            let mut call_context_precision_withheld = false;
            if let Some(mir) = &analysis.mir {
                let contextual = runmat_mir::analysis::diagnose_call_contexts(mir);
                call_context_budget_exhausted = contextual.budget_exhausted;
                call_context_precision_withheld = contextual.precision_withheld;
                analysis.extend_diagnostics(
                    contextual.diagnostics,
                    frontend::DiagnosticStage::Analysis,
                );
                analysis.sort_diagnostics(true);
            }
            if !incomplete_sources.is_empty() {
                analysis.domains.name_resolution = AnalysisCompleteness::Partial;
            }
            if call_context_budget_exhausted || call_context_precision_withheld {
                analysis.domains.types = AnalysisCompleteness::Partial;
                analysis.domains.shapes = AnalysisCompleteness::Partial;
            }
            return Ok(ProjectAnalysis {
                analysis,
                sources,
                incomplete_sources,
                call_context_budget_exhausted,
                call_context_precision_withheld,
            });
        }
        for definition in pending {
            if !namespace.is_absolute(&definition.source_path) {
                return Err(ProjectAnalysisError::InvalidSelectedPath {
                    path: definition.source_path,
                });
            }
            let key = reachability::path_key(&definition.source_path);
            if !seen.insert(key) {
                continue;
            }
            let text = load(definition.clone()).await.map_err(|message| {
                ProjectAnalysisError::SourceRead {
                    path: definition.source_path.clone(),
                    message,
                }
            })?;
            if definition.source_id.as_ref().is_some_and(|source| {
                runmat_package::ContentDigest::sha256(text.as_bytes()) != source.content_digest
            }) {
                return Err(ProjectAnalysisError::RevisionChanged {
                    path: definition.source_path,
                });
            }
            let ordinal = u32::try_from(sources.len())
                .map_err(|_| ProjectAnalysisError::SourceOrdinalOverflow)?;
            let source = ProgramSourceId(ordinal);
            sources.push(ProjectSource {
                path: definition.source_path.clone(),
                text,
            });
            let mut unit =
                match runmat_parser::parse_with_options(&sources[ordinal as usize].text, options) {
                    Ok(unit) => unit,
                    Err(_) => {
                        let mut analysis = frontend::analyze_source_with_catalog(
                            &sources[ordinal as usize].text,
                            compat,
                            &LoweringContext::empty()
                                .with_known_project_symbols(&symbols)
                                .with_runmat_extensions_enabled(compat.allows_runmat_extensions()),
                            catalog,
                        );
                        for diagnostic in &mut analysis.diagnostics {
                            diagnostic.set_source(source);
                        }
                        return Ok(ProjectAnalysis {
                            analysis,
                            sources,
                            incomplete_sources,
                            call_context_budget_exhausted: false,
                            call_context_precision_withheld: false,
                        });
                    }
                };
            // HIR's current top-level registry is global rather than module-local.
            // Do not rename multiple file-local functions into one identity or
            // invent nested capture semantics to make such a unit composable.
            let functions = unit
                .body
                .iter()
                .filter(|statement| matches!(statement, runmat_parser::Stmt::Function { .. }))
                .count();
            if functions > 1
                || !(composition::is_function_source_body(&unit.body)
                    || composition::is_class_source_body(&unit.body))
            {
                incomplete_sources.push(definition.source_path);
                continue;
            }
            let qualified = project_aliases
                .get(&definition.name)
                .unwrap_or(&definition.qualified_name);
            let owner = composition::function_owner_scope_from_qualified_name(qualified);
            let directory = if definition.is_private {
                definition
                    .source_path
                    .parent()
                    .and_then(std::path::Path::parent)
            } else {
                definition.source_path.parent()
            };
            let directory =
                reachability::path_key(directory.unwrap_or_else(|| std::path::Path::new("")));
            // The existing HIR private-alias key is a qualified-name prefix,
            // not a directory identity. Refuse an assembly that would leak a
            // private alias between unrelated source roots with the same key.
            let conflicting_private_scope = if definition.is_private {
                scope_paths
                    .get(&owner)
                    .is_some_and(|paths| paths.iter().any(|path| path != &directory))
            } else {
                private_scope_paths
                    .get(&owner)
                    .is_some_and(|path| path != &directory)
            };
            if conflicting_private_scope {
                incomplete_sources.push(definition.source_path);
                continue;
            }
            if definition.is_private {
                let visible =
                    reachability::private_visible(&sources[0].path, &definition.source_path);
                let aliases = composition::qualify_private_companion_functions(
                    &mut unit.body,
                    &owner,
                    visible,
                );
                if aliases.iter().any(|(name, target)| {
                    private_aliases
                        .get(&owner)
                        .and_then(|aliases| aliases.get(name))
                        .is_some_and(|existing| existing != target)
                }) {
                    incomplete_sources.push(definition.source_path);
                    continue;
                }
                for statement in &unit.body {
                    if let runmat_parser::Stmt::Function { name, .. } = statement {
                        private_owners.insert(name.clone(), owner.clone());
                    }
                }
                private_scope_paths.insert(owner.clone(), directory.clone());
                private_aliases.entry(owner).or_default().extend(aliases);
            } else if composition::is_class_source_body(&unit.body) {
                composition::qualify_companion_classdefs(&mut unit.body, qualified);
            } else {
                composition::qualify_companion_functions(&mut unit.body, qualified);
            }
            let owner = composition::function_owner_scope_from_qualified_name(qualified);
            scope_paths.entry(owner).or_default().insert(directory);
            statement_sources.extend(std::iter::repeat_n(source, unit.body.len()));
            program.body.extend(unit.body);
        }
    }
}

#[cfg(test)]
mod tests;
