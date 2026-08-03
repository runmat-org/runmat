use std::collections::BTreeMap;

use runmat_test::coverage::{CoverageFragment, CoverageMetric, CoverageSite};
use sha2::{Digest, Sha256};

use super::{ExecutableRevision, ExecutableSource, ExecutableSourceMap};

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CoveragePlan {
    sites: Vec<CoverageSite>,
}

impl CoveragePlan {
    pub fn sites(&self) -> &[CoverageSite] {
        &self.sites
    }

    pub fn revision(&self) -> String {
        let mut digest = Sha256::new();
        frame(&mut digest, b"runmat-coverage-plan-v1");
        for site in &self.sites {
            frame(&mut digest, site.id.as_bytes());
        }
        format!("sha256:{:x}", digest.finalize())
    }

    pub(crate) fn instrument(
        source: &ExecutableSource,
        revision: &ExecutableRevision,
        source_map: &ExecutableSourceMap,
        bytecode: &mut runmat_vm::Bytecode,
    ) -> Self {
        let mut builder = CoveragePlanBuilder::new(source, revision, source_map);
        instrument_program(
            &mut builder,
            "<entrypoint>",
            bytecode.source_id,
            &bytecode.instr_spans,
            &mut bytecode.coverage_sites,
        );

        let mut function_ids = bytecode
            .function_registry
            .functions
            .keys()
            .copied()
            .collect::<Vec<_>>();
        function_ids.sort_by_key(|function| function.0);
        for function_id in function_ids {
            if let Some(function) = bytecode.function_registry.functions.get_mut(&function_id) {
                instrument_program(
                    &mut builder,
                    &function.display_name,
                    function.source_id,
                    &function.instr_spans,
                    &mut function.coverage_sites,
                );
            }
        }
        for (function_id, function) in &mut bytecode.bound_functions {
            if let Some(instrumented) = bytecode.function_registry.functions.get(function_id) {
                function.coverage_sites = instrumented.coverage_sites.clone();
            } else {
                instrument_program(
                    &mut builder,
                    &function.display_name,
                    function.source_id,
                    &function.instr_spans,
                    &mut function.coverage_sites,
                );
            }
        }
        Self {
            sites: builder.sites,
        }
    }

    pub fn fragment(
        &self,
        program_revision: String,
        counts: BTreeMap<u64, u64>,
    ) -> CoverageFragment {
        CoverageFragment {
            program_revision,
            plan_revision: self.revision(),
            sites: self.sites.clone(),
            counts,
        }
    }
}

struct CoveragePlanBuilder<'a> {
    source: &'a ExecutableSource,
    revision: &'a ExecutableRevision,
    source_map: &'a ExecutableSourceMap,
    indexes: BTreeMap<SiteKey, u64>,
    sites: Vec<CoverageSite>,
}

impl<'a> CoveragePlanBuilder<'a> {
    fn new(
        source: &'a ExecutableSource,
        revision: &'a ExecutableRevision,
        source_map: &'a ExecutableSourceMap,
    ) -> Self {
        Self {
            source,
            revision,
            source_map,
            indexes: BTreeMap::new(),
            sites: Vec::new(),
        }
    }

    fn site(
        &mut self,
        metric: CoverageMetric,
        semantic_path: &str,
        source_id: Option<runmat_hir::SourceId>,
        span: runmat_hir::Span,
        instrumented: bool,
    ) -> u64 {
        let source_id = source_id.map_or(0, |id| id.0);
        let source = self
            .source_map
            .entries()
            .iter()
            .find(|entry| entry.source_id == source_id);
        let owner_identity = source
            .map(|entry| entry.owner_identity.as_str())
            .unwrap_or(&self.source.owner_identity);
        let relative_path = source
            .map(|entry| entry.relative_path.as_str())
            .unwrap_or(&self.source.relative_path);
        let key = SiteKey {
            kind_rank: match metric {
                CoverageMetric::Function => 0,
                CoverageMetric::Statement => 1,
                CoverageMetric::Decision => 2,
                CoverageMetric::Condition => 3,
                CoverageMetric::McdcCondition => 4,
            },
            owner_identity: owner_identity.into(),
            relative_path: relative_path.into(),
            semantic_path: semantic_path.into(),
            start_byte: span.start,
            end_byte: span.end,
        };
        if let Some(index) = self.indexes.get(&key) {
            return *index;
        }
        let (id, counter_key) = stable_site_id(self.revision, &key);
        if let Some(collision) = self
            .sites
            .iter()
            .find(|site| site.counter_key == counter_key && site.id != id)
        {
            panic!(
                "coverage counter-key collision between '{}' and '{}'",
                collision.id, id
            );
        }
        let text = source.map(|entry| entry.text.as_str()).unwrap_or_default();
        let (start_line, start_column) = line_column(text, span.start);
        let (end_line, end_column) = line_column(text, span.end);
        self.sites.push(CoverageSite {
            id,
            counter_key,
            metric,
            owner_identity: key.owner_identity.clone(),
            relative_path: key.relative_path.clone(),
            semantic_path: key.semantic_path.clone(),
            source_id,
            start_byte: span.start,
            end_byte: span.end,
            start_line,
            start_column,
            end_line,
            end_column,
            instrumented,
            unsupported_reason: (!instrumented)
                .then(|| "procedure has no executable bytecode".into()),
        });
        self.indexes.insert(key, counter_key);
        counter_key
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct SiteKey {
    kind_rank: u8,
    owner_identity: String,
    relative_path: String,
    semantic_path: String,
    start_byte: usize,
    end_byte: usize,
}

fn instrument_program(
    builder: &mut CoveragePlanBuilder<'_>,
    semantic_path: &str,
    source_id: Option<runmat_hir::SourceId>,
    spans: &[runmat_hir::Span],
    mapping: &mut Vec<Vec<u64>>,
) {
    mapping.clear();
    mapping.resize(spans.len(), Vec::new());
    let first = spans.iter().position(nonempty_span);
    let function_span = first
        .map(|start| {
            spans
                .iter()
                .skip(start)
                .filter(|span| nonempty_span(span))
                .fold(spans[start], |aggregate, span| runmat_hir::Span {
                    start: aggregate.start.min(span.start),
                    end: aggregate.end.max(span.end),
                })
        })
        .unwrap_or_default();
    let function = builder.site(
        CoverageMetric::Function,
        semantic_path,
        source_id,
        function_span,
        first.is_some(),
    );
    if let Some(first) = first {
        mapping[first].push(function);
    }
    for (pc, span) in spans.iter().copied().enumerate() {
        if nonempty_span(&span) {
            let statement = builder.site(
                CoverageMetric::Statement,
                semantic_path,
                source_id,
                span,
                true,
            );
            mapping[pc].push(statement);
        }
    }
}

fn nonempty_span(span: &runmat_hir::Span) -> bool {
    span.end > span.start
}

fn stable_site_id(revision: &ExecutableRevision, key: &SiteKey) -> (String, u64) {
    let mut digest = Sha256::new();
    frame(&mut digest, b"runmat-coverage-site-v1");
    let revision = revision
        .program_revision
        .as_ref()
        .map(runmat_execution::ProgramRevision::canonical_identity)
        .unwrap_or_else(|| revision.source_digest.clone());
    frame(&mut digest, revision.as_bytes());
    frame(&mut digest, &[key.kind_rank]);
    frame(&mut digest, key.owner_identity.as_bytes());
    frame(&mut digest, key.relative_path.as_bytes());
    frame(&mut digest, key.semantic_path.as_bytes());
    frame(&mut digest, &(key.start_byte as u64).to_be_bytes());
    frame(&mut digest, &(key.end_byte as u64).to_be_bytes());
    let digest = digest.finalize();
    let counter_key = u64::from_be_bytes(
        digest[..8]
            .try_into()
            .expect("SHA-256 digest contains eight key bytes"),
    );
    (format!("sha256:{digest:x}"), counter_key)
}

fn line_column(text: &str, byte: usize) -> (u32, u32) {
    let mut bounded = byte.min(text.len());
    while !text.is_char_boundary(bounded) {
        bounded -= 1;
    }
    let prefix = &text[..bounded];
    let line = prefix.bytes().filter(|byte| *byte == b'\n').count() as u32 + 1;
    let column = prefix
        .rsplit_once('\n')
        .map_or(prefix.chars().count(), |(_, tail)| tail.chars().count()) as u32
        + 1;
    (line, column)
}

fn frame(digest: &mut Sha256, bytes: &[u8]) {
    digest.update((bytes.len() as u64).to_be_bytes());
    digest.update(bytes);
}
