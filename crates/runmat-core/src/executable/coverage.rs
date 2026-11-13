use std::collections::{BTreeMap, BTreeSet, HashMap};

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
        format!("sha256:{}", hex::encode(digest.finalize()))
    }

    #[cfg(test)]
    pub(crate) fn instrument(
        source: &ExecutableSource,
        revision: &ExecutableRevision,
        source_map: &ExecutableSourceMap,
        bytecode: &mut runmat_vm::Bytecode,
    ) -> Self {
        let registered_bodies = bytecode
            .function_registry
            .functions
            .keys()
            .copied()
            .collect();
        Self::instrument_for_product(source, revision, source_map, bytecode, &registered_bodies)
    }

    pub(crate) fn instrument_for_product(
        source: &ExecutableSource,
        revision: &ExecutableRevision,
        source_map: &ExecutableSourceMap,
        bytecode: &mut runmat_vm::Bytecode,
        registered_bodies: &BTreeSet<runmat_hir::FunctionId>,
    ) -> Self {
        let mut builder = CoveragePlanBuilder::new(source, revision, source_map);
        instrument_program(
            &mut builder,
            CoverageProcedureIdentity::Entrypoint,
            "<entrypoint>",
            bytecode.source_id,
            &bytecode.instr_spans,
            &mut bytecode.coverage_sites,
        );

        let mut function_ids = bytecode
            .function_registry
            .functions
            .keys()
            .filter(|id| registered_bodies.contains(id))
            .copied()
            .collect::<Vec<_>>();
        function_ids.sort_by_key(|function| function.0);
        for function_id in function_ids {
            if let Some(function) = bytecode.function_registry.functions.get_mut(&function_id) {
                let procedure = function_semantic_identity(function);
                instrument_program(
                    &mut builder,
                    procedure,
                    &function.display_name,
                    function.source_id,
                    &function.instr_spans,
                    &mut function.coverage_sites,
                );
            }
        }
        for (function_id, function) in &mut bytecode.bound_functions {
            if !registered_bodies.contains(function_id) {
                continue;
            }
            if let Some(instrumented) = bytecode.function_registry.functions.get(function_id) {
                function.coverage_sites = instrumented.coverage_sites.clone();
            } else {
                let procedure = function_semantic_identity(function);
                instrument_program(
                    &mut builder,
                    procedure,
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
    fallback_content_revision: runmat_execution::Digest,
    source_map: &'a ExecutableSourceMap,
    indexes: HashMap<SiteKey, u64>,
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
            fallback_content_revision: revision
                .source_digest
                .parse()
                .expect("executable revision carries a canonical source digest"),
            source_map,
            indexes: HashMap::new(),
            sites: Vec::new(),
        }
    }

    fn site(
        &mut self,
        metric: CoverageMetric,
        procedure: CoverageProcedureIdentity,
        semantic_path: &str,
        source_id: Option<runmat_vm::BytecodeSourceRef>,
        span: runmat_hir::Span,
        instrumented: bool,
    ) -> u64 {
        let source_id = source_id
            .and_then(runmat_vm::BytecodeSourceRef::program)
            .unwrap_or(runmat_types::ProgramSourceId(0));
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
        let content_revision = source
            .map(|entry| entry.content_revision)
            .unwrap_or(self.fallback_content_revision);
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
            content_revision,
            procedure,
            start_byte: span.start,
            end_byte: span.end,
        };
        if let Some(index) = self.indexes.get(&key) {
            return *index;
        }
        let (id, counter_key) = stable_site_id(&key);
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
            semantic_path: semantic_path.into(),
            source_id: usize::try_from(source_id.0).expect("validated program source ordinal"),
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

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum CoverageProcedureIdentity {
    Entrypoint,
    Bound(runmat_execution::FunctionBindingOrigin),
    Portable(runmat_types::ProgramFunctionId),
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct SiteKey {
    kind_rank: u8,
    owner_identity: String,
    relative_path: String,
    content_revision: runmat_execution::Digest,
    procedure: CoverageProcedureIdentity,
    start_byte: usize,
    end_byte: usize,
}

fn instrument_program(
    builder: &mut CoveragePlanBuilder<'_>,
    procedure: CoverageProcedureIdentity,
    semantic_path: &str,
    source_id: Option<runmat_vm::BytecodeSourceRef>,
    spans: &[runmat_hir::Span],
    mapping: &mut Vec<Vec<u64>>,
) {
    let mut instrumented_statements = std::collections::BTreeSet::new();
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
        procedure,
        semantic_path,
        source_id,
        function_span,
        first.is_some(),
    );
    if let Some(first) = first {
        mapping[first].push(function);
    }
    for (pc, span) in spans.iter().copied().enumerate() {
        if nonempty_span(&span) && instrumented_statements.insert((span.start, span.end)) {
            let statement = builder.site(
                CoverageMetric::Statement,
                procedure,
                semantic_path,
                source_id,
                span,
                true,
            );
            mapping[pc].push(statement);
        }
    }
}

fn function_semantic_identity(function: &runmat_vm::FunctionBytecode) -> CoverageProcedureIdentity {
    function.binding_origin.map_or(
        CoverageProcedureIdentity::Portable(function.portable_function),
        CoverageProcedureIdentity::Bound,
    )
}

fn nonempty_span(span: &runmat_hir::Span) -> bool {
    span.end > span.start
}

fn stable_site_id(key: &SiteKey) -> (String, u64) {
    let mut digest = Sha256::new();
    frame(&mut digest, b"runmat-coverage-site-v2");
    frame(&mut digest, &[key.kind_rank]);
    frame(&mut digest, key.owner_identity.as_bytes());
    frame(&mut digest, key.relative_path.as_bytes());
    frame(&mut digest, key.content_revision.bytes());
    match key.procedure {
        CoverageProcedureIdentity::Entrypoint => frame(&mut digest, &[0]),
        CoverageProcedureIdentity::Bound(origin) => {
            frame(&mut digest, &[1]);
            frame(&mut digest, origin.program.bytes());
            frame(&mut digest, &origin.function.0.to_be_bytes());
        }
        CoverageProcedureIdentity::Portable(function) => {
            frame(&mut digest, &[2]);
            frame(&mut digest, &function.0.to_be_bytes());
        }
    }
    frame(&mut digest, &(key.start_byte as u64).to_be_bytes());
    frame(&mut digest, &(key.end_byte as u64).to_be_bytes());
    let digest = digest.finalize();
    let counter_key = u64::from_be_bytes(
        digest[..8]
            .try_into()
            .expect("SHA-256 digest contains eight key bytes"),
    );
    (format!("sha256:{}", hex::encode(digest)), counter_key)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExecutableSource, ExecutableSourceMap, SourceMapEntry};
    use runmat_hir::{FunctionId, Span};
    use runmat_vm::{Bytecode, BytecodeSourceRef, FunctionBytecode, FunctionRegistry};
    use std::collections::HashMap;

    fn function(
        id: usize,
        source: u32,
        span: Span,
        origin: Option<runmat_execution::FunctionBindingOrigin>,
    ) -> FunctionBytecode {
        FunctionBytecode {
            function: FunctionId(id),
            display_name: format!("function_{id}"),
            portable_function: runmat_types::ProgramFunctionId(
                u32::try_from(id).expect("test function identity fits portable schema"),
            ),
            source_id: Some(BytecodeSourceRef::Program(runmat_types::ProgramSourceId(
                source,
            ))),
            binding_origin: origin,
            instr_spans: vec![span],
            instructions: vec![runmat_vm::Instr::Return],
            ..FunctionBytecode::default()
        }
    }

    fn source_entry(ordinal: u32, owner: &str, path: &str, text: &str) -> SourceMapEntry {
        SourceMapEntry {
            source_id: runmat_types::ProgramSourceId(ordinal),
            owner_identity: owner.into(),
            relative_path: path.into(),
            content_revision: runmat_execution::Digest::sha256(text.as_bytes()),
            display_name: path.into(),
            full_path: None,
            text: text.into(),
        }
    }

    fn instrument(
        primary: &ExecutableSource,
        source_map: &ExecutableSourceMap,
        source: u32,
        origin: Option<runmat_execution::FunctionBindingOrigin>,
    ) -> (Vec<Vec<u64>>, CoveragePlan) {
        let mut bytecode = Bytecode::empty();
        let retained = function(7, source, Span { start: 0, end: 5 }, origin);
        bytecode.function_registry =
            FunctionRegistry::new(HashMap::from([(FunctionId(7), retained.clone())]));
        bytecode.bound_functions.insert(FunctionId(7), retained);
        let revision = crate::ExecutableRevision::derive(
            primary,
            None,
            crate::program_environment(crate::CompatMode::RunMat),
        );
        let plan = CoveragePlan::instrument(primary, &revision, source_map, &mut bytecode);
        (
            bytecode.function_registry.functions[&FunctionId(7)]
                .coverage_sites
                .clone(),
            plan,
        )
    }

    #[test]
    fn retained_coverage_identity_is_stable_across_packaging_and_reported_by_current_plan() {
        let retained_text = "y = x + 1;";
        let retained = ExecutableSource::new("coverage-owner", "dependency.m", retained_text);
        let original_map = ExecutableSourceMap::new(vec![source_entry(
            0,
            &retained.owner_identity,
            &retained.relative_path,
            retained_text,
        )]);
        let (original_sites, _) = instrument(&retained, &original_map, 0, None);

        let primary = ExecutableSource::new("caller-owner", "caller.m", "z = dependency(1);");
        let repackaged_map = ExecutableSourceMap::new(vec![
            source_entry(0, "caller-owner", "caller.m", &primary.text),
            source_entry(1, "coverage-owner", "dependency.m", retained_text),
        ]);
        let (repackaged_sites, current_plan) = instrument(&primary, &repackaged_map, 1, None);

        assert_eq!(repackaged_sites, original_sites);
        let current_keys = current_plan
            .sites()
            .iter()
            .map(|site| site.counter_key)
            .collect::<std::collections::BTreeSet<_>>();
        assert!(repackaged_sites
            .iter()
            .flatten()
            .all(|site| current_keys.contains(site)));
        let executed_retained_site = repackaged_sites[0][0];
        let fragment = current_plan.fragment(
            "current-package".into(),
            BTreeMap::from([(executed_retained_site, 3)]),
        );
        assert_eq!(fragment.counts.get(&executed_retained_site), Some(&3));
        assert!(fragment
            .sites
            .iter()
            .any(|site| site.counter_key == executed_retained_site
                && site.owner_identity == "coverage-owner"
                && site.relative_path == "dependency.m"));

        let changed_map = ExecutableSourceMap::new(vec![
            source_entry(0, "caller-owner", "caller.m", &primary.text),
            source_entry(1, "coverage-owner", "dependency.m", "y = x + 2;"),
        ]);
        let (changed_sites, _) = instrument(&primary, &changed_map, 1, None);
        assert_ne!(changed_sites, original_sites);

        let retained_revision = crate::ExecutableRevision::derive(
            &retained,
            None,
            crate::program_environment(crate::CompatMode::RunMat),
        );
        let origin = runmat_execution::FunctionBindingOrigin::from_revision(
            &retained_revision.program_revision,
            runmat_types::ProgramFunctionId(7),
        )
        .expect("bound test origin");
        let (bound_original, _) = instrument(&retained, &original_map, 0, Some(origin));
        let (bound_repackaged, _) = instrument(&primary, &repackaged_map, 1, Some(origin));
        assert_eq!(bound_repackaged, bound_original);
        assert_ne!(bound_original, original_sites);
    }
}
