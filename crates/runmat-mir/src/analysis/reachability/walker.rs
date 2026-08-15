mod resolution;
mod visit;

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use runmat_hir::{FunctionId, FunctionKind};

use super::{
    ReachabilityCertainty as Certainty, ReachabilityEdge, ReachabilityNames, ReachabilityNode,
    ReachabilityNodeKind as Kind, ReachabilityReason as Reason, ReachabilityReport,
    REACHABILITY_SCHEMA_VERSION,
};

pub fn analyze_reachability(
    assembly: &crate::MirAssembly,
    names: &ReachabilityNames,
) -> ReachabilityReport {
    Walker::new(assembly, names).run()
}

struct Walker<'a> {
    assembly: &'a crate::MirAssembly,
    names: &'a ReachabilityNames,
    nodes: BTreeMap<String, ReachabilityNode>,
    edges: BTreeSet<ReachabilityEdge>,
    pending: VecDeque<FunctionId>,
    scanned: BTreeSet<FunctionId>,
    current_function: Option<FunctionId>,
}

impl<'a> Walker<'a> {
    fn new(assembly: &'a crate::MirAssembly, names: &'a ReachabilityNames) -> Self {
        Self {
            assembly,
            names,
            nodes: BTreeMap::new(),
            edges: BTreeSet::new(),
            pending: VecDeque::new(),
            scanned: BTreeSet::new(),
            current_function: None,
        }
    }

    fn run(mut self) -> ReachabilityReport {
        for function in &self.assembly.entrypoints {
            self.retain_function(
                *function,
                Certainty::Definite,
                None,
                Reason::Entrypoint,
                None,
            );
        }
        for (function, metadata) in &self.assembly.functions {
            let is_public = metadata.parent.is_none()
                && !self.assembly.entrypoints.contains(function)
                && matches!(
                    metadata.kind,
                    FunctionKind::Named | FunctionKind::ClassMethod { .. }
                );
            if is_public {
                self.retain_function(
                    *function,
                    Certainty::FiniteDynamic,
                    None,
                    Reason::PublicCallable,
                    Some("externally addressable program callable".into()),
                );
            }
        }
        while let Some(function) = self.pending.pop_front() {
            if !self.scanned.insert(function) {
                continue;
            }
            self.current_function = Some(function);
            if let Some(body) = self.assembly.bodies.get(&function) {
                let from = function_id(function);
                for block in &body.blocks {
                    for statement in &block.statements {
                        self.statement(&from, statement);
                    }
                    self.terminator(&from, &block.terminator.kind);
                }
            }
        }
        self.current_function = None;
        let has_unknown_edges = self
            .edges
            .iter()
            .any(|edge| edge.certainty == Certainty::Unknown);
        ReachabilityReport {
            schema_version: REACHABILITY_SCHEMA_VERSION,
            nodes: self.nodes.into_values().collect(),
            edges: self.edges.into_iter().collect(),
            has_unknown_edges,
        }
    }

    fn retain_function(
        &mut self,
        function: FunctionId,
        certainty: Certainty,
        from: Option<String>,
        reason: Reason,
        detail: Option<String>,
    ) {
        let id = function_id(function);
        let (module, symbol) = self
            .assembly
            .functions
            .get(&function)
            .map(|metadata| {
                (
                    self.names
                        .sources
                        .get(&metadata.source)
                        .cloned()
                        .unwrap_or_else(|| format!("source:{}", metadata.source.0)),
                    metadata.name.0.clone(),
                )
            })
            .unwrap_or_else(|| ("program".into(), format!("function#{}", function.0)));
        self.node(id.clone(), Kind::Function, module, symbol, certainty);
        self.edge(from, id, certainty, reason, detail);
        self.pending.push_back(function);
    }

    fn node(
        &mut self,
        id: String,
        kind: Kind,
        module: String,
        symbol: String,
        certainty: Certainty,
    ) {
        self.nodes
            .entry(id.clone())
            .and_modify(|node| node.certainty = node.certainty.min(certainty))
            .or_insert(ReachabilityNode {
                id,
                kind,
                module,
                symbol,
                certainty,
            });
    }

    fn edge(
        &mut self,
        from: Option<String>,
        to: String,
        certainty: Certainty,
        reason: Reason,
        detail: Option<String>,
    ) {
        self.edges.insert(ReachabilityEdge {
            from,
            to,
            certainty,
            reason,
            detail,
        });
    }
}

fn function_id(function: FunctionId) -> String {
    format!("function:{}", function.0)
}

fn kind_token(kind: Kind) -> &'static str {
    match kind {
        Kind::Function => "function",
        Kind::Builtin => "builtin",
        Kind::Class => "class",
        Kind::Method => "method",
        Kind::Provider => "provider",
        Kind::Extension => "extension",
        Kind::ArtifactDependency => "artifact_dependency",
        Kind::GlobalState => "global",
        Kind::PersistentState => "persistent",
        Kind::RuntimeCatalog => "runtime_catalog",
        Kind::ExternalCallable => "external_callable",
    }
}
