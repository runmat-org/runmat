mod resolution;
mod visit;

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use runmat_hir::{FunctionId, FunctionKind};
use runmat_types::BuiltinId;

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

/// The typed retention set and its presentation report from one canonical walk.
pub fn analyze_reachability_with_functions(
    assembly: &crate::MirAssembly,
    names: &ReachabilityNames,
) -> (ReachabilityReport, BTreeSet<FunctionId>) {
    let (report, functions, _) = Walker::new(assembly, names).run_analysis();
    (report, functions)
}

/// Exact function-to-function edges found by the canonical reachability walk.
/// Presentation IDs in `ReachabilityReport` are never parsed for admission.
pub fn function_dependencies(
    assembly: &crate::MirAssembly,
    roots: &BTreeSet<FunctionId>,
) -> BTreeMap<FunctionId, BTreeSet<FunctionId>> {
    Walker::new(assembly, &ReachabilityNames::default())
        .with_roots(roots.iter().copied())
        .run_analysis()
        .2
}

/// Exact static callable closure for one selected native entry. Unlike the
/// report and its existing dependency projection, unrelated public methods
/// are not implicit roots. Dynamic edges remain deferred to invocation.
pub fn selected_function_closure(
    assembly: &crate::MirAssembly,
    roots: &BTreeSet<FunctionId>,
) -> BTreeSet<FunctionId> {
    Walker::new(assembly, &ReachabilityNames::default())
        .with_selected_roots(roots.iter().copied())
        .run_analysis()
        .1
}

/// Walk only the bodies included in one function-only product. Omitted Local
/// siblings belong to their own sealed imported products; neither their body
/// effects nor their presentation nodes belong to this product's manifest.
pub fn selected_body_reachability(
    assembly: &crate::MirAssembly,
    names: &ReachabilityNames,
    included_bodies: &BTreeSet<FunctionId>,
) -> (ReachabilityReport, BTreeSet<FunctionId>) {
    let (report, functions, _) = Walker::new(assembly, names)
        .with_selected_bodies(included_bodies)
        .run_analysis();
    (report, functions)
}

struct Walker<'a> {
    assembly: &'a crate::MirAssembly,
    names: &'a ReachabilityNames,
    nodes: BTreeMap<String, ReachabilityNode>,
    edges: BTreeSet<ReachabilityEdge>,
    pending: VecDeque<FunctionId>,
    scanned: BTreeSet<FunctionId>,
    extra_roots: Vec<FunctionId>,
    selected_roots_only: bool,
    selected_bodies: Option<BTreeSet<FunctionId>>,
    function_dependencies: BTreeMap<FunctionId, BTreeSet<FunctionId>>,
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
            extra_roots: Vec::new(),
            selected_roots_only: false,
            selected_bodies: None,
            function_dependencies: BTreeMap::new(),
            current_function: None,
        }
    }

    fn with_roots(mut self, roots: impl IntoIterator<Item = FunctionId>) -> Self {
        self.extra_roots.extend(roots);
        self
    }

    fn with_selected_roots(mut self, roots: impl IntoIterator<Item = FunctionId>) -> Self {
        self.extra_roots.extend(roots);
        self.selected_roots_only = true;
        self
    }

    fn with_selected_bodies(mut self, bodies: &BTreeSet<FunctionId>) -> Self {
        self.extra_roots.extend(bodies.iter().copied());
        self.selected_roots_only = true;
        self.selected_bodies = Some(bodies.clone());
        self
    }

    fn run(self) -> ReachabilityReport {
        self.run_analysis().0
    }

    fn run_analysis(
        mut self,
    ) -> (
        ReachabilityReport,
        BTreeSet<FunctionId>,
        BTreeMap<FunctionId, BTreeSet<FunctionId>>,
    ) {
        if !self.selected_roots_only {
            for function in &self.assembly.entrypoints {
                self.retain_function(
                    *function,
                    Certainty::Definite,
                    None,
                    Reason::Entrypoint,
                    None,
                );
            }
        }
        for function in std::mem::take(&mut self.extra_roots) {
            self.retain_function(
                function,
                Certainty::Definite,
                None,
                Reason::PublicCallable,
                None,
            );
        }
        for (function, metadata) in &self.assembly.functions {
            if self.selected_roots_only {
                continue;
            }
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
            if self
                .selected_bodies
                .as_ref()
                .is_some_and(|bodies| !bodies.contains(&function))
            {
                continue;
            }
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
        let report = ReachabilityReport {
            schema_version: REACHABILITY_SCHEMA_VERSION,
            nodes: self.nodes.into_values().collect(),
            edges: self.edges.into_iter().collect(),
            has_unknown_edges,
        };
        (report, self.scanned, self.function_dependencies)
    }

    fn retain_function(
        &mut self,
        function: FunctionId,
        certainty: Certainty,
        from: Option<String>,
        reason: Reason,
        detail: Option<String>,
    ) {
        if from.is_some() {
            if let Some(caller) = self.current_function {
                self.function_dependencies
                    .entry(caller)
                    .or_default()
                    .insert(function);
            }
        }
        if self
            .selected_bodies
            .as_ref()
            .is_some_and(|bodies| !bodies.contains(&function))
        {
            return;
        }
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

    fn builtin_node(
        &mut self,
        id: String,
        builtin: BuiltinId,
        symbol: String,
        certainty: Certainty,
    ) {
        self.nodes
            .entry(id.clone())
            .and_modify(|node| {
                node.certainty = node.certainty.min(certainty);
                debug_assert_eq!(node.builtin_id(), Some(&builtin));
            })
            .or_insert(ReachabilityNode {
                id,
                kind: Kind::Builtin(builtin),
                module: "runmat-builtins".into(),
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

fn kind_token(kind: &Kind) -> &'static str {
    match kind {
        Kind::Function => "function",
        Kind::Builtin(_) => "builtin",
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
