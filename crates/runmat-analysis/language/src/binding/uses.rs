use std::collections::BTreeMap;

use runmat_design_syntax::{
    FeaBlock, FeaCallArgument, FeaExpression, FeaExpressionKind, FeaModuleItem, FeaReference,
    FeaStatementKind,
};

use crate::{
    BoundFeaSymbol, BoundFeaSymbolUse, FeaModuleGraph, FeaSymbolKey, FeaSymbolKind,
    FeaSymbolOrigin, FunctionFacts, LogicalSourcePath,
};

pub(super) fn collect_symbol_uses(
    graph: &FeaModuleGraph,
    symbols: &BTreeMap<LogicalSourcePath, BTreeMap<FeaSymbolKey, BoundFeaSymbol>>,
    functions: &BTreeMap<(LogicalSourcePath, String), FunctionFacts>,
) -> Vec<BoundFeaSymbolUse> {
    let mut uses = Vec::new();
    for (module_path, module) in &graph.modules {
        let Some(syntax) = &module.syntax else {
            continue;
        };
        for item in &syntax.items {
            let FeaModuleItem::Declaration(declaration) = item else {
                continue;
            };
            let mut collector = UseCollector {
                module: module_path,
                symbols,
                functions,
                uses: &mut uses,
            };
            if let Some(source) = &declaration.source {
                collector.reference(source);
            }
            if let Some(source) = &declaration.material_source {
                collector.expression(source);
            }
            if let Some(target) = &declaration.target {
                collector.reference(target);
            }
            collector.block(&declaration.body);
        }
    }
    uses.sort_by(|left, right| {
        (&left.module, left.span.start, left.span.end).cmp(&(
            &right.module,
            right.span.start,
            right.span.end,
        ))
    });
    uses
}

struct UseCollector<'a> {
    module: &'a LogicalSourcePath,
    symbols: &'a BTreeMap<LogicalSourcePath, BTreeMap<FeaSymbolKey, BoundFeaSymbol>>,
    functions: &'a BTreeMap<(LogicalSourcePath, String), FunctionFacts>,
    uses: &'a mut Vec<BoundFeaSymbolUse>,
}

impl UseCollector<'_> {
    fn block(&mut self, block: &FeaBlock) {
        for statement in &block.statements {
            match &statement.kind {
                FeaStatementKind::GeneratedByFunction { function, body } => {
                    self.reference(function);
                    self.block(body);
                }
                FeaStatementKind::UniformSample { start, end, rate } => {
                    self.expression(start);
                    self.expression(end);
                    self.expression(rate);
                }
                FeaStatementKind::AppliedLoad { target, value, .. } => {
                    self.expression(target);
                    self.expression(value);
                }
                FeaStatementKind::SignalDefinition { body, .. }
                | FeaStatementKind::ModelDefinition { body, .. }
                | FeaStatementKind::Block { body, .. }
                | FeaStatementKind::OutputDefinition { body, .. } => self.block(body),
                FeaStatementKind::Input { default, .. } => {
                    if let Some(default) = default {
                        self.expression(default);
                    }
                }
                FeaStatementKind::Let { value, .. }
                | FeaStatementKind::Feature { value, .. }
                | FeaStatementKind::Include(value)
                | FeaStatementKind::Retain(value)
                | FeaStatementKind::Output { value, .. } => self.expression(value),
                FeaStatementKind::SourceParameterUpdate { target, value }
                | FeaStatementKind::Requirement { target, value }
                | FeaStatementKind::Assignment { target, value, .. } => {
                    self.reference(target);
                    self.expression(value);
                }
                FeaStatementKind::Use(reference) => self.reference(reference),
                FeaStatementKind::PropertyAssignment { property, target } => {
                    self.expression(property);
                    self.expression(target);
                }
                FeaStatementKind::Command { .. } => {}
            }
        }
    }

    fn expression(&mut self, expression: &FeaExpression) {
        match &expression.kind {
            FeaExpressionKind::Reference(reference) => self.reference(reference),
            FeaExpressionKind::List(values) => {
                for value in values {
                    self.expression(value);
                }
            }
            FeaExpressionKind::Record(entries) => {
                for entry in entries {
                    self.expression(&entry.value);
                }
            }
            FeaExpressionKind::Group(value)
            | FeaExpressionKind::Unary { operand: value, .. }
            | FeaExpressionKind::Quantity { value, .. } => self.expression(value),
            FeaExpressionKind::Binary { left, right, .. } => {
                self.expression(left);
                self.expression(right);
            }
            FeaExpressionKind::Call { callee, arguments } => {
                self.expression(callee);
                self.arguments(arguments);
            }
            FeaExpressionKind::PrefixApplication { callee, argument } => {
                self.expression(callee);
                self.expression(argument);
            }
            FeaExpressionKind::Number(_)
            | FeaExpressionKind::String(_)
            | FeaExpressionKind::Missing => {}
        }
    }

    fn arguments(&mut self, arguments: &[FeaCallArgument]) {
        for argument in arguments {
            match argument {
                FeaCallArgument::Positional(value) | FeaCallArgument::Named { value, .. } => {
                    self.expression(value);
                }
            }
        }
    }

    fn reference(&mut self, reference: &FeaReference) {
        let [namespace, name, ..] = reference.segments.as_slice() else {
            return;
        };
        let Some(kind) = symbol_kind(&namespace.text) else {
            return;
        };
        let key = FeaSymbolKey {
            kind,
            name: name.text.clone(),
        };
        let Some(symbol) = self
            .symbols
            .get(self.module)
            .and_then(|symbols| symbols.get(&key))
        else {
            return;
        };
        let (target_module, target) = match &symbol.origin {
            FeaSymbolOrigin::Local => (symbol.module.clone(), symbol.key.clone()),
            FeaSymbolOrigin::Import {
                target_module,
                target,
            } => (target_module.clone(), target.clone()),
            FeaSymbolOrigin::Function => {
                let Some(facts) = self
                    .functions
                    .get(&(self.module.clone(), symbol.key.name.clone()))
                else {
                    return;
                };
                (
                    facts.module_path.clone(),
                    FeaSymbolKey {
                        kind: FeaSymbolKind::Function,
                        name: facts.symbol.as_str().to_string(),
                    },
                )
            }
        };
        self.uses.push(BoundFeaSymbolUse {
            module: self.module.clone(),
            span: name.span,
            target_module,
            target,
        });
    }
}

fn symbol_kind(namespace: &str) -> Option<FeaSymbolKind> {
    [
        FeaSymbolKind::Geometry,
        FeaSymbolKind::Mesh,
        FeaSymbolKind::Material,
        FeaSymbolKind::Model,
        FeaSymbolKind::Study,
        FeaSymbolKind::Sweep,
        FeaSymbolKind::Execution,
        FeaSymbolKind::Table,
        FeaSymbolKind::Function,
    ]
    .into_iter()
    .find(|kind| kind.namespace() == namespace)
}
