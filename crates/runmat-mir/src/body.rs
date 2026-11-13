use crate::{
    BasicBlock, BasicBlockId, CatchScopeId, MirLocalId, MirOutputTarget, MirPlace,
    MirStatementOutputLabel, MirStatementOutputSource, MirStmtKind,
};
use runmat_types::{BindingId, FunctionAbi, FunctionId, Span};
use serde::{Deserialize, Serialize};

#[path = "body/expression_region_validation/mod.rs"]
mod expression_region_validation;
#[path = "body/prepared_assignment_validation.rs"]
mod prepared_assignment_validation;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MirBody {
    pub function: FunctionId,
    pub abi: FunctionAbi,
    pub locals: Vec<MirLocal>,
    pub blocks: Vec<BasicBlock>,
    /// Lexical catch membership authored while lowering HIR catch bodies.
    /// The outer-to-inner scope order is authoritative for every backend.
    pub catch_scopes: std::collections::BTreeMap<BasicBlockId, Vec<CatchScopeId>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MirLocal {
    pub id: MirLocalId,
    pub binding: Option<BindingId>,
    pub kind: MirLocalKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MirLocalKind {
    Parameter,
    Output,
    Binding,
    IncomingWorkspace,
    /// A name made visible by a dynamic workspace effect. Its slot is resolved
    /// at execution time; unlike IncomingWorkspace, presence is not proven.
    ExternalWorkspace,
    Temporary,
    Capture,
}

impl MirBody {
    /// A display effect must consume the exact completed source statement, not
    /// a stale expression register or an unrelated local with the same name.
    pub fn validate_statement_outputs(&self) -> Result<(), String> {
        fn root(place: &MirPlace) -> Option<MirLocalId> {
            match place {
                MirPlace::Local(local) => Some(*local),
                MirPlace::Member(base, _)
                | MirPlace::DynamicMember(base, _)
                | MirPlace::Index(base, _) => root(base),
                MirPlace::Binding(_) => None,
            }
        }

        for block in &self.blocks {
            let mut assignment_roots = Vec::new();
            let mut assignment_span = None;
            let mut previous_expression = None;
            for statement in &block.statements {
                match &statement.kind {
                    MirStmtKind::PreparedAssign { preparation, .. } => {
                        assignment_roots = root(&preparation.mutation.place).into_iter().collect();
                        assignment_span = Some(statement.span);
                        previous_expression = None;
                    }
                    MirStmtKind::Assign { place, .. } => {
                        assignment_roots = root(place).into_iter().collect();
                        assignment_span = Some(statement.span);
                        previous_expression = None;
                    }
                    MirStmtKind::MultiAssign { targets, .. } => {
                        assignment_roots = targets
                            .targets
                            .iter()
                            .filter_map(|target| match target {
                                MirOutputTarget::Place(place) => root(place),
                                MirOutputTarget::Sequence(target) => root(target.base()),
                                MirOutputTarget::Discard => None,
                            })
                            .collect();
                        assignment_span = Some(statement.span);
                        previous_expression = None;
                    }
                    MirStmtKind::Expr(value) => {
                        assignment_roots.clear();
                        assignment_span = None;
                        let produces_value = value.has_statement_result();
                        let direct_binding = match value {
                            crate::MirRvalue::Use(crate::MirOperand::Local(local)) => self
                                .locals
                                .iter()
                                .find(|entry| entry.id == *local)
                                .and_then(|entry| entry.binding),
                            _ => None,
                        };
                        previous_expression =
                            produces_value.then_some((statement.span, direct_binding));
                    }
                    MirStmtKind::StatementOutput(output) => {
                        match output.source {
                            MirStatementOutputSource::Expression => {
                                let expected = match output.label {
                                    MirStatementOutputLabel::Ans => Some((statement.span, None)),
                                    MirStatementOutputLabel::Binding(binding) => {
                                        Some((statement.span, Some(binding)))
                                    }
                                };
                                if previous_expression != expected {
                                    return Err("expression output label differs from its exact expression source".into());
                                }
                                previous_expression = None;
                            }
                            MirStatementOutputSource::Local(local) => {
                                if assignment_span != Some(statement.span)
                                    || !assignment_roots.contains(&local)
                                {
                                    return Err("binding output is not attached to its exact assignment root".into());
                                }
                                let binding = self
                                    .locals
                                    .iter()
                                    .find(|entry| entry.id == local)
                                    .and_then(|entry| entry.binding);
                                if !matches!(output.label, MirStatementOutputLabel::Binding(label) if Some(label) == binding)
                                {
                                    return Err(
                                        "binding output label differs from its source local".into(),
                                    );
                                }
                                previous_expression = None;
                            }
                        }
                    }
                    _ => {
                        assignment_roots.clear();
                        assignment_span = None;
                        previous_expression = None;
                    }
                }
            }
        }
        Ok(())
    }

    /// Validates contextual expression regions against their owning body's
    /// local namespace. Region temporaries are private straight-line values:
    /// they may not be defined or observed by the surrounding control-flow
    /// graph.
    pub fn validate_expression_regions(&self) -> Result<(), String> {
        prepared_assignment_validation::validate(self)?;
        expression_region_validation::validate(self)
    }
}
