//! Immutable class-constructor facts owned by the verified Native IR product.

use runmat_mir::{MirCallee, MirExpressionStep, MirRvalue, MirStmtKind};
use runmat_types::{
    ClassIdentity, EventDeclaration, FunctionId, MemberAccess, MemberName, MethodAttributes,
    MethodName, PropertyAttributes,
};
use serde::{Deserialize, Serialize};

pub use runmat_mir::MirClassDefaultLiteral as NativeClassDefaultLiteral;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeConstructorMethod {
    pub function: FunctionId,
    pub access: MemberAccess,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeClassDeclaration {
    pub class: ClassIdentity,
    pub parent: Option<ClassIdentity>,
    pub is_sealed: bool,
    pub is_abstract: bool,
    pub properties: Vec<NativeClassProperty>,
    pub methods: Vec<NativeClassMethod>,
    pub events: Vec<EventDeclaration>,
    pub enumerations: Vec<MemberName>,
    pub constructor: Option<NativeConstructorMethod>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeClassProperty {
    pub name: MemberName,
    pub attributes: PropertyAttributes,
    pub has_default: bool,
    pub default_literal: Option<NativeClassDefaultLiteral>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeClassMethod {
    pub name: MethodName,
    pub function: FunctionId,
    pub is_static: bool,
    pub attributes: MethodAttributes,
}

impl super::NativeFunction {
    /// Typed superclass call identities embedded in this verified function.
    pub fn super_constructor_calls(&self) -> Vec<(ClassIdentity, ClassIdentity)> {
        let mut calls = Vec::new();
        for instruction in self.blocks.iter().flat_map(|block| &block.instructions) {
            match &instruction.operation {
                super::NativeOperation::Rvalue { value, .. } => visit_rvalue(value, &mut calls),
                super::NativeOperation::Statement(statement) => {
                    visit_statement(statement, &mut calls)
                }
            }
        }
        calls.sort();
        calls.dedup();
        calls
    }
}

fn visit_statement(statement: &MirStmtKind, calls: &mut Vec<(ClassIdentity, ClassIdentity)>) {
    match statement {
        MirStmtKind::PrepareFixedAssignment(preparation) => {
            for step in preparation.steps() {
                if let MirExpressionStep::Let { value, .. } = step {
                    visit_rvalue(value, calls);
                }
            }
        }
        MirStmtKind::PreparedAssign { value, .. }
        | MirStmtKind::Assign { value, .. }
        | MirStmtKind::MultiAssign { value, .. }
        | MirStmtKind::SequenceAssign { value, .. }
        | MirStmtKind::Expr(value) => visit_rvalue(value, calls),
        MirStmtKind::CaptureSequence { .. }
        | MirStmtKind::StatementOutput(_)
        | MirStmtKind::PlaceMutation(_)
        | MirStmtKind::WorkspaceEffect { .. }
        | MirStmtKind::EnvironmentEffect(_) => {}
    }
    statement.visit_expression_regions(|region| {
        for step in region.steps() {
            if let MirExpressionStep::Let { value, .. } = step {
                visit_rvalue(value, calls);
            }
        }
    });
}

fn visit_rvalue(value: &MirRvalue, calls: &mut Vec<(ClassIdentity, ClassIdentity)>) {
    if let MirRvalue::Call(call) = value {
        if let MirCallee::SuperConstructor {
            current_class,
            super_class,
        } = &call.callee
        {
            calls.push((current_class.clone(), super_class.clone()));
        }
    }
    if let MirRvalue::ShortCircuit { right_temps, .. } = value {
        for statement in right_temps {
            visit_statement(&statement.kind, calls);
        }
    }
    value.visit_expression_regions(|region| {
        for step in region.steps() {
            if let MirExpressionStep::Let { value, .. } = step {
                visit_rvalue(value, calls);
            }
        }
    });
}
