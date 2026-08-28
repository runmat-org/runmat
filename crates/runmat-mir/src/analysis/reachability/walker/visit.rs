use runmat_hir::{CallableIdentity, WorkspaceEffect};

use crate::{
    MirCallArg, MirCallee, MirIndexComponent, MirIndexing, MirOperand, MirPlace, MirRvalue,
    MirStmt, MirStmtKind, MirTerminatorKind,
};

use super::{Certainty, Kind, Reason, Walker};

impl Walker<'_> {
    fn operand(&mut self, from: &str, operand: &MirOperand, reason: Reason) {
        if let MirOperand::FunctionHandle(identity) = operand {
            self.identity(from, identity, reason);
        }
    }

    fn call_arg(&mut self, from: &str, argument: &MirCallArg) {
        match argument {
            MirCallArg::Single(operand) => self.operand(from, operand, Reason::FunctionHandle),
            MirCallArg::Expansion { base, indices, .. } => {
                self.operand(from, base, Reason::FunctionHandle);
                for index in indices {
                    self.operand(from, index, Reason::FunctionHandle);
                }
            }
        }
    }

    fn indexing(&mut self, from: &str, indexing: &MirIndexing) {
        for component in &indexing.components {
            if let MirIndexComponent::Expr(operand) = component {
                self.operand(from, operand, Reason::FunctionHandle);
            }
        }
    }

    fn place(&mut self, from: &str, place: &MirPlace) {
        match place {
            MirPlace::Member(base, _) => self.place(from, base),
            MirPlace::DynamicMember(base, member) => {
                self.place(from, base);
                self.operand(from, member, Reason::FunctionHandle);
            }
            MirPlace::Index(base, indexing) => {
                self.place(from, base);
                self.indexing(from, indexing);
            }
            MirPlace::Local(_) | MirPlace::Binding(_) => {}
        }
    }

    fn rvalue(&mut self, from: &str, value: &MirRvalue) {
        match value {
            MirRvalue::Use(value) | MirRvalue::Spawn(value) => {
                self.operand(from, value, Reason::FunctionHandle);
            }
            MirRvalue::Unary(operator, value) => {
                self.operand(from, value, Reason::FunctionHandle);
                self.operator(from, *operator);
            }
            MirRvalue::Binary(left, operator, right) => {
                self.operand(from, left, Reason::FunctionHandle);
                self.operand(from, right, Reason::FunctionHandle);
                self.operator(from, *operator);
            }
            MirRvalue::ShortCircuit {
                left,
                right_temps,
                right,
                ..
            } => {
                self.operand(from, left, Reason::FunctionHandle);
                for statement in right_temps {
                    self.statement(from, statement);
                }
                self.operand(from, right, Reason::FunctionHandle);
            }
            MirRvalue::Range { start, step, end } => {
                self.operand(from, start, Reason::FunctionHandle);
                if let Some(step) = step {
                    self.operand(from, step, Reason::FunctionHandle);
                }
                self.operand(from, end, Reason::FunctionHandle);
            }
            MirRvalue::Call(call) => {
                let dynamic_target_builtin = match &call.callee {
                    MirCallee::Static(CallableIdentity::Builtin(id))
                        if id.0.eq_ignore_ascii_case("feval")
                            || id.0.eq_ignore_ascii_case("str2func") =>
                    {
                        Some(id.0.as_str())
                    }
                    _ => None,
                };
                match &call.callee {
                    MirCallee::Static(identity) => {
                        self.identity(from, identity, Reason::DirectCall);
                    }
                    MirCallee::Dynamic(operand) => match operand {
                        MirOperand::FunctionHandle(identity) => {
                            self.identity(from, identity, Reason::FunctionHandle);
                        }
                        MirOperand::Constant(crate::MirConstant::String(value)) => {
                            self.resolve_named_dynamic(
                                from,
                                &value.runtime_text(),
                                "constant string dynamic call",
                            );
                        }
                        MirOperand::Constant(crate::MirConstant::Symbol(value)) => {
                            self.resolve_named_dynamic(
                                from,
                                &value.0,
                                "constant symbol dynamic call",
                            );
                        }
                        MirOperand::Local(_) | MirOperand::Constant(_) => {
                            self.unknown_dynamic(from, "runtime call target is not finite");
                        }
                    },
                    MirCallee::SuperConstructor { super_class, .. } => {
                        self.class(from, super_class, Reason::SuperDispatch);
                    }
                    MirCallee::SuperMethod {
                        super_class,
                        method,
                        ..
                    } => self.named_dynamic(
                        from,
                        Kind::Method,
                        format!("{super_class}.{method}"),
                        "super method dispatch",
                    ),
                }
                if let Some(name) = dynamic_target_builtin {
                    if let Some(target) = call.args.first() {
                        self.dynamic_target(from, target.operand(), &format!("{name} target"));
                    } else {
                        self.unknown_dynamic(from, &format!("{name} target is absent"));
                    }
                }
                for argument in &call.args {
                    self.call_arg(from, argument);
                }
            }
            MirRvalue::Aggregate { elements, .. } => {
                for element in elements {
                    self.operand(from, element, Reason::FunctionHandle);
                }
            }
            MirRvalue::StructLiteral { fields } => {
                for (_, value) in fields {
                    self.operand(from, value, Reason::FunctionHandle);
                }
            }
            MirRvalue::ObjectLiteral { class_name, fields } => {
                self.class(
                    from,
                    &class_name
                        .display_name()
                        .unwrap_or_else(|| "<class>".into()),
                    Reason::ClassReference,
                );
                for (_, value) in fields {
                    self.operand(from, value, Reason::FunctionHandle);
                }
            }
            MirRvalue::Index { base, indexing } => {
                self.operand(from, base, Reason::FunctionHandle);
                self.indexing(from, indexing);
            }
            MirRvalue::Member { base, .. } => self.operand(from, base, Reason::FunctionHandle),
            MirRvalue::DynamicMember { base, member } => {
                self.operand(from, base, Reason::FunctionHandle);
                self.operand(from, member, Reason::FunctionHandle);
            }
            MirRvalue::WorkspaceFirstStaticProperty { class_name, .. } => {
                self.class(from, class_name, Reason::ClassReference);
            }
            MirRvalue::MetaClass(class_name) => self.class(
                from,
                &class_name
                    .display_name()
                    .unwrap_or_else(|| "<class>".into()),
                Reason::ClassReference,
            ),
            MirRvalue::Future { function, args, .. } => {
                self.retain_function(
                    *function,
                    Certainty::Definite,
                    Some(from.into()),
                    Reason::FutureCall,
                    None,
                );
                for argument in args {
                    self.call_arg(from, argument);
                }
            }
            MirRvalue::Distributed(operation) => {
                if let crate::parallel::MirDistributedOp::Create { input, .. } = operation {
                    self.operand(from, input, Reason::FunctionHandle);
                }
            }
            MirRvalue::Collective(operation) => match operation {
                crate::parallel::MirCollectiveOp::Broadcast { input, .. }
                | crate::parallel::MirCollectiveOp::Gather { input, .. }
                | crate::parallel::MirCollectiveOp::Scatter { input, .. }
                | crate::parallel::MirCollectiveOp::AllGather { input, .. }
                | crate::parallel::MirCollectiveOp::Send { input, .. } => {
                    self.operand(from, input, Reason::FunctionHandle);
                }
                crate::parallel::MirCollectiveOp::Reduce {
                    input, operator, ..
                }
                | crate::parallel::MirCollectiveOp::AllReduce {
                    input, operator, ..
                } => {
                    self.operand(from, input, Reason::FunctionHandle);
                    self.operator(from, *operator);
                }
                crate::parallel::MirCollectiveOp::Barrier { .. }
                | crate::parallel::MirCollectiveOp::Receive { .. } => {}
            },
            MirRvalue::Colon | MirRvalue::End => {}
        }
    }

    pub(super) fn statement(&mut self, from: &str, statement: &MirStmt) {
        match &statement.kind {
            MirStmtKind::Assign { place, value } => {
                self.place(from, place);
                self.rvalue(from, value);
            }
            MirStmtKind::MultiAssign { targets, value } => {
                for target in &targets.targets {
                    if let crate::MirOutputTarget::Place(place) = target {
                        self.place(from, place);
                    }
                }
                self.rvalue(from, value);
            }
            MirStmtKind::Expr(value) => self.rvalue(from, value),
            MirStmtKind::PlaceMutation(mutation) => self.place(from, &mutation.place),
            MirStmtKind::WorkspaceEffect { effect, bindings } => {
                let (kind, reason, family) = match effect {
                    WorkspaceEffect::MutatesGlobal => {
                        (Kind::GlobalState, Reason::WorkspaceGlobal, "global")
                    }
                    WorkspaceEffect::MutatesPersistent => (
                        Kind::PersistentState,
                        Reason::WorkspacePersistent,
                        "persistent",
                    ),
                    _ => return,
                };
                for local in bindings {
                    let binding = self
                        .current_function
                        .and_then(|function| self.assembly.bodies.get(&function))
                        .into_iter()
                        .flat_map(|body| &body.locals)
                        .find(|candidate| candidate.id == *local)
                        .and_then(|local| local.binding);
                    let symbol = binding
                        .and_then(|binding| self.names.bindings.get(&binding).cloned())
                        .unwrap_or_else(|| format!("local#{}", local.0));
                    let id = format!("{family}:{symbol}");
                    self.node(
                        id.clone(),
                        kind,
                        "workspace".into(),
                        symbol,
                        Certainty::Definite,
                    );
                    self.edge(Some(from.into()), id, Certainty::Definite, reason, None);
                }
            }
            MirStmtKind::EnvironmentEffect(_) => {}
        }
    }

    pub(super) fn terminator(&mut self, from: &str, terminator: &MirTerminatorKind) {
        match terminator {
            MirTerminatorKind::Branch { cond, .. } => {
                self.operand(from, cond, Reason::FunctionHandle);
            }
            MirTerminatorKind::Switch { discr, cases, .. } => {
                self.operand(from, discr, Reason::FunctionHandle);
                for (case, _) in cases {
                    self.operand(from, case, Reason::FunctionHandle);
                }
            }
            MirTerminatorKind::For { iterable, .. }
            | MirTerminatorKind::ParFor { iterable, .. } => self.rvalue(from, iterable),
            MirTerminatorKind::Spmd { header, .. } => match header.as_ref() {
                crate::parallel::MirSpmdHeader::Default => {}
                crate::parallel::MirSpmdHeader::One(a) => self.rvalue(from, a),
                crate::parallel::MirSpmdHeader::Two(a, b) => {
                    self.rvalue(from, a);
                    self.rvalue(from, b);
                }
                crate::parallel::MirSpmdHeader::Three(a, b, c) => {
                    self.rvalue(from, a);
                    self.rvalue(from, b);
                    self.rvalue(from, c);
                }
            },
            MirTerminatorKind::Return(values) => {
                for value in values {
                    self.operand(from, value, Reason::FunctionHandle);
                }
            }
            MirTerminatorKind::Await { future, result, .. } => {
                self.operand(from, future, Reason::FunctionHandle);
                if let Some(result) = result {
                    self.place(from, result);
                }
            }
            MirTerminatorKind::Goto(_)
            | MirTerminatorKind::TryCatch { .. }
            | MirTerminatorKind::Unreachable => {}
        }
        if let MirTerminatorKind::ParFor {
            maximum_workers: Some(maximum_workers),
            ..
        } = terminator
        {
            self.rvalue(from, maximum_workers);
        }
    }
}
