use crate::inference::function_outputs::infer_function_output_types;
use crate::inference::function_vars::infer_function_variable_types;
use crate::inference::globals::infer_global_variable_types;
use crate::validation::classdefs::validate_classdefs;
use crate::{
    BindingId, BindingName, BindingOwner, BindingRole, BindingStorage, BuiltinId, CallSyntax,
    CapturedBinding, ClassArgumentBlock, ClassEnumeration, ClassEvent, ClassId, ClassKind,
    ClassMethod, ClassProperty, EntrypointId, EntrypointName, EntrypointOrigin, EntrypointPolicy,
    ExprId, FunctionAbi, FunctionId, FunctionKind, FunctionModifiers, FunctionName, HirAssembly,
    HirBinding, HirBlock, HirCall, HirCallableRef, HirClass, HirEntrypoint, HirExpr, HirExprKind,
    HirFunction, HirImport, HirModule, HirPlace, HirStmt as SemanticHirStmt, HirStmtKind,
    IndexComponent, IndexKind, IndexResultContext, IndexingSemantics,
    LegacyHirProgram as HirProgram, LegacyHirStmt as HirStmt, LoopIterationSemantics,
    LoweringContext, LoweringResult, ModuleId, OperatorKind, QualifiedName, RequestedOutputCount,
    SemanticError, SourceId, SourceUnitKind, Span, StmtId, StringLiteral, SymbolName, Type, VarId,
    WorkspaceExportPolicy, WorkspaceVisibility,
};
use runmat_parser::{BinOp, Expr as AstExpr, Program as AstProgram, Stmt as AstStmt, UnOp};
use std::collections::HashMap;

pub(crate) struct Scope {
    pub(crate) parent: Option<usize>,
    pub(crate) bindings: HashMap<String, VarId>,
}

pub(crate) struct Ctx {
    pub(crate) scopes: Vec<Scope>,
    pub(crate) var_types: Vec<Type>,
    pub(crate) next_var: usize,
    pub(crate) functions: HashMap<String, HirStmt>,
    pub(crate) var_names: Vec<Option<String>>,
    pub(crate) allow_unqualified_statics: bool,
}

#[derive(Clone)]
struct SemanticScope {
    owner: FunctionId,
    bindings: HashMap<String, BindingId>,
    workspace_visibility: WorkspaceVisibility,
}

struct SemanticCtx {
    assembly: HirAssembly,
    module: ModuleId,
    next_expr: usize,
    next_stmt: usize,
    next_function: usize,
    scopes: Vec<SemanticScope>,
    function_names: HashMap<String, FunctionId>,
    captures: HashMap<FunctionId, Vec<CapturedBinding>>,
}

pub fn lower(
    prog: &AstProgram,
    context: &LoweringContext<'_>,
) -> Result<LoweringResult, SemanticError> {
    let mut ctx = Ctx::new();

    for (name, var_id) in context.variables {
        ctx.scopes[0].bindings.insert(name.clone(), VarId(*var_id));
        while ctx.var_types.len() <= *var_id {
            ctx.var_types.push(Type::Unknown);
        }
        while ctx.var_names.len() <= *var_id {
            ctx.var_names.push(None);
        }
        ctx.var_names[*var_id] = Some(name.clone());
        if *var_id >= ctx.next_var {
            ctx.next_var = var_id + 1;
        }
    }

    for (name, func_stmt) in context.functions {
        ctx.functions.insert(name.clone(), func_stmt.clone());
    }

    let body = ctx.lower_stmts(&prog.body)?;
    let var_types = ctx.var_types.clone();
    let hir = HirProgram { body, var_types };
    validate_classdefs(&hir)?;

    let mut variables: HashMap<String, usize> = HashMap::new();
    for (name, var_id) in &ctx.scopes[0].bindings {
        variables.insert(name.clone(), var_id.0);
    }
    let mut var_names = HashMap::new();
    for (idx, name_opt) in ctx.var_names.iter().enumerate() {
        if let Some(name) = name_opt {
            var_names.insert(VarId(idx), name.clone());
        }
    }

    let inferred_function_returns = infer_function_output_types(&hir);
    let inferred_function_envs = infer_function_variable_types(&hir);
    let inferred_globals = infer_global_variable_types(&hir, &inferred_function_returns);

    let assembly = SemanticCtx::lower_program(prog)?;

    Ok(LoweringResult {
        assembly,
        hir,
        variables,
        functions: ctx.functions,
        var_types: ctx.var_types,
        var_names,
        inferred_globals,
        inferred_function_envs,
        inferred_function_returns,
    })
}

impl SemanticCtx {
    fn lower_program(prog: &AstProgram) -> Result<HirAssembly, SemanticError> {
        let mut ctx = Self {
            assembly: HirAssembly::default(),
            module: ModuleId(0),
            next_expr: 0,
            next_stmt: 0,
            next_function: 0,
            scopes: Vec::new(),
            function_names: HashMap::new(),
            captures: HashMap::new(),
        };

        ctx.assembly.modules.push(HirModule {
            id: ctx.module,
            name: QualifiedName(vec![SymbolName("main".into())]),
            source_id: SourceId(0),
            source_unit: SourceUnitKind::ScriptFile,
            imports: Vec::new(),
            top_level_functions: Vec::new(),
            classes: Vec::new(),
            synthetic_entry_function: None,
        });

        for stmt in &prog.body {
            if let AstStmt::Function { name, .. } = stmt {
                let id = ctx.reserve_function_name(name);
                ctx.assembly.modules[ctx.module.0]
                    .top_level_functions
                    .push(id);
            }
        }

        for stmt in &prog.body {
            if let AstStmt::Import { path, wildcard, .. } = stmt {
                ctx.assembly.modules[ctx.module.0].imports.push(HirImport {
                    path: qualified_name(path),
                    wildcard: *wildcard,
                    span: stmt.span(),
                });
            }
        }

        let executable: Vec<_> = prog
            .body
            .iter()
            .filter(|stmt| !matches!(stmt, AstStmt::Function { .. } | AstStmt::ClassDef { .. }))
            .collect();
        if !executable.is_empty() {
            let entry_function = ctx.take_function_id();
            ctx.assembly.modules[ctx.module.0].synthetic_entry_function = Some(entry_function);
            ctx.assembly.entrypoints.push(HirEntrypoint {
                id: EntrypointId(ctx.assembly.entrypoints.len()),
                name: Some(EntrypointName("main".into())),
                target: entry_function,
                origin: EntrypointOrigin::SourcePath,
                policy: EntrypointPolicy {
                    workspace_export: WorkspaceExportPolicy::ExportTopLevelBindings,
                    top_level_await: true,
                },
            });
            let body = ctx.with_scope(entry_function, WorkspaceVisibility::TopLevel, |ctx| {
                ctx.lower_stmt_refs(&executable)
            })?;
            let locals = ctx.binding_ids_for_owner(entry_function);
            ctx.assembly.functions.push(HirFunction {
                id: entry_function,
                module: ctx.module,
                parent: None,
                enclosing_class: None,
                name: FunctionName("main".into()),
                kind: FunctionKind::SyntheticEntrypoint,
                params: vec![],
                outputs: vec![],
                abi: FunctionAbi::empty(),
                locals,
                captures: vec![],
                modifiers: FunctionModifiers::default(),
                body,
                span: prog
                    .body
                    .first()
                    .map(AstStmt::span)
                    .unwrap_or(Span { start: 0, end: 0 }),
            });
        }

        for stmt in &prog.body {
            match stmt {
                AstStmt::Function {
                    name,
                    params,
                    outputs,
                    body,
                    span,
                } => {
                    let id = ctx.function_names[name];
                    let function = ctx.lower_function(
                        id,
                        name,
                        params,
                        outputs,
                        body,
                        *span,
                        FunctionKind::Named,
                        None,
                        None,
                    )?;
                    ctx.assembly.functions.push(function);
                }
                AstStmt::ClassDef {
                    name,
                    super_class,
                    members,
                    span,
                } => {
                    let class = ctx.lower_class(name, super_class.as_deref(), members, *span)?;
                    ctx.assembly.classes.push(class);
                }
                _ => {}
            }
        }

        Ok(ctx.assembly)
    }

    fn reserve_function_name(&mut self, name: &str) -> FunctionId {
        if let Some(id) = self.function_names.get(name) {
            return *id;
        }
        let id = self.take_function_id();
        self.function_names.insert(name.to_string(), id);
        id
    }

    fn take_function_id(&mut self) -> FunctionId {
        let id = FunctionId(self.next_function);
        self.next_function += 1;
        id
    }

    fn alloc_binding_id(&self) -> BindingId {
        BindingId(self.assembly.bindings.len())
    }

    fn alloc_expr_id(&mut self) -> ExprId {
        let id = ExprId(self.next_expr);
        self.next_expr += 1;
        id
    }

    fn alloc_stmt_id(&mut self) -> StmtId {
        let id = StmtId(self.next_stmt);
        self.next_stmt += 1;
        id
    }

    fn with_scope<T>(
        &mut self,
        owner: FunctionId,
        workspace_visibility: WorkspaceVisibility,
        f: impl FnOnce(&mut Self) -> Result<T, SemanticError>,
    ) -> Result<T, SemanticError> {
        self.scopes.push(SemanticScope {
            owner,
            bindings: HashMap::new(),
            workspace_visibility,
        });
        let result = f(self);
        self.scopes.pop();
        result
    }

    fn current_scope(&self) -> &SemanticScope {
        self.scopes.last().expect("semantic lowering scope")
    }

    fn current_scope_mut(&mut self) -> &mut SemanticScope {
        self.scopes.last_mut().expect("semantic lowering scope")
    }

    fn define_binding(
        &mut self,
        name: &str,
        role: BindingRole,
        storage: BindingStorage,
        span: Span,
    ) -> BindingId {
        if let Some(id) = self.current_scope().bindings.get(name) {
            return *id;
        }
        let id = self.alloc_binding_id();
        let owner = self.current_scope().owner;
        let workspace_visibility = self.current_scope().workspace_visibility.clone();
        self.current_scope_mut()
            .bindings
            .insert(name.to_string(), id);
        self.assembly.bindings.push(HirBinding {
            id,
            owner: BindingOwner::Function(owner),
            name: BindingName(name.to_string()),
            role,
            storage,
            workspace_visibility,
            declared_span: span,
        });
        id
    }

    fn binding_for_read(&mut self, name: &str) -> Option<BindingId> {
        let binding = self.lookup_binding(name)?;
        self.record_capture_if_outer(binding);
        Some(binding)
    }

    fn binding_for_write(&mut self, name: &str, span: Span) -> BindingId {
        if let Some(binding) = self.lookup_binding(name) {
            self.record_capture_if_outer(binding);
            binding
        } else {
            self.define_binding(name, BindingRole::Local, BindingStorage::Lexical, span)
        }
    }

    fn record_capture_if_outer(&mut self, binding: BindingId) {
        let current = self.current_scope().owner;
        let owner = self
            .assembly
            .bindings
            .iter()
            .find(|candidate| candidate.id == binding)
            .and_then(|candidate| match candidate.owner {
                BindingOwner::Function(owner) => Some(owner),
                _ => None,
            });
        if let Some(owner) = owner {
            if owner != current {
                let captures = self.captures.entry(current).or_default();
                if !captures
                    .iter()
                    .any(|capture| capture.binding == binding && capture.from_function == owner)
                {
                    captures.push(CapturedBinding {
                        binding,
                        from_function: owner,
                    });
                }
            }
        }
    }

    fn lookup_binding(&self, name: &str) -> Option<BindingId> {
        self.scopes
            .iter()
            .rev()
            .find_map(|scope| scope.bindings.get(name).copied())
    }

    fn binding_ids_for_owner(&self, owner: FunctionId) -> Vec<BindingId> {
        self.assembly
            .bindings
            .iter()
            .filter_map(|binding| match binding.owner {
                BindingOwner::Function(id) if id == owner => Some(binding.id),
                _ => None,
            })
            .collect()
    }

    fn lower_function(
        &mut self,
        id: FunctionId,
        name: &str,
        params: &[String],
        outputs: &[String],
        body: &[AstStmt],
        span: Span,
        kind: FunctionKind,
        parent: Option<FunctionId>,
        enclosing_class: Option<ClassId>,
    ) -> Result<HirFunction, SemanticError> {
        self.with_scope(id, WorkspaceVisibility::Hidden, |ctx| {
            for stmt in body {
                if let AstStmt::Function { name, .. } = stmt {
                    ctx.reserve_function_name(name);
                }
            }
            let mut param_ids = Vec::new();
            for param in params {
                param_ids.push(ctx.define_binding(
                    param,
                    BindingRole::Parameter,
                    BindingStorage::Lexical,
                    span,
                ));
            }
            let mut output_ids = Vec::new();
            for output in outputs {
                output_ids.push(ctx.lookup_binding(output).unwrap_or_else(|| {
                    ctx.define_binding(output, BindingRole::Output, BindingStorage::Lexical, span)
                }));
            }
            let implicit_nargin = Some(ctx.define_binding(
                "nargin",
                BindingRole::Local,
                BindingStorage::Lexical,
                span,
            ));
            let implicit_nargout = Some(ctx.define_binding(
                "nargout",
                BindingRole::Local,
                BindingStorage::Lexical,
                span,
            ));
            let hir_body = ctx.lower_stmts_semantic(body)?;
            for stmt in body {
                if let AstStmt::Function {
                    name,
                    params,
                    outputs,
                    body,
                    span,
                } = stmt
                {
                    let nested_id = ctx.function_names[name];
                    let nested = ctx.lower_function(
                        nested_id,
                        name,
                        params,
                        outputs,
                        body,
                        *span,
                        FunctionKind::Named,
                        Some(id),
                        enclosing_class,
                    )?;
                    ctx.assembly.functions.push(nested);
                }
            }
            let locals = ctx.binding_ids_for_owner(id);
            let captures = ctx.captures.remove(&id).unwrap_or_default();
            Ok(HirFunction {
                id,
                module: ctx.module,
                parent,
                enclosing_class,
                name: FunctionName(name.to_string()),
                kind,
                params: param_ids.clone(),
                outputs: output_ids.clone(),
                abi: FunctionAbi {
                    fixed_inputs: param_ids,
                    varargin: params
                        .last()
                        .filter(|p| p.as_str() == "varargin")
                        .and_then(|p| ctx.lookup_binding(p)),
                    fixed_outputs: output_ids,
                    varargout: outputs
                        .last()
                        .filter(|p| p.as_str() == "varargout")
                        .and_then(|p| ctx.lookup_binding(p)),
                    implicit_nargin,
                    implicit_nargout,
                },
                locals,
                captures,
                modifiers: FunctionModifiers::default(),
                body: hir_body,
                span,
            })
        })
    }

    fn lower_class(
        &mut self,
        name: &str,
        super_class: Option<&str>,
        members: &[runmat_parser::ClassMember],
        span: Span,
    ) -> Result<HirClass, SemanticError> {
        let class_id = ClassId(self.assembly.classes.len());
        self.assembly.modules[self.module.0].classes.push(class_id);
        let mut properties = Vec::new();
        let mut methods = Vec::new();
        let mut events = Vec::new();
        let mut enumerations = Vec::new();
        let mut arguments = Vec::new();

        for member in members {
            match member {
                runmat_parser::ClassMember::Properties { names, .. } => {
                    properties.extend(names.iter().map(|name| ClassProperty {
                        name: crate::MemberName(name.clone()),
                        attributes: crate::PropertyAttributes::default(),
                        default: None,
                        span,
                    }));
                }
                runmat_parser::ClassMember::Methods { attributes, body } => {
                    let is_static = attributes
                        .iter()
                        .any(|attr| attr.name.eq_ignore_ascii_case("Static"));
                    for stmt in body {
                        if let AstStmt::Function {
                            name,
                            params,
                            outputs,
                            body,
                            span,
                        } = stmt
                        {
                            let function_id = self.take_function_id();
                            let function = self.lower_function(
                                function_id,
                                name,
                                params,
                                outputs,
                                body,
                                *span,
                                FunctionKind::ClassMethod { is_static },
                                None,
                                Some(class_id),
                            )?;
                            self.assembly.functions.push(function);
                            methods.push(ClassMethod {
                                function: function_id,
                                name: crate::MethodName(name.clone()),
                                is_static,
                                span: *span,
                            });
                        }
                    }
                }
                runmat_parser::ClassMember::Events { names, .. } => {
                    events.extend(names.iter().map(|name| ClassEvent {
                        name: SymbolName(name.clone()),
                        span,
                    }));
                }
                runmat_parser::ClassMember::Enumeration { names, .. } => {
                    enumerations.extend(names.iter().map(|name| ClassEnumeration {
                        name: SymbolName(name.clone()),
                        span,
                    }));
                }
                runmat_parser::ClassMember::Arguments { .. } => {
                    arguments.push(ClassArgumentBlock { span });
                }
            }
        }

        Ok(HirClass {
            id: class_id,
            module: self.module,
            name: QualifiedName(vec![SymbolName(name.to_string())]),
            super_class: None,
            kind: if super_class
                .map(|name| name.eq_ignore_ascii_case("handle"))
                .unwrap_or(false)
            {
                ClassKind::Handle
            } else {
                ClassKind::Value
            },
            properties,
            methods,
            events,
            enumerations,
            arguments,
            span,
        })
    }

    fn lower_stmt_refs(&mut self, stmts: &[&AstStmt]) -> Result<HirBlock, SemanticError> {
        let mut statements = Vec::new();
        for stmt in stmts {
            if let Some(stmt) = self.lower_stmt_semantic(stmt)? {
                statements.push(stmt);
            }
        }
        Ok(HirBlock { statements })
    }

    fn lower_stmts_semantic(&mut self, stmts: &[AstStmt]) -> Result<HirBlock, SemanticError> {
        let refs: Vec<_> = stmts.iter().collect();
        self.lower_stmt_refs(&refs)
    }

    fn lower_stmt_semantic(
        &mut self,
        stmt: &AstStmt,
    ) -> Result<Option<SemanticHirStmt>, SemanticError> {
        let span = stmt.span();
        let kind = match stmt {
            AstStmt::ExprStmt(expr, suppressed, _) => HirStmtKind::ExprStmt(
                self.lower_expr_semantic_requested(expr, RequestedOutputCount::Zero)?,
                *suppressed,
            ),
            AstStmt::Assign(name, expr, suppressed, _) => {
                let binding = self.binding_for_write(name, span);
                HirStmtKind::Assign(
                    HirPlace::Binding(binding),
                    self.lower_expr_semantic_requested(expr, RequestedOutputCount::One)?,
                    *suppressed,
                )
            }
            AstStmt::MultiAssign(names, expr, suppressed, _) => {
                let targets = names
                    .iter()
                    .map(|name| {
                        if name == "~" {
                            crate::OutputTarget::Discard
                        } else {
                            let binding = self.binding_for_write(name, span);
                            crate::OutputTarget::Place(HirPlace::Binding(binding))
                        }
                    })
                    .collect();
                HirStmtKind::MultiAssign(
                    crate::OutputTargetList { targets },
                    self.lower_expr_semantic_requested(
                        expr,
                        RequestedOutputCount::Exactly(names.len()),
                    )?,
                    *suppressed,
                )
            }
            AstStmt::If {
                cond,
                then_body,
                elseif_blocks,
                else_body,
                ..
            } => HirStmtKind::If {
                cond: self.lower_expr_semantic(cond)?,
                then_body: self.lower_stmts_semantic(then_body)?,
                elseif_blocks: elseif_blocks
                    .iter()
                    .map(|(cond, body)| {
                        Ok((
                            self.lower_expr_semantic(cond)?,
                            self.lower_stmts_semantic(body)?,
                        ))
                    })
                    .collect::<Result<_, SemanticError>>()?,
                else_body: else_body
                    .as_ref()
                    .map(|body| self.lower_stmts_semantic(body))
                    .transpose()?,
            },
            AstStmt::While { cond, body, .. } => HirStmtKind::While {
                cond: self.lower_expr_semantic(cond)?,
                body: self.lower_stmts_semantic(body)?,
            },
            AstStmt::For {
                var, expr, body, ..
            } => {
                let binding =
                    self.define_binding(var, BindingRole::Local, BindingStorage::Lexical, span);
                HirStmtKind::For {
                    binding,
                    range: self.lower_expr_semantic(expr)?,
                    body: self.lower_stmts_semantic(body)?,
                    semantics: LoopIterationSemantics::ForColumns,
                }
            }
            AstStmt::Global(names, _) => {
                let ids = names
                    .iter()
                    .map(|name| {
                        self.define_binding(name, BindingRole::Local, BindingStorage::Global, span)
                    })
                    .collect();
                HirStmtKind::Global(ids)
            }
            AstStmt::Persistent(names, _) => {
                let ids = names
                    .iter()
                    .map(|name| {
                        self.define_binding(
                            name,
                            BindingRole::Local,
                            BindingStorage::Persistent,
                            span,
                        )
                    })
                    .collect();
                HirStmtKind::Persistent(ids)
            }
            AstStmt::Break(_) => HirStmtKind::Break,
            AstStmt::Continue(_) => HirStmtKind::Continue,
            AstStmt::Return(_) => HirStmtKind::Return,
            AstStmt::Import { path, wildcard, .. } => HirStmtKind::Import(HirImport {
                path: qualified_name(path),
                wildcard: *wildcard,
                span,
            }),
            AstStmt::Function { .. } | AstStmt::ClassDef { .. } => return Ok(None),
            AstStmt::Switch {
                expr,
                cases,
                otherwise,
                ..
            } => HirStmtKind::Switch {
                expr: self.lower_expr_semantic(expr)?,
                cases: cases
                    .iter()
                    .map(|(expr, body)| {
                        Ok((
                            self.lower_expr_semantic(expr)?,
                            self.lower_stmts_semantic(body)?,
                        ))
                    })
                    .collect::<Result<_, SemanticError>>()?,
                otherwise: otherwise
                    .as_ref()
                    .map(|body| self.lower_stmts_semantic(body))
                    .transpose()?,
            },
            AstStmt::TryCatch {
                try_body,
                catch_var,
                catch_body,
                ..
            } => {
                let catch_binding = catch_var.as_ref().map(|name| {
                    self.define_binding(name, BindingRole::Local, BindingStorage::Lexical, span)
                });
                HirStmtKind::TryCatch {
                    try_body: self.lower_stmts_semantic(try_body)?,
                    catch_binding,
                    catch_body: self.lower_stmts_semantic(catch_body)?,
                }
            }
            AstStmt::AssignLValue(lvalue, expr, suppressed, _) => HirStmtKind::Assign(
                self.lower_lvalue_semantic(lvalue, span)?,
                self.lower_expr_semantic(expr)?,
                *suppressed,
            ),
        };
        Ok(Some(SemanticHirStmt {
            id: self.alloc_stmt_id(),
            kind,
            span,
        }))
    }

    fn lower_lvalue_semantic(
        &mut self,
        lvalue: &runmat_parser::LValue,
        span: Span,
    ) -> Result<HirPlace, SemanticError> {
        use runmat_parser::LValue;
        Ok(match lvalue {
            LValue::Var(name) => HirPlace::Binding(self.binding_for_write(name, span)),
            LValue::Member(base, name) => HirPlace::Member(
                Box::new(self.lower_expr_semantic(base)?),
                crate::MemberName(name.clone()),
            ),
            LValue::MemberDynamic(base, name) => HirPlace::MemberDynamic(
                Box::new(self.lower_expr_semantic(base)?),
                Box::new(self.lower_expr_semantic(name)?),
            ),
            LValue::Index(base, indices) => HirPlace::Index(
                Box::new(self.lower_expr_semantic(base)?),
                self.lower_indexing(indices, IndexKind::Paren)?,
            ),
            LValue::IndexCell(base, indices) => HirPlace::IndexCell(
                Box::new(self.lower_expr_semantic(base)?),
                self.lower_indexing(indices, IndexKind::Brace)?,
            ),
        })
    }

    fn lower_expr_semantic(&mut self, expr: &AstExpr) -> Result<HirExpr, SemanticError> {
        self.lower_expr_semantic_requested(expr, RequestedOutputCount::One)
    }

    fn lower_expr_semantic_requested(
        &mut self,
        expr: &AstExpr,
        requested_outputs: RequestedOutputCount,
    ) -> Result<HirExpr, SemanticError> {
        let span = expr.span();
        let kind = match expr {
            AstExpr::Number(value, _) => HirExprKind::Number(value.clone()),
            AstExpr::String(value, _) => HirExprKind::String(StringLiteral(value.clone())),
            AstExpr::Ident(name, _) => self
                .binding_for_read(name)
                .map(HirExprKind::Binding)
                .unwrap_or_else(|| {
                    if runmat_builtins::constants()
                        .iter()
                        .any(|c| c.name == name.as_str())
                    {
                        HirExprKind::Constant(SymbolName(name.clone()))
                    } else {
                        HirExprKind::Call(self.call_for_name(
                            name,
                            Vec::new(),
                            CallSyntax::Plain,
                            requested_outputs,
                        ))
                    }
                }),
            AstExpr::FuncCall(name, args, _) => {
                let args: Vec<HirExpr> = args
                    .iter()
                    .map(|arg| self.lower_expr_semantic(arg))
                    .collect::<Result<_, _>>()?;
                if let Some(binding) = self.binding_for_read(name) {
                    let base = HirExpr {
                        id: self.alloc_expr_id(),
                        kind: HirExprKind::Binding(binding),
                        span,
                    };
                    HirExprKind::Index(
                        Box::new(base),
                        IndexingSemantics {
                            kind: IndexKind::Paren,
                            components: args.into_iter().map(IndexComponent::Expr).collect(),
                            result_context: IndexResultContext::ReadSingle,
                        },
                    )
                } else {
                    HirExprKind::Call(self.call_for_name(
                        name,
                        args,
                        CallSyntax::Plain,
                        requested_outputs,
                    ))
                }
            }
            AstExpr::FuncHandle(name, _) => {
                HirExprKind::FunctionHandle(if let Some(function) = self.function_names.get(name) {
                    crate::FunctionHandleTarget::Function(*function)
                } else if is_builtin(name) {
                    crate::FunctionHandleTarget::Builtin(BuiltinId(name.clone()))
                } else {
                    crate::FunctionHandleTarget::DynamicName(SymbolName(name.clone()))
                })
            }
            AstExpr::AnonFunc { params, body, span } => {
                let function_id = self.take_function_id();
                let function =
                    self.with_scope(function_id, WorkspaceVisibility::Hidden, |ctx| {
                        let param_ids = params
                            .iter()
                            .map(|param| {
                                ctx.define_binding(
                                    param,
                                    BindingRole::Parameter,
                                    BindingStorage::Lexical,
                                    *span,
                                )
                            })
                            .collect::<Vec<_>>();
                        let body_expr = ctx.lower_expr_semantic(body)?;
                        let stmt = SemanticHirStmt {
                            id: ctx.alloc_stmt_id(),
                            kind: HirStmtKind::ExprStmt(body_expr, false),
                            span: *span,
                        };
                        let locals = ctx.binding_ids_for_owner(function_id);
                        Ok(HirFunction {
                            id: function_id,
                            module: ctx.module,
                            parent: Some(ctx.scopes[ctx.scopes.len() - 2].owner),
                            enclosing_class: None,
                            name: FunctionName(format!("anonymous#{}", function_id.0)),
                            kind: FunctionKind::Anonymous,
                            params: param_ids.clone(),
                            outputs: vec![],
                            abi: FunctionAbi {
                                fixed_inputs: param_ids,
                                varargin: None,
                                fixed_outputs: vec![],
                                varargout: None,
                                implicit_nargin: None,
                                implicit_nargout: None,
                            },
                            locals,
                            captures: ctx.captures.remove(&function_id).unwrap_or_default(),
                            modifiers: FunctionModifiers::default(),
                            body: HirBlock {
                                statements: vec![stmt],
                            },
                            span: *span,
                        })
                    })?;
                self.assembly.functions.push(function);
                HirExprKind::AnonymousFunction(function_id)
            }
            AstExpr::Unary(op, inner, _) => {
                HirExprKind::Unary(unary_op(*op), Box::new(self.lower_expr_semantic(inner)?))
            }
            AstExpr::Binary(left, op, right, _) => HirExprKind::Binary(
                Box::new(self.lower_expr_semantic(left)?),
                binary_op(*op),
                Box::new(self.lower_expr_semantic(right)?),
            ),
            AstExpr::Tensor(rows, _) => HirExprKind::Tensor(
                rows.iter()
                    .map(|row| {
                        row.iter()
                            .map(|expr| self.lower_expr_semantic(expr))
                            .collect()
                    })
                    .collect::<Result<_, _>>()?,
            ),
            AstExpr::Cell(rows, _) => HirExprKind::Cell(
                rows.iter()
                    .map(|row| {
                        row.iter()
                            .map(|expr| self.lower_expr_semantic(expr))
                            .collect()
                    })
                    .collect::<Result<_, _>>()?,
            ),
            AstExpr::Index(base, indices, _) => HirExprKind::Index(
                Box::new(self.lower_expr_semantic(base)?),
                self.lower_indexing(indices, IndexKind::Paren)?,
            ),
            AstExpr::IndexCell(base, indices, _) => HirExprKind::Index(
                Box::new(self.lower_expr_semantic(base)?),
                self.lower_indexing(indices, IndexKind::Brace)?,
            ),
            AstExpr::Range(start, step, end, _) => HirExprKind::Range(
                Box::new(self.lower_expr_semantic(start)?),
                step.as_ref()
                    .map(|step| self.lower_expr_semantic(step).map(Box::new))
                    .transpose()?,
                Box::new(self.lower_expr_semantic(end)?),
            ),
            AstExpr::Colon(_) => HirExprKind::Colon,
            AstExpr::EndKeyword(_) => HirExprKind::End,
            AstExpr::Member(base, name, _) => HirExprKind::Member(
                Box::new(self.lower_expr_semantic(base)?),
                crate::MemberName(name.clone()),
            ),
            AstExpr::MemberDynamic(base, name, _) => HirExprKind::MemberDynamic(
                Box::new(self.lower_expr_semantic(base)?),
                Box::new(self.lower_expr_semantic(name)?),
            ),
            AstExpr::DottedInvoke(base, name, args, _)
            | AstExpr::MethodCall(base, name, args, _) => {
                let mut call_args = vec![self.lower_expr_semantic(base)?];
                call_args.extend(
                    args.iter()
                        .map(|arg| self.lower_expr_semantic(arg))
                        .collect::<Result<Vec<_>, _>>()?,
                );
                HirExprKind::Call(self.call_for_name(
                    name,
                    call_args,
                    CallSyntax::Method,
                    RequestedOutputCount::One,
                ))
            }
            AstExpr::MetaClass(name, _) => {
                HirExprKind::MetaClass(QualifiedName(vec![SymbolName(name.clone())]))
            }
        };
        Ok(HirExpr {
            id: self.alloc_expr_id(),
            kind,
            span,
        })
    }

    fn lower_indexing(
        &mut self,
        indices: &[AstExpr],
        kind: IndexKind,
    ) -> Result<IndexingSemantics, SemanticError> {
        Ok(IndexingSemantics {
            kind,
            components: indices
                .iter()
                .map(|expr| {
                    Ok(match expr {
                        AstExpr::Colon(_) => IndexComponent::Colon,
                        AstExpr::EndKeyword(_) => IndexComponent::End {
                            dim: None,
                            offset: 0,
                        },
                        _ => IndexComponent::Expr(self.lower_expr_semantic(expr)?),
                    })
                })
                .collect::<Result<_, SemanticError>>()?,
            result_context: IndexResultContext::ReadSingle,
        })
    }

    fn call_for_name(
        &self,
        name: &str,
        args: Vec<HirExpr>,
        syntax: CallSyntax,
        requested_outputs: RequestedOutputCount,
    ) -> HirCall {
        let callee = if let Some(function) = self.function_names.get(name) {
            HirCallableRef::Function(*function)
        } else if is_builtin(name) {
            HirCallableRef::Builtin(BuiltinId(name.to_string()))
        } else {
            HirCallableRef::Unresolved(QualifiedName(vec![SymbolName(name.to_string())]))
        };
        HirCall {
            callee,
            args,
            syntax,
            requested_outputs,
        }
    }
}

impl FunctionAbi {
    fn empty() -> Self {
        Self {
            fixed_inputs: vec![],
            varargin: None,
            fixed_outputs: vec![],
            varargout: None,
            implicit_nargin: None,
            implicit_nargout: None,
        }
    }
}

fn qualified_name(parts: &[String]) -> QualifiedName {
    QualifiedName(parts.iter().cloned().map(SymbolName).collect())
}

fn is_builtin(name: &str) -> bool {
    runmat_builtins::builtin_functions()
        .iter()
        .any(|builtin| builtin.name == name)
}

fn unary_op(op: UnOp) -> OperatorKind {
    match op {
        UnOp::Plus => OperatorKind::UnaryPlus,
        UnOp::Minus => OperatorKind::UnaryMinus,
        UnOp::Not => OperatorKind::Not,
        UnOp::Transpose => OperatorKind::ConjugateTranspose,
        UnOp::NonConjugateTranspose => OperatorKind::Transpose,
    }
}

fn binary_op(op: BinOp) -> OperatorKind {
    match op {
        BinOp::Add => OperatorKind::Add,
        BinOp::Sub => OperatorKind::Subtract,
        BinOp::Mul => OperatorKind::MatrixMultiply,
        BinOp::RightDiv => OperatorKind::Mrdivide,
        BinOp::Pow => OperatorKind::MatrixPower,
        BinOp::LeftDiv => OperatorKind::Mldivide,
        BinOp::Colon => OperatorKind::Add,
        BinOp::ElemMul => OperatorKind::ElementwiseMultiply,
        BinOp::ElemDiv => OperatorKind::ElementwiseDivide,
        BinOp::ElemPow => OperatorKind::ElementwisePower,
        BinOp::ElemLeftDiv => OperatorKind::ElementwiseLeftDivide,
        BinOp::AndAnd => OperatorKind::ShortCircuitAnd,
        BinOp::OrOr => OperatorKind::ShortCircuitOr,
        BinOp::BitAnd => OperatorKind::ElementwiseAnd,
        BinOp::BitOr => OperatorKind::ElementwiseOr,
        BinOp::Equal => OperatorKind::Equal,
        BinOp::NotEqual => OperatorKind::NotEqual,
        BinOp::Less => OperatorKind::Less,
        BinOp::LessEqual => OperatorKind::LessEqual,
        BinOp::Greater => OperatorKind::Greater,
        BinOp::GreaterEqual => OperatorKind::GreaterEqual,
    }
}

impl Ctx {
    pub(crate) fn new() -> Self {
        Self {
            scopes: vec![Scope {
                parent: None,
                bindings: HashMap::new(),
            }],
            var_types: Vec::new(),
            next_var: 0,
            functions: HashMap::new(),
            var_names: Vec::new(),
            allow_unqualified_statics: false,
        }
    }

    pub(crate) fn push_scope(&mut self) -> usize {
        let parent = Some(self.scopes.len() - 1);
        self.scopes.push(Scope {
            parent,
            bindings: HashMap::new(),
        });
        self.scopes.len() - 1
    }

    pub(crate) fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    pub(crate) fn define(&mut self, name: String) -> VarId {
        let id = VarId(self.next_var);
        self.next_var += 1;
        let current = self.scopes.len() - 1;
        self.scopes[current].bindings.insert(name.clone(), id);
        self.var_types.push(Type::Unknown);
        self.var_names.push(Some(name));
        id
    }

    pub(crate) fn lookup_current_scope(&self, name: &str) -> Option<VarId> {
        let current = self.scopes.len() - 1;
        self.scopes[current].bindings.get(name).copied()
    }

    pub(crate) fn lookup(&self, name: &str) -> Option<VarId> {
        let mut scope_idx = Some(self.scopes.len() - 1);
        while let Some(idx) = scope_idx {
            if let Some(id) = self.scopes[idx].bindings.get(name) {
                return Some(*id);
            }
            scope_idx = self.scopes[idx].parent;
        }
        None
    }

    pub(crate) fn is_constant(&self, name: &str) -> bool {
        runmat_builtins::constants().iter().any(|c| c.name == name)
    }

    pub(crate) fn is_builtin_function(&self, name: &str) -> bool {
        runmat_builtins::builtin_functions()
            .iter()
            .any(|b| b.name == name)
    }

    pub(crate) fn is_user_defined_function(&self, name: &str) -> bool {
        self.functions.contains_key(name)
    }

    pub(crate) fn is_function(&self, name: &str) -> bool {
        self.is_user_defined_function(name) || self.is_builtin_function(name)
    }

    pub(crate) fn is_static_method_class(&self, name: &str) -> bool {
        matches!(
            name,
            "gpuArray"
                | "logical"
                | "double"
                | "single"
                | "int8"
                | "int16"
                | "int32"
                | "int64"
                | "uint8"
                | "uint16"
                | "uint32"
                | "uint64"
                | "char"
                | "string"
                | "cell"
                | "struct"
        )
    }
}
