use crate::validation::classdefs::validate_classdefs;
use crate::{
    AssignmentCreationPolicy, AssignmentShapePolicy, BindingId, BindingName, BindingOwner,
    BindingResolution, BindingRole, BindingStorage, BuiltinId, CallKind, CallResolution,
    CallSyntax, CapturedBinding, ClassArgumentBlock, ClassEnumeration, ClassEvent, ClassId,
    ClassKind, ClassMethod, ClassProperty, ClassResolution, CommandArgument, EntrypointId,
    EntrypointName, EntrypointOrigin, EntrypointPolicy, ExprId, FunctionAbi, FunctionId,
    FunctionKind, FunctionModifiers, FunctionName, FunctionResolution, HirAssembly, HirBinding,
    HirBlock, HirCall, HirCallableRef, HirClass, HirCommandCall, HirEntrypoint, HirExpr,
    HirExprKind, HirFunction, HirImport, HirModule, HirPlace, HirStmt as SemanticHirStmt,
    HirStmtKind, ImportResolution, IndexComponent, IndexKind, IndexResultContext,
    IndexingSemantics, LegacyHirProgram as HirProgram, LegacyHirStmt as HirStmt,
    LoopIterationSemantics, LoweringContext, LoweringResult, ModuleId, OperatorKind, PlaceMutation,
    PlaceMutationKind, QualifiedName, ReferenceKind, ReferenceResolution, RequestedOutputCount,
    SemanticError, SemanticIndex, SourceId, SourceUnitKind, Span, StmtId, StringLiteral,
    SymbolName, Type, VarId, WorkspaceExportPolicy, WorkspaceVisibility, AWAIT_EXTENSION_NAME,
    DISCARD_OUTPUT_NAME, NARGIN_BUILTIN_NAME, NARGOUT_BUILTIN_NAME, SPAWN_EXTENSION_NAME,
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
    semantic_index: SemanticIndex,
    module: ModuleId,
    next_expr: usize,
    next_stmt: usize,
    next_function: usize,
    scopes: Vec<SemanticScope>,
    function_modifiers: Vec<FunctionModifiers>,
    top_level_await: Vec<bool>,
    compatibility_mode: Option<crate::CompatibilityMode>,
    function_names: HashMap<String, FunctionId>,
    function_input_signatures: HashMap<String, (usize, bool)>,
    function_output_signatures: HashMap<String, (usize, bool)>,
    external_function_names: HashMap<String, FunctionId>,
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

    let (mut assembly, semantic_index) = SemanticCtx::lower_program(prog, context)?;
    assembly.compatibility_mode = context.compatibility_mode.clone();

    Ok(LoweringResult {
        assembly,
        compatibility_mode: context.compatibility_mode.clone(),
        semantic_index,
        hir,
        variables,
        functions: ctx.functions,
        var_types: ctx.var_types,
        var_names,
        inferred_globals: HashMap::new(),
        inferred_function_envs: HashMap::new(),
        inferred_function_returns: HashMap::new(),
    })
}

impl SemanticCtx {
    fn lower_program(
        prog: &AstProgram,
        context: &LoweringContext<'_>,
    ) -> Result<(HirAssembly, SemanticIndex), SemanticError> {
        let mut ctx = Self {
            assembly: HirAssembly::default(),
            semantic_index: SemanticIndex::default(),
            module: ModuleId(0),
            next_expr: 0,
            next_stmt: 0,
            next_function: 0,
            scopes: Vec::new(),
            function_modifiers: Vec::new(),
            top_level_await: Vec::new(),
            compatibility_mode: context.compatibility_mode.clone(),
            function_names: HashMap::new(),
            function_input_signatures: HashMap::new(),
            function_output_signatures: HashMap::new(),
            external_function_names: context.semantic_functions.clone(),
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
            if let AstStmt::Function {
                name,
                params,
                outputs,
                ..
            } = stmt
            {
                let id = ctx.reserve_function_name(name);
                ctx.reserve_function_input_signature(name, params);
                ctx.reserve_function_output_signature(name, outputs);
                ctx.assembly.modules[ctx.module.0]
                    .top_level_functions
                    .push(id);
            }
        }

        for stmt in &prog.body {
            if let AstStmt::Import { path, wildcard, .. } = stmt {
                let import = HirImport {
                    path: qualified_name(path),
                    wildcard: *wildcard,
                    span: stmt.span(),
                };
                ctx.semantic_index.imports.push(ImportResolution {
                    import: import.clone(),
                });
                ctx.assembly.modules[ctx.module.0].imports.push(import);
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
            let body = ctx.with_scope(
                entry_function,
                WorkspaceVisibility::TopLevel,
                FunctionModifiers::default(),
                true,
                |ctx| {
                    ctx.seed_existing_workspace_bindings(context.variables, entry_function);
                    ctx.lower_stmt_refs(&executable)
                },
            )?;
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
                    isolated,
                    is_async,
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
                        FunctionModifiers {
                            isolated: *isolated,
                            is_async: *is_async,
                        },
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

        Ok((ctx.assembly, ctx.semantic_index))
    }

    fn reserve_function_name(&mut self, name: &str) -> FunctionId {
        if let Some(id) = self.function_names.get(name) {
            return *id;
        }
        let id = self.take_function_id();
        self.function_names.insert(name.to_string(), id);
        id
    }

    fn reserve_function_input_signature(&mut self, name: &str, params: &[String]) {
        let has_varargin = params.last().is_some_and(|param| param == "varargin");
        let fixed_count = params.len() - usize::from(has_varargin);
        self.function_input_signatures
            .insert(name.to_string(), (fixed_count, has_varargin));
    }

    fn reserve_function_output_signature(&mut self, name: &str, outputs: &[String]) {
        let has_varargout = outputs.last().is_some_and(|output| output == "varargout");
        let fixed_count = outputs.len() - usize::from(has_varargout);
        self.function_output_signatures
            .insert(name.to_string(), (fixed_count, has_varargout));
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
        modifiers: FunctionModifiers,
        top_level_await: bool,
        f: impl FnOnce(&mut Self) -> Result<T, SemanticError>,
    ) -> Result<T, SemanticError> {
        self.scopes.push(SemanticScope {
            owner,
            bindings: HashMap::new(),
            workspace_visibility,
        });
        self.function_modifiers.push(modifiers);
        self.top_level_await.push(top_level_await);
        let result = f(self);
        self.top_level_await.pop();
        self.function_modifiers.pop();
        self.scopes.pop();
        result
    }

    fn current_allows_await(&self) -> bool {
        self.function_modifiers
            .last()
            .map(|modifiers| modifiers.is_async)
            .unwrap_or(false)
            || self.top_level_await.last().copied().unwrap_or(false)
    }

    fn spawn_arg_captures_lexical_binding(&self, arg: &HirExpr) -> bool {
        match &arg.kind {
            HirExprKind::AnonymousFunction(function_id) => self
                .assembly
                .functions
                .iter()
                .find(|function| function.id == *function_id)
                .map(|function| !function.captures.is_empty())
                .unwrap_or(false),
            _ => false,
        }
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
        self.semantic_index.bindings.push(BindingResolution {
            name: BindingName(name.to_string()),
            binding: id,
            owner: BindingOwner::Function(owner),
            span,
        });
        id
    }

    fn binding_for_read(&mut self, name: &str, span: Span) -> Option<BindingId> {
        let binding = self.lookup_binding(name)?;
        self.record_capture_if_outer(binding);
        self.semantic_index.references.push(ReferenceResolution {
            name: SymbolName(name.to_string()),
            kind: ReferenceKind::Binding(binding),
            span,
        });
        Some(binding)
    }

    fn seed_existing_workspace_bindings(
        &mut self,
        variables: &HashMap<String, usize>,
        owner: FunctionId,
    ) {
        let mut variables = variables.iter().collect::<Vec<_>>();
        variables.sort_by_key(|(_, slot)| **slot);
        for (name, _) in variables {
            if self.current_scope().bindings.contains_key(name) {
                continue;
            }
            let id = self.alloc_binding_id();
            self.current_scope_mut().bindings.insert(name.clone(), id);
            self.assembly.bindings.push(HirBinding {
                id,
                owner: BindingOwner::Function(owner),
                name: BindingName(name.clone()),
                role: BindingRole::Local,
                storage: BindingStorage::Lexical,
                workspace_visibility: WorkspaceVisibility::TopLevel,
                declared_span: Span { start: 0, end: 0 },
            });
        }
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
                if let Some(owner_index) = self.scopes.iter().position(|scope| scope.owner == owner)
                {
                    let functions = self
                        .scopes
                        .iter()
                        .skip(owner_index + 1)
                        .map(|scope| scope.owner)
                        .collect::<Vec<_>>();
                    for function in functions {
                        self.record_capture(function, binding, owner);
                    }
                } else {
                    self.record_capture(current, binding, owner);
                }
            }
        }
    }

    fn record_capture(&mut self, function: FunctionId, binding: BindingId, owner: FunctionId) {
        let captures = self.captures.entry(function).or_default();
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
        modifiers: FunctionModifiers,
        parent: Option<FunctionId>,
        enclosing_class: Option<ClassId>,
    ) -> Result<HirFunction, SemanticError> {
        self.with_scope(
            id,
            WorkspaceVisibility::Hidden,
            modifiers.clone(),
            false,
            |ctx| {
                for stmt in body {
                    if let AstStmt::Function {
                        name,
                        params,
                        outputs,
                        ..
                    } = stmt
                    {
                        ctx.reserve_function_name(name);
                        ctx.reserve_function_input_signature(name, params);
                        ctx.reserve_function_output_signature(name, outputs);
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
                        ctx.define_binding(
                            output,
                            BindingRole::Output,
                            BindingStorage::Lexical,
                            span,
                        )
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
                        isolated,
                        is_async,
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
                            FunctionModifiers {
                                isolated: *isolated,
                                is_async: *is_async,
                            },
                            Some(id),
                            enclosing_class,
                        )?;
                        ctx.assembly.functions.push(nested);
                    }
                }
                let locals = ctx.binding_ids_for_owner(id);
                let captures = ctx.captures.remove(&id).unwrap_or_default();
                if modifiers.isolated && !captures.is_empty() {
                    return Err(SemanticError::new(
                        "isolated functions cannot capture outer lexical bindings",
                    )
                    .with_span(span));
                }
                ctx.semantic_index.functions.push(FunctionResolution {
                    name: FunctionName(name.to_string()),
                    function: id,
                    parent,
                    span,
                });
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
                    modifiers,
                    body: hir_body,
                    span,
                })
            },
        )
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
                runmat_parser::ClassMember::Properties { attributes, names } => {
                    let attributes = property_attributes(attributes);
                    properties.extend(names.iter().map(|name| ClassProperty {
                        name: crate::MemberName(name.clone()),
                        attributes: attributes.clone(),
                        default: None,
                        span,
                    }));
                }
                runmat_parser::ClassMember::Methods { attributes, body } => {
                    let is_static = attributes
                        .iter()
                        .any(|attr| attr.name.eq_ignore_ascii_case("Static"));
                    let method_attributes = method_attributes(attributes);
                    for stmt in body {
                        if let AstStmt::Function {
                            name,
                            params,
                            outputs,
                            body,
                            isolated,
                            is_async,
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
                                FunctionModifiers {
                                    isolated: *isolated,
                                    is_async: *is_async,
                                },
                                None,
                                Some(class_id),
                            )?;
                            self.assembly.functions.push(function);
                            methods.push(ClassMethod {
                                function: function_id,
                                name: crate::MethodName(name.clone()),
                                is_static,
                                attributes: method_attributes.clone(),
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

        let qualified = QualifiedName(vec![SymbolName(name.to_string())]);
        self.semantic_index.classes.push(ClassResolution {
            name: qualified.clone(),
            class: class_id,
            span,
        });

        Ok(HirClass {
            id: class_id,
            module: self.module,
            name: qualified,
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
                let place = HirPlace::Binding(binding);
                self.record_mutation(
                    place.clone(),
                    PlaceMutationKind::BindOrAssign,
                    AssignmentCreationPolicy::CreateBinding,
                    AssignmentShapePolicy::Exact,
                );
                HirStmtKind::Assign(
                    place,
                    self.lower_expr_semantic_requested(expr, RequestedOutputCount::One)?,
                    *suppressed,
                )
            }
            AstStmt::MultiAssign(names, expr, suppressed, _) => {
                let targets = names
                    .iter()
                    .map(|name| {
                        if name == DISCARD_OUTPUT_NAME {
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
            AstStmt::AssignLValue(lvalue, expr, suppressed, _) => {
                let deletion = is_empty_array_expr(expr);
                let place = self.lower_lvalue_semantic(lvalue, span, deletion)?;
                let kind = mutation_kind_for_place(&place, deletion);
                let creation_policy = creation_policy_for_place(&place, deletion);
                self.record_mutation(
                    place.clone(),
                    kind,
                    creation_policy,
                    AssignmentShapePolicy::MatlabCompatible,
                );
                HirStmtKind::Assign(
                    place,
                    self.lower_expr_semantic_requested(
                        expr,
                        requested_outputs_for_lvalue_assignment(lvalue, expr),
                    )?,
                    *suppressed,
                )
            }
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
        deletion: bool,
    ) -> Result<HirPlace, SemanticError> {
        use runmat_parser::LValue;
        Ok(match lvalue {
            LValue::Var(name) => HirPlace::Binding(self.binding_for_write(name, span)),
            LValue::Member(base, name) => HirPlace::Member(
                Box::new(self.lower_assignment_base_expr(base, span)?),
                crate::MemberName(name.clone()),
            ),
            LValue::MemberDynamic(base, name) => HirPlace::MemberDynamic(
                Box::new(self.lower_assignment_base_expr(base, span)?),
                Box::new(self.lower_expr_semantic(name)?),
            ),
            LValue::Index(base, indices) => HirPlace::Index(
                Box::new(self.lower_assignment_base_expr(base, span)?),
                self.lower_indexing_with_context(
                    indices,
                    IndexKind::Paren,
                    assignment_index_context(deletion),
                )?,
            ),
            LValue::IndexCell(base, indices) => HirPlace::IndexCell(
                Box::new(self.lower_assignment_base_expr(base, span)?),
                self.lower_indexing_with_context(
                    indices,
                    IndexKind::Brace,
                    assignment_index_context(deletion),
                )?,
            ),
        })
    }

    fn lower_assignment_base_expr(
        &mut self,
        expr: &AstExpr,
        span: Span,
    ) -> Result<HirExpr, SemanticError> {
        match expr {
            AstExpr::Ident(name, _) => {
                let binding = self.binding_for_write(name, span);
                Ok(HirExpr {
                    id: self.alloc_expr_id(),
                    kind: HirExprKind::Binding(binding),
                    span: expr.span(),
                })
            }
            AstExpr::Member(base, name, _) => Ok(HirExpr {
                id: self.alloc_expr_id(),
                kind: HirExprKind::Member(
                    Box::new(self.lower_assignment_base_expr(base, span)?),
                    crate::MemberName(name.clone()),
                ),
                span: expr.span(),
            }),
            AstExpr::MemberDynamic(base, name, _) => Ok(HirExpr {
                id: self.alloc_expr_id(),
                kind: HirExprKind::MemberDynamic(
                    Box::new(self.lower_assignment_base_expr(base, span)?),
                    Box::new(self.lower_expr_semantic(name)?),
                ),
                span: expr.span(),
            }),
            _ => self.lower_expr_semantic(expr),
        }
    }

    fn record_mutation(
        &mut self,
        place: HirPlace,
        kind: PlaceMutationKind,
        creation_policy: AssignmentCreationPolicy,
        shape_policy: AssignmentShapePolicy,
    ) {
        self.semantic_index.mutations.push(PlaceMutation {
            place,
            kind,
            creation_policy,
            shape_policy,
        });
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
                .binding_for_read(name, span)
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
                            span,
                        ))
                    }
                }),
            AstExpr::FuncCall(name, args, _) => {
                let args: Vec<HirExpr> = self.lower_call_arguments_for_name(name, args)?;
                if name == AWAIT_EXTENSION_NAME && args.len() == 1 {
                    if matches!(
                        self.compatibility_mode,
                        Some(crate::CompatibilityMode::MatlabStrict)
                    ) {
                        return Err(SemanticError::new(
                            "await is a RunMat extension and is not available in MATLAB strict mode",
                        )
                        .with_span(span));
                    }
                    if !self.current_allows_await() {
                        return Err(SemanticError::new(
                            "await is only allowed in async functions or top-level script code",
                        )
                        .with_span(span));
                    }
                    HirExprKind::Await(Box::new(args.into_iter().next().unwrap()))
                } else if name == SPAWN_EXTENSION_NAME && args.len() == 1 {
                    if matches!(
                        self.compatibility_mode,
                        Some(crate::CompatibilityMode::MatlabStrict)
                    ) {
                        return Err(SemanticError::new(
                            "spawn is a RunMat extension and is not available in MATLAB strict mode",
                        )
                        .with_span(span));
                    }
                    let arg = args.into_iter().next().unwrap();
                    if self.spawn_arg_captures_lexical_binding(&arg) {
                        return Err(SemanticError::new(
                            "spawn cannot capture outer lexical bindings in this context",
                        )
                        .with_span(span));
                    }
                    HirExprKind::Spawn(Box::new(arg))
                } else if matches!(name.as_str(), NARGIN_BUILTIN_NAME | NARGOUT_BUILTIN_NAME) {
                    HirExprKind::Call(self.call_for_name(
                        name,
                        args,
                        CallSyntax::Plain,
                        requested_outputs,
                        span,
                    ))
                } else if let Some(binding) = self.binding_for_read(name, span) {
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
                        span,
                    ))
                }
            }
            AstExpr::CommandCall(name, args, _) => HirExprKind::CommandCall(HirCommandCall {
                command: self
                    .call_for_name(
                        name,
                        Vec::new(),
                        CallSyntax::Plain,
                        RequestedOutputCount::Zero,
                        span,
                    )
                    .callee,
                args: args.iter().map(command_argument).collect(),
            }),
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
                let function = self.with_scope(
                    function_id,
                    WorkspaceVisibility::Hidden,
                    FunctionModifiers::default(),
                    false,
                    |ctx| {
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
                        let output_id = ctx.define_binding(
                            "__anon_out",
                            BindingRole::Output,
                            BindingStorage::Lexical,
                            *span,
                        );
                        let body_expr = ctx.lower_expr_semantic(body)?;
                        let stmt = SemanticHirStmt {
                            id: ctx.alloc_stmt_id(),
                            kind: HirStmtKind::Assign(
                                HirPlace::Binding(output_id),
                                body_expr,
                                true,
                            ),
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
                            outputs: vec![output_id],
                            abi: FunctionAbi {
                                fixed_inputs: param_ids,
                                varargin: None,
                                fixed_outputs: vec![output_id],
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
                    },
                )?;
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
                self.lower_indexing_with_context(
                    indices,
                    IndexKind::Brace,
                    IndexResultContext::ReadCommaList,
                )?,
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
            AstExpr::Member(base, name, _) => {
                let lowered_base = if let AstExpr::MetaClass(class_name, _) = &**base {
                    self.classref_expr(class_name, base.span())
                } else {
                    self.lower_expr_semantic(base)?
                };
                HirExprKind::Member(Box::new(lowered_base), crate::MemberName(name.clone()))
            }
            AstExpr::MemberDynamic(base, name, _) => HirExprKind::MemberDynamic(
                Box::new(self.lower_expr_semantic(base)?),
                Box::new(self.lower_expr_semantic(name)?),
            ),
            AstExpr::DottedInvoke(base, name, args, _)
            | AstExpr::MethodCall(base, name, args, _) => {
                if let AstExpr::MetaClass(class_name, _) = &**base {
                    let mut call_args = vec![self.classref_expr(class_name, base.span())];
                    call_args.extend(
                        args.iter()
                            .map(|arg| self.lower_call_argument(arg, RequestedOutputCount::One))
                            .collect::<Result<Vec<_>, _>>()?,
                    );
                    return Ok(HirExpr {
                        id: self.alloc_expr_id(),
                        kind: HirExprKind::Call(self.call_for_name(
                            name,
                            call_args,
                            CallSyntax::Method,
                            RequestedOutputCount::One,
                            span,
                        )),
                        span,
                    });
                }
                if let AstExpr::Ident(class_name, _) = &**base {
                    if let Some((method, _)) = runmat_builtins::lookup_method(class_name, name) {
                        if !method.is_static || method.access != runmat_builtins::Access::Public {
                            return Err(SemanticError::new(format!(
                                "method {class_name}.{name} is not accessible as a public static method"
                            ))
                            .with_span(span));
                        }
                        let mut call_args: Vec<HirExpr> = args
                            .iter()
                            .map(|arg| self.lower_call_argument(arg, RequestedOutputCount::One))
                            .collect::<Result<_, _>>()?;
                        if let Some(class_argument) = method.implicit_class_argument {
                            call_args.push(HirExpr {
                                id: self.alloc_expr_id(),
                                kind: HirExprKind::String(StringLiteral(class_argument)),
                                span: base.span(),
                            });
                        }
                        let function_name = method.function_name;
                        return Ok(HirExpr {
                            id: self.alloc_expr_id(),
                            kind: HirExprKind::Call(self.call_for_name(
                                &function_name,
                                call_args,
                                CallSyntax::Plain,
                                RequestedOutputCount::One,
                                span,
                            )),
                            span,
                        });
                    }
                    let qualified_name = format!("{class_name}.{name}");
                    if is_builtin(&qualified_name) {
                        let call_args =
                            self.lower_call_arguments_for_name(&qualified_name, args)?;
                        return Ok(HirExpr {
                            id: self.alloc_expr_id(),
                            kind: HirExprKind::Call(self.call_for_name(
                                &qualified_name,
                                call_args,
                                CallSyntax::Plain,
                                RequestedOutputCount::One,
                                span,
                            )),
                            span,
                        });
                    }
                }
                let mut call_args = vec![self.lower_expr_semantic(base)?];
                call_args.extend(
                    args.iter()
                        .map(|arg| self.lower_call_argument(arg, RequestedOutputCount::One))
                        .collect::<Result<Vec<_>, _>>()?,
                );
                HirExprKind::Call(self.call_for_name(
                    name,
                    call_args,
                    CallSyntax::Method,
                    RequestedOutputCount::One,
                    span,
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
        self.lower_indexing_with_context(indices, kind, IndexResultContext::ReadSingle)
    }

    fn lower_call_arguments_for_name(
        &mut self,
        name: &str,
        args: &[AstExpr],
    ) -> Result<Vec<HirExpr>, SemanticError> {
        let expanded_last_outputs = self.expanded_last_output_count_for_call(name, args);

        args.iter()
            .enumerate()
            .map(|(index, arg)| {
                let mut requested_outputs = match (
                    Some(index) == args.len().checked_sub(1),
                    expanded_last_outputs,
                ) {
                    (true, Some(count)) => RequestedOutputCount::Exactly(count),
                    _ => RequestedOutputCount::One,
                };
                if matches!(name, "max" | "min")
                    && args.len() == 1
                    && self.call_argument_has_varargout(arg)
                {
                    requested_outputs = RequestedOutputCount::Exactly(2);
                }
                self.lower_call_argument(arg, requested_outputs)
            })
            .collect()
    }

    fn call_argument_has_varargout(&self, arg: &AstExpr) -> bool {
        match arg {
            AstExpr::FuncCall(name, _, _) => self
                .function_output_signatures
                .get(name)
                .is_some_and(|(_, has_varargout)| *has_varargout),
            _ => false,
        }
    }

    fn expanded_last_output_count_for_call(&self, name: &str, args: &[AstExpr]) -> Option<usize> {
        let arg_count = args.len();
        if let Some((fixed_inputs, has_varargin)) = self.function_input_signatures.get(name) {
            if !has_varargin && *fixed_inputs > arg_count {
                return Some(*fixed_inputs - arg_count + 1);
            }
            return None;
        }

        if self.lookup_binding(name).is_some() {
            return None;
        }

        let builtin = runmat_builtins::builtin_function_by_name(name)?;
        if builtin_has_variadic_tail(builtin) {
            return None;
        }
        let fixed_inputs = builtin.param_types.len();
        if fixed_inputs <= arg_count {
            return None;
        }
        let requested = fixed_inputs - arg_count + 1;
        self.call_argument_can_supply_outputs(args.last()?, requested)
            .then_some(requested)
    }

    fn call_argument_can_supply_outputs(&self, arg: &AstExpr, requested: usize) -> bool {
        match arg {
            AstExpr::FuncCall(name, _, _) => self.function_output_signatures.get(name).is_some_and(
                |(fixed_outputs, has_varargout)| *has_varargout || *fixed_outputs >= requested,
            ),
            _ => false,
        }
    }

    fn lower_call_argument(
        &mut self,
        arg: &AstExpr,
        requested_outputs: RequestedOutputCount,
    ) -> Result<HirExpr, SemanticError> {
        if let AstExpr::IndexCell(base, indices, _) = arg {
            return Ok(HirExpr {
                id: self.alloc_expr_id(),
                kind: HirExprKind::Index(
                    Box::new(self.lower_expr_semantic(base)?),
                    self.lower_indexing_with_context(
                        indices,
                        IndexKind::Brace,
                        IndexResultContext::FunctionArgumentExpansion,
                    )?,
                ),
                span: arg.span(),
            });
        }
        self.lower_expr_semantic_requested(arg, requested_outputs)
    }

    fn lower_indexing_with_context(
        &mut self,
        indices: &[AstExpr],
        kind: IndexKind,
        result_context: IndexResultContext,
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
            result_context,
        })
    }

    fn classref_expr(&mut self, class_name: &str, span: Span) -> HirExpr {
        let arg = HirExpr {
            id: self.alloc_expr_id(),
            kind: HirExprKind::String(StringLiteral(class_name.to_string())),
            span,
        };
        HirExpr {
            id: self.alloc_expr_id(),
            kind: HirExprKind::Call(self.call_for_name(
                "classref",
                vec![arg],
                CallSyntax::Plain,
                RequestedOutputCount::One,
                span,
            )),
            span,
        }
    }

    fn call_for_name(
        &mut self,
        name: &str,
        args: Vec<HirExpr>,
        syntax: CallSyntax,
        requested_outputs: RequestedOutputCount,
        span: Span,
    ) -> HirCall {
        let (callee, kind) = if let Some(function) = self.function_names.get(name) {
            (
                HirCallableRef::Function(*function),
                CallKind::DirectFunction(*function),
            )
        } else if let Some(function) = self.external_function_names.get(name) {
            (
                HirCallableRef::Function(*function),
                CallKind::DirectFunction(*function),
            )
        } else if is_builtin(name) {
            let builtin = BuiltinId(name.to_string());
            (
                HirCallableRef::Builtin(builtin.clone()),
                CallKind::Builtin(builtin),
            )
        } else {
            (
                HirCallableRef::Unresolved(QualifiedName(vec![SymbolName(name.to_string())])),
                CallKind::Dynamic,
            )
        };
        self.semantic_index.calls.push(CallResolution {
            name: QualifiedName(vec![SymbolName(name.to_string())]),
            callee: callee.clone(),
            kind,
            requested_outputs: requested_outputs.clone(),
            span,
        });
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

fn command_argument(expr: &AstExpr) -> CommandArgument {
    match expr {
        AstExpr::Ident(word, _) | AstExpr::Number(word, _) => {
            CommandArgument::Word(SymbolName(word.clone()))
        }
        AstExpr::String(value, _) => CommandArgument::StringLiteral(StringLiteral(value.clone())),
        AstExpr::EndKeyword(_) => CommandArgument::Word(SymbolName("end".to_string())),
        _ => CommandArgument::StringLiteral(StringLiteral(format!("{expr:?}"))),
    }
}

fn is_empty_array_expr(expr: &AstExpr) -> bool {
    matches!(expr, AstExpr::Tensor(rows, _) if rows.is_empty() || rows.iter().all(Vec::is_empty))
}

fn requested_outputs_for_lvalue_assignment(
    lvalue: &runmat_parser::LValue,
    expr: &AstExpr,
) -> RequestedOutputCount {
    if !matches!(expr, AstExpr::FuncCall(_, _, _) | AstExpr::Ident(_, _)) {
        return RequestedOutputCount::One;
    }
    static_lvalue_assignment_count(lvalue)
        .filter(|count| *count > 1)
        .map(RequestedOutputCount::Exactly)
        .unwrap_or(RequestedOutputCount::One)
}

fn static_lvalue_assignment_count(lvalue: &runmat_parser::LValue) -> Option<usize> {
    use runmat_parser::LValue;
    match lvalue {
        LValue::Index(_, indices) => indices.iter().try_fold(1usize, |acc, index| {
            static_index_component_count(index).map(|count| acc.saturating_mul(count))
        }),
        _ => None,
    }
}

fn static_index_component_count(expr: &AstExpr) -> Option<usize> {
    match expr {
        AstExpr::Number(_, _) | AstExpr::Ident(_, _) | AstExpr::EndKeyword(_) => Some(1),
        AstExpr::Tensor(rows, _) => Some(rows.iter().map(Vec::len).sum()),
        AstExpr::Range(start, step, end, _) => {
            let start = static_numeric_literal(start)?;
            let step = step
                .as_deref()
                .and_then(static_numeric_literal)
                .unwrap_or_else(|| {
                    if start <= static_numeric_literal(end).unwrap_or(start) {
                        1.0
                    } else {
                        -1.0
                    }
                });
            let end = static_numeric_literal(end)?;
            static_range_count(start, step, end)
        }
        _ => None,
    }
}

fn static_numeric_literal(expr: &AstExpr) -> Option<f64> {
    match expr {
        AstExpr::Number(value, _) => value.parse().ok(),
        _ => None,
    }
}

fn static_range_count(start: f64, step: f64, end: f64) -> Option<usize> {
    if step == 0.0 || !start.is_finite() || !step.is_finite() || !end.is_finite() {
        return None;
    }
    if (step > 0.0 && start > end) || (step < 0.0 && start < end) {
        return Some(0);
    }
    Some(((end - start) / step).floor().max(0.0) as usize + 1)
}

fn assignment_index_context(deletion: bool) -> IndexResultContext {
    if deletion {
        IndexResultContext::DeletionTarget
    } else {
        IndexResultContext::AssignmentTarget
    }
}

fn mutation_kind_for_place(place: &HirPlace, deletion: bool) -> PlaceMutationKind {
    if deletion {
        return PlaceMutationKind::Delete;
    }
    match place {
        HirPlace::Binding(_) => PlaceMutationKind::BindOrAssign,
        HirPlace::Index(_, _) => PlaceMutationKind::IndexedAssign,
        HirPlace::IndexCell(_, _) => PlaceMutationKind::CellAssign,
        HirPlace::Member(_, _) | HirPlace::MemberDynamic(_, _) => PlaceMutationKind::MemberAssign,
    }
}

fn creation_policy_for_place(place: &HirPlace, deletion: bool) -> AssignmentCreationPolicy {
    if deletion {
        return AssignmentCreationPolicy::ExistingOnly;
    }
    match place {
        HirPlace::Binding(_) => AssignmentCreationPolicy::CreateBinding,
        HirPlace::Index(_, _) | HirPlace::IndexCell(_, _) => {
            AssignmentCreationPolicy::CreateArrayByIndex
        }
        HirPlace::Member(_, _) | HirPlace::MemberDynamic(_, _) => {
            AssignmentCreationPolicy::CreateStructFieldPath
        }
    }
}

fn property_attributes(attrs: &[runmat_parser::Attr]) -> crate::PropertyAttributes {
    let mut result = crate::PropertyAttributes::default();
    for attr in attrs {
        if attr.name.eq_ignore_ascii_case("Static") {
            result.is_static = true;
        } else if attr.name.eq_ignore_ascii_case("Constant") {
            result.is_constant = true;
        } else if attr.name.eq_ignore_ascii_case("Dependent") {
            result.is_dependent = true;
        } else if attr.name.eq_ignore_ascii_case("Transient") {
            result.is_transient = true;
        } else if attr.name.eq_ignore_ascii_case("Hidden") {
            result.is_hidden = true;
        } else if attr.name.eq_ignore_ascii_case("Access") {
            if let Some(access) = attr.value.as_deref().and_then(member_access) {
                result.access = access.clone();
                result.get_access = access.clone();
                result.set_access = access;
            }
        } else if attr.name.eq_ignore_ascii_case("GetAccess") {
            if let Some(access) = attr.value.as_deref().and_then(member_access) {
                result.get_access = access;
            }
        } else if attr.name.eq_ignore_ascii_case("SetAccess") {
            if let Some(access) = attr.value.as_deref().and_then(member_access) {
                result.set_access = access;
            }
        }
    }
    result
}

fn method_attributes(attrs: &[runmat_parser::Attr]) -> crate::MethodAttributes {
    let mut result = crate::MethodAttributes::default();
    for attr in attrs {
        if attr.name.eq_ignore_ascii_case("Access") {
            if let Some(access) = attr.value.as_deref().and_then(member_access) {
                result.access = access;
            }
        } else if attr.name.eq_ignore_ascii_case("Hidden") {
            result.is_hidden = true;
        } else if attr.name.eq_ignore_ascii_case("Abstract") {
            result.is_abstract = true;
        } else if attr.name.eq_ignore_ascii_case("Sealed") {
            result.is_sealed = true;
        }
    }
    result
}

fn member_access(value: &str) -> Option<crate::MemberAccess> {
    match value
        .trim()
        .trim_matches('\'')
        .to_ascii_lowercase()
        .as_str()
    {
        "public" => Some(crate::MemberAccess::Public),
        "private" => Some(crate::MemberAccess::Private),
        _ => None,
    }
}

fn is_builtin(name: &str) -> bool {
    runmat_builtins::builtin_functions()
        .iter()
        .any(|builtin| builtin.name == name)
        || runmat_builtins::builtin_semantics_for_name(name).is_some()
}

fn builtin_has_variadic_tail(builtin: &runmat_builtins::BuiltinFunction) -> bool {
    matches!(
        builtin.param_types.last(),
        Some(runmat_builtins::Type::Cell { length: None, .. })
    )
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
        is_static_method_class_name(name)
    }
}

fn is_static_method_class_name(name: &str) -> bool {
    matches!(
        name,
        "logical"
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
