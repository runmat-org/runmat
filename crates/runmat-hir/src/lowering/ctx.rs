use crate::{
    AssignmentCreationPolicy, AssignmentShapePolicy, BindingId, BindingName, BindingOwner,
    BindingResolution, BindingRole, BindingStorage, BuiltinId, CallKind, CallResolution,
    CallSyntax, CapturedBinding, ClassArgumentBlock, ClassEnumeration, ClassEvent, ClassId,
    ClassKind, ClassMethod, ClassProperty, ClassResolution, CommandArgument, DefPath,
    DefPathSegment, EntrypointId, EntrypointName, EntrypointOrigin, EntrypointPolicy, ExprId,
    FunctionAbi, FunctionId, FunctionKind, FunctionModifiers, FunctionName, FunctionResolution,
    HirAssembly, HirBinding, HirBlock, HirCall, HirCallableRef, HirClass, HirCommandCall,
    HirEntrypoint, HirError, HirExpr, HirExprKind, HirFunction, HirImport, HirIndex, HirModule,
    HirPlace, HirStmt as HirStmtNode, HirStmtKind, ImportResolution, IndexComponent, IndexKind,
    IndexResultContext, IndexingSemantics, LoweringContext, LoweringResult, ModuleId, OperatorKind,
    PackageName, PlaceMutation, PlaceMutationKind, QualifiedName, ReferenceKind,
    ReferenceResolution, RequestedOutputCount, SourceId, SourceUnitKind, Span, StmtId,
    StringLiteral, SymbolName, WorkspaceExportPolicy, WorkspaceVisibility, AWAIT_EXTENSION_NAME,
    DISCARD_OUTPUT_NAME, NARGIN_BUILTIN_NAME, NARGOUT_BUILTIN_NAME, SPAWN_EXTENSION_NAME,
};
use runmat_parser::{BinOp, Expr as AstExpr, Program as AstProgram, Stmt as AstStmt, UnOp};
use std::collections::{HashMap, HashSet};

const IDENT_AWAIT_EXTENSION_DISABLED: &str = "RunMat:AwaitExtensionDisabled";
const IDENT_AWAIT_CONTEXT_INVALID: &str = "RunMat:AwaitContextInvalid";
const IDENT_SPAWN_EXTENSION_DISABLED: &str = "RunMat:SpawnExtensionDisabled";
const IDENT_SPAWN_LEXICAL_CAPTURE_UNSUPPORTED: &str = "RunMat:SpawnLexicalCaptureUnsupported";
const IDENT_ISOLATED_LEXICAL_CAPTURE_UNSUPPORTED: &str = "RunMat:IsolatedLexicalCaptureUnsupported";
const IDENT_UNDEFINED_VARIABLE: &str = "RunMat:UndefinedVariable";
const IDENT_CLASS_PROPERTY_ATTRIBUTE_CONFLICT: &str = "RunMat:ClassPropertyAttributeConflict";
const IDENT_CLASS_METHOD_ATTRIBUTE_CONFLICT: &str = "RunMat:ClassMethodAttributeConflict";
const IDENT_CLASS_ACCESS_VALUE_INVALID: &str = "RunMat:ClassAccessValueInvalid";
const IDENT_CLASS_SELF_INHERITANCE_INVALID: &str = "RunMat:ClassSelfInheritanceInvalid";
const IDENT_CLASS_MEMBER_DUPLICATE: &str = "RunMat:ClassMemberDuplicate";
const IDENT_CLASS_MEMBER_NAME_CONFLICT: &str = "RunMat:ClassMemberNameConflict";
const IDENT_AGGREGATE_SHAPE_MISMATCH: &str = "RunMat:AggregateShapeMismatch";
const IDENT_IMPORT_AMBIGUOUS: &str = "RunMat:ImportAmbiguous";
const IDENT_IMPORT_DUPLICATE: &str = "RunMat:ImportDuplicate";

#[derive(Clone)]
struct ScopeFrame {
    owner: FunctionId,
    bindings: HashMap<String, BindingId>,
    workspace_visibility: WorkspaceVisibility,
}

struct LoweringCtx {
    assembly: HirAssembly,
    hir_index: HirIndex,
    module: ModuleId,
    next_expr: usize,
    next_stmt: usize,
    next_function: usize,
    scopes: Vec<ScopeFrame>,
    function_modifiers: Vec<FunctionModifiers>,
    top_level_await: Vec<bool>,
    runmat_extensions_enabled: bool,
    top_level_await_enabled: bool,
    function_names: HashMap<String, FunctionId>,
    function_input_signatures: HashMap<String, (usize, bool)>,
    function_output_signatures: HashMap<String, (usize, bool)>,
    external_function_names: HashMap<String, FunctionId>,
    known_project_symbols: HashSet<String>,
    captures: HashMap<FunctionId, Vec<CapturedBinding>>,
}

struct LowerFunctionSpec<'a> {
    id: FunctionId,
    name: &'a str,
    params: &'a [String],
    outputs: &'a [String],
    body: &'a [AstStmt],
    span: Span,
    kind: FunctionKind,
    modifiers: FunctionModifiers,
    parent: Option<FunctionId>,
    enclosing_class: Option<ClassId>,
}

pub fn lower(prog: &AstProgram, context: &LoweringContext<'_>) -> Result<LoweringResult, HirError> {
    let (assembly, hir_index) = LoweringCtx::lower_program(prog, context)?;

    Ok(LoweringResult {
        assembly,
        hir_index,
    })
}

impl LoweringCtx {
    fn qualified_name_string(name: &QualifiedName) -> String {
        name.0
            .iter()
            .map(|segment| segment.0.as_str())
            .collect::<Vec<_>>()
            .join(".")
    }

    fn wildcard_candidate_is_resolvable(&self, qualified: &QualifiedName) -> bool {
        qualified.display_name().as_deref().is_some_and(is_builtin)
            || self
                .known_project_symbols
                .contains(&Self::qualified_name_string(qualified))
    }

    fn wildcard_import_candidate(import_path: &QualifiedName, name: &str) -> QualifiedName {
        let mut segments = import_path.0.clone();
        segments.push(SymbolName(name.to_string()));
        QualifiedName(segments)
    }

    fn is_well_formed_qualified_name(name: &str) -> bool {
        let segments = name.split('.').collect::<Vec<_>>();
        segments.len() > 1 && segments.iter().all(|segment| !segment.is_empty())
    }

    fn validate_rectangular_aggregate(
        &self,
        kind: &str,
        rows: &[Vec<AstExpr>],
    ) -> Result<(), HirError> {
        let Some(expected_cols) = rows.first().map(Vec::len) else {
            return Ok(());
        };
        if rows.iter().all(|row| row.len() == expected_cols) {
            return Ok(());
        }

        Err(HirError::new(format!(
            "{kind} literal rows must have consistent column counts",
        ))
        .with_identifier(IDENT_AGGREGATE_SHAPE_MISMATCH))
    }

    fn lower_program(
        prog: &AstProgram,
        context: &LoweringContext<'_>,
    ) -> Result<(HirAssembly, HirIndex), HirError> {
        let mut ctx = Self {
            assembly: HirAssembly::default(),
            hir_index: HirIndex::default(),
            module: ModuleId(0),
            next_expr: 0,
            next_stmt: 0,
            next_function: 0,
            scopes: Vec::new(),
            function_modifiers: Vec::new(),
            top_level_await: Vec::new(),
            runmat_extensions_enabled: context.runmat_extensions_enabled,
            top_level_await_enabled: context.top_level_await_enabled,
            function_names: HashMap::new(),
            function_input_signatures: HashMap::new(),
            function_output_signatures: HashMap::new(),
            external_function_names: context.bound_functions.clone(),
            known_project_symbols: context.known_project_symbols.clone(),
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
                ctx.hir_index.imports.push(ImportResolution {
                    import: import.clone(),
                });
                ctx.assembly.modules[ctx.module.0].imports.push(import);
            }
        }
        validate_semantic_imports(&ctx.assembly.modules[ctx.module.0].imports)?;

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
                    top_level_await: context.top_level_await_enabled,
                },
            });
            let body = ctx.with_scope(
                entry_function,
                WorkspaceVisibility::TopLevel,
                FunctionModifiers::default(),
                ctx.top_level_await_enabled,
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
                    let function = ctx.lower_function(LowerFunctionSpec {
                        id,
                        name,
                        params,
                        outputs,
                        body,
                        span: *span,
                        kind: FunctionKind::Named,
                        modifiers: FunctionModifiers {
                            isolated: *isolated,
                            is_async: *is_async,
                        },
                        parent: None,
                        enclosing_class: None,
                    })?;
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

        Ok((ctx.assembly, ctx.hir_index))
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
        f: impl FnOnce(&mut Self) -> Result<T, HirError>,
    ) -> Result<T, HirError> {
        self.scopes.push(ScopeFrame {
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

    fn current_scope(&self) -> &ScopeFrame {
        self.scopes.last().expect("semantic lowering scope")
    }

    fn current_scope_mut(&mut self) -> &mut ScopeFrame {
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
        self.hir_index.bindings.push(BindingResolution {
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
        self.hir_index.references.push(ReferenceResolution {
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

    fn lower_function(&mut self, spec: LowerFunctionSpec<'_>) -> Result<HirFunction, HirError> {
        let LowerFunctionSpec {
            id,
            name,
            params,
            outputs,
            body,
            span,
            kind,
            modifiers,
            parent,
            enclosing_class,
        } = spec;
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
                        let nested = ctx.lower_function(LowerFunctionSpec {
                            id: nested_id,
                            name,
                            params,
                            outputs,
                            body,
                            span: *span,
                            kind: FunctionKind::Named,
                            modifiers: FunctionModifiers {
                                isolated: *isolated,
                                is_async: *is_async,
                            },
                            parent: Some(id),
                            enclosing_class,
                        })?;
                        ctx.assembly.functions.push(nested);
                    }
                }
                let locals = ctx.binding_ids_for_owner(id);
                let captures = ctx.captures.remove(&id).unwrap_or_default();
                if modifiers.isolated && !captures.is_empty() {
                    return Err(HirError::new(
                        "isolated functions cannot capture outer lexical bindings",
                    )
                    .with_identifier(IDENT_ISOLATED_LEXICAL_CAPTURE_UNSUPPORTED)
                    .with_span(span));
                }
                ctx.hir_index.functions.push(FunctionResolution {
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
    ) -> Result<HirClass, HirError> {
        if super_class.is_some_and(|sup| sup == name) {
            return Err(
                HirError::new(format!("Class '{name}' cannot inherit from itself"))
                    .with_identifier(IDENT_CLASS_SELF_INHERITANCE_INVALID),
            );
        }
        let class_id = ClassId(self.assembly.classes.len());
        self.assembly.modules[self.module.0].classes.push(class_id);
        let mut properties = Vec::new();
        let mut methods = Vec::new();
        let mut events = Vec::new();
        let mut enumerations = Vec::new();
        let mut arguments = Vec::new();
        let mut property_names = HashSet::new();
        let mut method_names = HashSet::new();

        for member in members {
            match member {
                runmat_parser::ClassMember::Properties { attributes, names } => {
                    let attributes = property_attributes(name, attributes)?;
                    for prop_name in names {
                        if !property_names.insert(prop_name.clone()) {
                            return Err(HirError::new(format!(
                                "Duplicate property '{prop_name}' in class {name}"
                            ))
                            .with_identifier(IDENT_CLASS_MEMBER_DUPLICATE));
                        }
                        if method_names.contains(prop_name) {
                            return Err(HirError::new(format!(
                                "Name '{prop_name}' used for both property and method in class {name}"
                            ))
                            .with_identifier(IDENT_CLASS_MEMBER_NAME_CONFLICT));
                        }
                        properties.push(ClassProperty {
                            name: crate::MemberName(prop_name.clone()),
                            attributes: attributes.clone(),
                            default: None,
                            span,
                        });
                    }
                }
                runmat_parser::ClassMember::Methods { attributes, body } => {
                    let is_static = attributes
                        .iter()
                        .any(|attr| attr.name.eq_ignore_ascii_case("Static"));
                    let method_attributes = method_attributes(name, attributes)?;
                    for stmt in body {
                        if let AstStmt::Function {
                            name: method_name,
                            params,
                            outputs,
                            body,
                            isolated,
                            is_async,
                            span,
                        } = stmt
                        {
                            let function_id = self.take_function_id();
                            let function = self.lower_function(LowerFunctionSpec {
                                id: function_id,
                                name: method_name,
                                params,
                                outputs,
                                body,
                                span: *span,
                                kind: FunctionKind::ClassMethod { is_static },
                                modifiers: FunctionModifiers {
                                    isolated: *isolated,
                                    is_async: *is_async,
                                },
                                parent: None,
                                enclosing_class: Some(class_id),
                            })?;
                            self.assembly.functions.push(function);
                            if !method_names.insert(method_name.clone()) {
                                return Err(HirError::new(format!(
                                    "Duplicate method '{method_name}' in class {name}",
                                ))
                                .with_identifier(IDENT_CLASS_MEMBER_DUPLICATE));
                            }
                            if property_names.contains(method_name) {
                                return Err(HirError::new(format!(
                                    "Name '{method_name}' used for both property and method in class {name}",
                                ))
                                .with_identifier(IDENT_CLASS_MEMBER_NAME_CONFLICT));
                            }
                            methods.push(ClassMethod {
                                function: function_id,
                                name: crate::MethodName(method_name.clone()),
                                is_static,
                                attributes: method_attributes.clone(),
                                span: *span,
                            });
                        }
                    }
                }
                runmat_parser::ClassMember::Events { names, .. } => {
                    let mut seen = HashSet::new();
                    for event_name in names {
                        if !seen.insert(event_name) {
                            return Err(HirError::new(format!(
                                "Duplicate event '{event_name}' in class {name}"
                            ))
                            .with_identifier(IDENT_CLASS_MEMBER_DUPLICATE));
                        }
                        if property_names.contains(event_name) || method_names.contains(event_name)
                        {
                            return Err(HirError::new(format!(
                                "Name '{event_name}' used for event conflicts with existing member in class {name}"
                            ))
                            .with_identifier(IDENT_CLASS_MEMBER_NAME_CONFLICT));
                        }
                        events.push(ClassEvent {
                            name: SymbolName(event_name.clone()),
                            span,
                        });
                    }
                }
                runmat_parser::ClassMember::Enumeration { names, .. } => {
                    let mut seen = HashSet::new();
                    for enum_name in names {
                        if !seen.insert(enum_name) {
                            return Err(HirError::new(format!(
                                "Duplicate enumeration '{enum_name}' in class {name}"
                            ))
                            .with_identifier(IDENT_CLASS_MEMBER_DUPLICATE));
                        }
                        if property_names.contains(enum_name) || method_names.contains(enum_name) {
                            return Err(HirError::new(format!(
                                "Name '{enum_name}' used for enumeration conflicts with existing member in class {name}"
                            ))
                            .with_identifier(IDENT_CLASS_MEMBER_NAME_CONFLICT));
                        }
                        enumerations.push(ClassEnumeration {
                            name: SymbolName(enum_name.clone()),
                            span,
                        });
                    }
                }
                runmat_parser::ClassMember::Arguments { .. } => {
                    arguments.push(ClassArgumentBlock { span });
                }
            }
        }

        let qualified = QualifiedName(vec![SymbolName(name.to_string())]);
        self.hir_index.classes.push(ClassResolution {
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

    fn lower_stmt_refs(&mut self, stmts: &[&AstStmt]) -> Result<HirBlock, HirError> {
        let mut statements = Vec::new();
        for stmt in stmts {
            if let Some(stmt) = self.lower_stmt_hir(stmt)? {
                statements.push(stmt);
            }
        }
        Ok(HirBlock { statements })
    }

    fn lower_stmts_semantic(&mut self, stmts: &[AstStmt]) -> Result<HirBlock, HirError> {
        let refs: Vec<_> = stmts.iter().collect();
        self.lower_stmt_refs(&refs)
    }

    fn lower_stmt_hir(&mut self, stmt: &AstStmt) -> Result<Option<HirStmtNode>, HirError> {
        let span = stmt.span();
        let kind = match stmt {
            AstStmt::ExprStmt(expr, suppressed, _) => {
                let requested_outputs =
                    if *suppressed || stmt_expr_call_loads_external_bindings(expr) {
                        RequestedOutputCount::Zero
                    } else {
                        RequestedOutputCount::One
                    };
                HirStmtKind::ExprStmt(
                    self.lower_expr_semantic_requested(expr, requested_outputs)?,
                    *suppressed,
                )
            }
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
                    crate::OutputTargetList {
                        requested_outputs: RequestedOutputCount::Exactly(names.len()),
                        targets,
                    },
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
                    .collect::<Result<_, HirError>>()?,
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
                    .collect::<Result<_, HirError>>()?,
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
                let deletion = lvalue_supports_deletion(lvalue) && is_empty_array_expr(expr);
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
        Ok(Some(HirStmtNode {
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
    ) -> Result<HirPlace, HirError> {
        use runmat_parser::LValue;
        Ok(match lvalue {
            LValue::Var(name) => HirPlace::Binding(self.binding_for_write(name, span)),
            LValue::Member(base, name) => HirPlace::Member(
                Box::new(self.lower_assignment_base_expr(
                    base,
                    span,
                    assignment_index_context(deletion),
                )?),
                crate::MemberName(name.clone()),
            ),
            LValue::MemberDynamic(base, name) => HirPlace::MemberDynamic(
                Box::new(self.lower_assignment_base_expr(
                    base,
                    span,
                    assignment_index_context(deletion),
                )?),
                Box::new(self.lower_expr_semantic(name)?),
            ),
            LValue::Index(base, indices) => HirPlace::Index(
                Box::new(self.lower_assignment_base_expr(
                    base,
                    span,
                    assignment_index_context(deletion),
                )?),
                self.lower_indexing_with_context(
                    indices,
                    IndexKind::Paren,
                    assignment_index_context(deletion),
                )?,
            ),
            LValue::IndexCell(base, indices) => HirPlace::IndexCell(
                Box::new(self.lower_assignment_base_expr(
                    base,
                    span,
                    assignment_index_context(deletion),
                )?),
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
        index_context: IndexResultContext,
    ) -> Result<HirExpr, HirError> {
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
                    Box::new(self.lower_assignment_base_expr(base, span, index_context)?),
                    crate::MemberName(name.clone()),
                ),
                span: expr.span(),
            }),
            AstExpr::MemberDynamic(base, name, _) => Ok(HirExpr {
                id: self.alloc_expr_id(),
                kind: HirExprKind::MemberDynamic(
                    Box::new(self.lower_assignment_base_expr(base, span, index_context)?),
                    Box::new(self.lower_expr_semantic(name)?),
                ),
                span: expr.span(),
            }),
            AstExpr::Index(base, indices, _) => Ok(HirExpr {
                id: self.alloc_expr_id(),
                kind: HirExprKind::Index(
                    Box::new(self.lower_assignment_base_expr(base, span, index_context.clone())?),
                    self.lower_indexing_with_context(indices, IndexKind::Paren, index_context)?,
                ),
                span: expr.span(),
            }),
            AstExpr::IndexCell(base, indices, _) => Ok(HirExpr {
                id: self.alloc_expr_id(),
                kind: HirExprKind::Index(
                    Box::new(self.lower_assignment_base_expr(base, span, index_context.clone())?),
                    self.lower_indexing_with_context(indices, IndexKind::Brace, index_context)?,
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
        self.hir_index.mutations.push(PlaceMutation {
            place,
            kind,
            creation_policy,
            shape_policy,
        });
    }

    fn lower_expr_semantic(&mut self, expr: &AstExpr) -> Result<HirExpr, HirError> {
        self.lower_expr_semantic_requested(expr, RequestedOutputCount::One)
    }

    fn lower_expr_semantic_requested(
        &mut self,
        expr: &AstExpr,
        requested_outputs: RequestedOutputCount,
    ) -> Result<HirExpr, HirError> {
        let span = expr.span();
        let kind = match expr {
            AstExpr::Number(value, _) => HirExprKind::Number(value.clone()),
            AstExpr::String(value, _) => HirExprKind::String(StringLiteral(value.clone())),
            AstExpr::Ident(name, _) => {
                if let Some(binding) = self.binding_for_read(name, span) {
                    HirExprKind::Binding(binding)
                } else if runmat_builtins::constants()
                    .iter()
                    .any(|c| c.name == name.as_str())
                {
                    HirExprKind::Constant(SymbolName(name.clone()))
                } else {
                    return Err(HirError::new(format!("undefined variable '{name}'"))
                        .with_span(span)
                        .with_identifier(IDENT_UNDEFINED_VARIABLE));
                }
            }
            AstExpr::FuncCall(name, args, _) => {
                let args: Vec<HirExpr> = self.lower_call_arguments_for_name(name, args)?;
                if name == AWAIT_EXTENSION_NAME && args.len() == 1 {
                    if !self.runmat_extensions_enabled {
                        return Err(HirError::new(
                            "await is a RunMat extension and is not available in MATLAB strict mode",
                        )
                        .with_span(span)
                        .with_identifier(IDENT_AWAIT_EXTENSION_DISABLED));
                    }
                    if !self.current_allows_await() {
                        return Err(HirError::new(
                            "await is only allowed in async functions or top-level script code",
                        )
                        .with_span(span)
                        .with_identifier(IDENT_AWAIT_CONTEXT_INVALID));
                    }
                    HirExprKind::Await(Box::new(args.into_iter().next().unwrap()))
                } else if name == SPAWN_EXTENSION_NAME && args.len() == 1 {
                    if !self.runmat_extensions_enabled {
                        return Err(HirError::new(
                            "spawn is a RunMat extension and is not available in MATLAB strict mode",
                        )
                        .with_span(span)
                        .with_identifier(IDENT_SPAWN_EXTENSION_DISABLED));
                    }
                    let arg = args.into_iter().next().unwrap();
                    if self.spawn_arg_captures_lexical_binding(&arg) {
                        return Err(HirError::new(
                            "spawn cannot capture outer lexical bindings in this context",
                        )
                        .with_span(span)
                        .with_identifier(IDENT_SPAWN_LEXICAL_CAPTURE_UNSUPPORTED));
                    }
                    HirExprKind::Spawn(Box::new(arg))
                } else if matches!(name.as_str(), NARGIN_BUILTIN_NAME | NARGOUT_BUILTIN_NAME) {
                    HirExprKind::Call(self.call_for_name(
                        name,
                        args,
                        CallSyntax::Plain,
                        requested_outputs,
                        span,
                    )?)
                } else if let Some(binding) = self.binding_for_read(name, span) {
                    let base = HirExpr {
                        id: self.alloc_expr_id(),
                        kind: HirExprKind::Binding(binding),
                        span,
                    };
                    let needs_dynamic_call_dispatch = requested_outputs.fixed_count() != 1
                        || args.iter().any(|arg| {
                            matches!(
                                &arg.kind,
                                HirExprKind::Index(_, indexing)
                                    if indexing.kind == IndexKind::Brace
                                        && matches!(
                                            indexing.result_context,
                                            IndexResultContext::FunctionArgumentExpansion
                                        )
                            )
                        });
                    if needs_dynamic_call_dispatch {
                        HirExprKind::Call(HirCall {
                            callee: HirCallableRef::DynamicExpr(Box::new(base)),
                            args,
                            syntax: CallSyntax::Plain,
                            requested_outputs,
                        })
                    } else {
                        HirExprKind::Index(
                            Box::new(base),
                            IndexingSemantics {
                                kind: IndexKind::Paren,
                                components: args.into_iter().map(IndexComponent::Expr).collect(),
                                result_context: IndexResultContext::ReadSingle,
                            },
                        )
                    }
                } else {
                    HirExprKind::Call(self.call_for_name(
                        name,
                        args,
                        CallSyntax::Plain,
                        requested_outputs,
                        span,
                    )?)
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
                    )?
                    .callee,
                args: args.iter().map(command_argument).collect(),
            }),
            AstExpr::FuncHandle(name, _) => {
                HirExprKind::FunctionHandle(self.resolve_function_handle_target(name, span)?)
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
                        let stmt = HirStmtNode {
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
            AstExpr::Tensor(rows, _) => {
                self.validate_rectangular_aggregate("tensor", rows)?;
                HirExprKind::Tensor(
                    rows.iter()
                        .map(|row| {
                            row.iter()
                                .map(|expr| self.lower_expr_semantic(expr))
                                .collect()
                        })
                        .collect::<Result<_, _>>()?,
                )
            }
            AstExpr::Cell(rows, _) => {
                self.validate_rectangular_aggregate("cell", rows)?;
                HirExprKind::Cell(
                    rows.iter()
                        .map(|row| {
                            row.iter()
                                .map(|expr| self.lower_expr_semantic(expr))
                                .collect()
                        })
                        .collect::<Result<_, _>>()?,
                )
            }
            AstExpr::StructLiteral(fields, _) => HirExprKind::StructLiteral(
                fields
                    .iter()
                    .map(|(name, expr)| {
                        Ok((
                            crate::MemberName(name.clone()),
                            self.lower_expr_semantic(expr)?,
                        ))
                    })
                    .collect::<Result<_, HirError>>()?,
            ),
            AstExpr::ObjectLiteral(class_name, fields, _) => HirExprKind::ObjectLiteral {
                class_name: qualified_name(
                    &class_name
                        .split('.')
                        .map(|segment| segment.to_string())
                        .collect::<Vec<_>>(),
                ),
                fields: fields
                    .iter()
                    .map(|(name, expr)| {
                        Ok((
                            crate::MemberName(name.clone()),
                            self.lower_expr_semantic(expr)?,
                        ))
                    })
                    .collect::<Result<_, HirError>>()?,
            },
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
                    self.classref_expr(class_name, base.span())?
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
                    let mut call_args = vec![self.classref_expr(class_name, base.span())?];
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
                            requested_outputs,
                            span,
                        )?),
                        span,
                    });
                }
                if let AstExpr::Ident(class_name, _) = &**base {
                    if let Some((method, _)) = runmat_builtins::lookup_method(class_name, name) {
                        if !method.is_static || method.access != runmat_builtins::Access::Public {
                            return Err(HirError::new(format!(
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
                                requested_outputs,
                                span,
                            )?),
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
                                requested_outputs,
                                span,
                            )?),
                            span,
                        });
                    }
                    if self.lookup_binding(class_name).is_none() {
                        let call_args =
                            self.lower_call_arguments_for_name(&qualified_name, args)?;
                        return Ok(HirExpr {
                            id: self.alloc_expr_id(),
                            kind: HirExprKind::Call(self.call_for_name(
                                &qualified_name,
                                call_args,
                                CallSyntax::Plain,
                                requested_outputs,
                                span,
                            )?),
                            span,
                        });
                    }
                }
                if let Some(qualified_base) = self.unbound_qualified_member_base(base) {
                    let qualified_name = format!("{qualified_base}.{name}");
                    let call_args = self.lower_call_arguments_for_name(&qualified_name, args)?;
                    return Ok(HirExpr {
                        id: self.alloc_expr_id(),
                        kind: HirExprKind::Call(self.call_for_name(
                            &qualified_name,
                            call_args,
                            CallSyntax::Plain,
                            requested_outputs,
                            span,
                        )?),
                        span,
                    });
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
                    requested_outputs,
                    span,
                )?)
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
    ) -> Result<IndexingSemantics, HirError> {
        self.lower_indexing_with_context(indices, kind, IndexResultContext::ReadSingle)
    }

    fn lower_call_arguments_for_name(
        &mut self,
        name: &str,
        args: &[AstExpr],
    ) -> Result<Vec<HirExpr>, HirError> {
        let expanded_last_outputs = self.expanded_last_output_count_for_call(name, args);

        args.iter()
            .enumerate()
            .map(|(index, arg)| {
                let requested_outputs = match (
                    Some(index) == args.len().checked_sub(1),
                    expanded_last_outputs,
                ) {
                    (true, Some(count)) => RequestedOutputCount::Exactly(count),
                    _ => RequestedOutputCount::One,
                };
                self.lower_call_argument(arg, requested_outputs)
            })
            .collect()
    }

    fn expanded_last_output_count_for_call(&self, name: &str, args: &[AstExpr]) -> Option<usize> {
        let arg_count = args.len();
        if let Some((fixed_inputs, has_varargin)) = self.function_input_signatures.get(name) {
            if !has_varargin && *fixed_inputs > arg_count {
                return Some(*fixed_inputs - arg_count + 1);
            }
            return None;
        }

        if let Some(min_inputs) = builtin_min_inputs_for_last_arg_expansion(name) {
            if min_inputs <= arg_count {
                return None;
            }
            let requested = min_inputs - arg_count + 1;
            return self
                .call_argument_can_supply_outputs(args.last()?, requested)
                .then_some(requested);
        }

        if self.lookup_binding(name).is_some() {
            return None;
        }

        // Builtin signature metadata currently includes optional/overloaded forms that are not
        // safe to treat as fixed-arity expansion requests (for example `sum(A(:))`). Keep
        // builtin expansion inference explicit to known variadic/min-input builtins above.
        if runmat_builtins::builtin_function_by_name(name).is_some() {
            return None;
        }
        None
    }

    fn call_argument_can_supply_outputs(&self, arg: &AstExpr, requested: usize) -> bool {
        match arg {
            AstExpr::FuncCall(name, _, _) => {
                if let Some((fixed_outputs, has_varargout)) =
                    self.function_output_signatures.get(name)
                {
                    return *has_varargout || *fixed_outputs >= requested;
                }
                if runmat_builtins::builtin_function_by_name(name).is_some() {
                    return builtin_can_supply_requested_outputs(name, requested);
                }
                // If the callee resolves at runtime (import/path/external), allow the expansion
                // request and let runtime call ABI enforce arity diagnostics.
                true
            }
            _ => false,
        }
    }

    fn lower_call_argument(
        &mut self,
        arg: &AstExpr,
        requested_outputs: RequestedOutputCount,
    ) -> Result<HirExpr, HirError> {
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
    ) -> Result<IndexingSemantics, HirError> {
        Ok(IndexingSemantics {
            kind,
            components: indices
                .iter()
                .map(|expr| {
                    Ok(match expr {
                        AstExpr::Colon(_) => IndexComponent::Colon,
                        _ => {
                            if let Some(offset) = ast_end_relative_offset(expr) {
                                IndexComponent::End { dim: None, offset }
                            } else {
                                IndexComponent::Expr(self.lower_expr_semantic(expr)?)
                            }
                        }
                    })
                })
                .collect::<Result<_, HirError>>()?,
            result_context,
        })
    }

    fn classref_expr(&mut self, class_name: &str, span: Span) -> Result<HirExpr, HirError> {
        let arg = HirExpr {
            id: self.alloc_expr_id(),
            kind: HirExprKind::String(StringLiteral(class_name.to_string())),
            span,
        };
        Ok(HirExpr {
            id: self.alloc_expr_id(),
            kind: HirExprKind::Call(self.call_for_name(
                "classref",
                vec![arg],
                CallSyntax::Plain,
                RequestedOutputCount::One,
                span,
            )?),
            span,
        })
    }

    fn call_for_name(
        &mut self,
        name: &str,
        args: Vec<HirExpr>,
        syntax: CallSyntax,
        requested_outputs: RequestedOutputCount,
        span: Span,
    ) -> Result<HirCall, HirError> {
        let qualified_call_name =
            qualified_name(&name.split('.').map(ToString::to_string).collect::<Vec<_>>());
        let method_like_syntax = matches!(syntax, CallSyntax::Method | CallSyntax::DottedInvoke);
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
        } else if method_like_syntax {
            (
                HirCallableRef::Unresolved(qualified_call_name.clone()),
                CallKind::Dynamic,
            )
        } else if is_builtin(name) {
            let builtin = BuiltinId(name.to_string());
            (
                HirCallableRef::Builtin(builtin.clone()),
                CallKind::Builtin(builtin),
            )
        } else if let Some(def_path) = self.resolve_imported_call_target(name, span)? {
            (
                HirCallableRef::Imported(def_path.clone()),
                CallKind::PackageFunction(def_path),
            )
        } else {
            (
                HirCallableRef::Unresolved(qualified_call_name.clone()),
                CallKind::Dynamic,
            )
        };
        self.hir_index.calls.push(CallResolution {
            name: qualified_call_name,
            callee: callee.clone(),
            kind,
            requested_outputs: requested_outputs.clone(),
            span,
        });
        Ok(HirCall {
            callee,
            args,
            syntax,
            requested_outputs,
        })
    }

    fn resolve_imported_call_target(
        &self,
        name: &str,
        span: Span,
    ) -> Result<Option<DefPath>, HirError> {
        let imports = &self.assembly.modules[self.module.0].imports;
        let specific_matches: Vec<&HirImport> = imports
            .iter()
            .filter(|import| {
                !import.wildcard && import.path.0.last().map(|part| part.0.as_str()) == Some(name)
            })
            .collect();
        if specific_matches.len() == 1 {
            return Ok(Some(def_path_for_import_path(&specific_matches[0].path)));
        }
        if specific_matches.len() > 1 {
            return Err(
                HirError::new(format!("ambiguous call target '{name}' via imports"))
                    .with_span(span)
                    .with_identifier(IDENT_IMPORT_AMBIGUOUS),
            );
        }
        let wildcard_matches: Vec<QualifiedName> = imports
            .iter()
            .filter(|import| import.wildcard)
            .map(|import| Self::wildcard_import_candidate(&import.path, name))
            .filter(|qualified| self.wildcard_candidate_is_resolvable(qualified))
            .collect();
        if wildcard_matches.len() == 1 {
            return Ok(Some(def_path_for_import_path(&wildcard_matches[0])));
        }
        if wildcard_matches.len() > 1 {
            return Err(HirError::new(format!(
                "ambiguous call target '{name}' via wildcard imports"
            ))
            .with_span(span)
            .with_identifier(IDENT_IMPORT_AMBIGUOUS));
        }
        Ok(None)
    }

    fn resolve_function_handle_target(
        &self,
        name: &str,
        span: Span,
    ) -> Result<crate::FunctionHandleTarget, HirError> {
        if let Some(function) = self.function_names.get(name) {
            return Ok(crate::FunctionHandleTarget::Function(*function));
        }
        if let Some(function) = self.external_function_names.get(name) {
            return Ok(crate::FunctionHandleTarget::Function(*function));
        }
        if is_builtin(name) {
            return Ok(crate::FunctionHandleTarget::Builtin(BuiltinId(
                name.to_string(),
            )));
        }
        if name.contains('.') {
            if Self::is_well_formed_qualified_name(name) {
                return Ok(crate::FunctionHandleTarget::DefPath(
                    def_path_for_import_path(&qualified_name(
                        &name.split('.').map(ToString::to_string).collect::<Vec<_>>(),
                    )),
                ));
            }
            return Ok(crate::FunctionHandleTarget::DynamicName(SymbolName(
                name.to_string(),
            )));
        }
        let imports = &self.assembly.modules[self.module.0].imports;
        let specific_matches: Vec<&HirImport> = imports
            .iter()
            .filter(|import| {
                !import.wildcard && import.path.0.last().map(|part| part.0.as_str()) == Some(name)
            })
            .collect();
        if specific_matches.len() > 1 {
            return Err(HirError::new(format!(
                "ambiguous function handle target '{name}' via imports"
            ))
            .with_span(span)
            .with_identifier(IDENT_IMPORT_AMBIGUOUS));
        }
        if let Some(import) = specific_matches.first() {
            return Ok(crate::FunctionHandleTarget::DefPath(
                def_path_for_import_path(&import.path),
            ));
        }
        let wildcard_matches: Vec<QualifiedName> = imports
            .iter()
            .filter(|import| import.wildcard)
            .map(|import| Self::wildcard_import_candidate(&import.path, name))
            .filter(|qualified| self.wildcard_candidate_is_resolvable(qualified))
            .collect();
        if wildcard_matches.len() > 1 {
            return Err(HirError::new(format!(
                "ambiguous function handle target '{name}' via wildcard imports"
            ))
            .with_span(span)
            .with_identifier(IDENT_IMPORT_AMBIGUOUS));
        }
        if let Some(qualified) = wildcard_matches.first().cloned() {
            return Ok(crate::FunctionHandleTarget::DefPath(
                def_path_for_import_path(&qualified),
            ));
        }
        Ok(crate::FunctionHandleTarget::DynamicName(SymbolName(
            name.to_string(),
        )))
    }

    fn unbound_qualified_member_base(&self, expr: &AstExpr) -> Option<String> {
        match expr {
            AstExpr::Ident(name, _) => self.lookup_binding(name).is_none().then(|| name.clone()),
            AstExpr::Member(base, member, _) => self
                .unbound_qualified_member_base(base)
                .map(|prefix| format!("{prefix}.{member}")),
            _ => None,
        }
    }
}

fn validate_semantic_imports(imports: &[HirImport]) -> Result<(), HirError> {
    let mut seen_exact: HashSet<(String, bool)> = HashSet::new();
    for import in imports {
        let path = import
            .path
            .0
            .iter()
            .map(|segment| segment.0.as_str())
            .collect::<Vec<_>>()
            .join(".");
        if !seen_exact.insert((path.clone(), import.wildcard)) {
            return Err(HirError::new(format!(
                "duplicate import '{}{}'",
                path,
                if import.wildcard { ".*" } else { "" }
            ))
            .with_identifier(IDENT_IMPORT_DUPLICATE));
        }
    }

    let mut by_unqualified: HashMap<String, Vec<String>> = HashMap::new();
    for import in imports {
        if import.wildcard {
            continue;
        }
        let segments = &import.path.0;
        let Some(last) = segments.last() else {
            continue;
        };
        let path = segments
            .iter()
            .map(|segment| segment.0.as_str())
            .collect::<Vec<_>>()
            .join(".");
        by_unqualified.entry(last.0.clone()).or_default().push(path);
    }
    for (name, sources) in by_unqualified {
        if sources.len() > 1 {
            return Err(HirError::new(format!(
                "ambiguous import for '{}': {}",
                name,
                sources.join(", ")
            ))
            .with_identifier(IDENT_IMPORT_AMBIGUOUS));
        }
    }
    Ok(())
}

fn def_path_for_import_path(path: &QualifiedName) -> DefPath {
    let package = path
        .0
        .first()
        .map(|segment| segment.0.clone())
        .unwrap_or_default();
    let item_name = path
        .0
        .last()
        .cloned()
        .unwrap_or_else(|| SymbolName(String::new()));
    DefPath {
        package: PackageName(package),
        module: path.clone(),
        item: vec![DefPathSegment::Function(item_name)],
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

fn lvalue_supports_deletion(lvalue: &runmat_parser::LValue) -> bool {
    matches!(lvalue, runmat_parser::LValue::Index(_, _))
}

fn stmt_expr_call_loads_external_bindings(expr: &AstExpr) -> bool {
    let call_name = match expr {
        AstExpr::FuncCall(name, _, _) | AstExpr::CommandCall(name, _, _) => name.as_str(),
        _ => return false,
    };
    runmat_builtins::builtin_function_by_name(call_name).is_some_and(|builtin| {
        matches!(
            builtin.semantics().workspace_effect,
            Some(runmat_builtins::BuiltinWorkspaceEffect::LoadsExternalBindings)
        )
    })
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

fn ast_end_relative_offset(expr: &AstExpr) -> Option<isize> {
    match expr {
        AstExpr::EndKeyword(_) => Some(0),
        AstExpr::Binary(left, BinOp::Add, right, _) => {
            if matches!(&**left, AstExpr::EndKeyword(_)) {
                ast_integer_offset_literal(right)
            } else if matches!(&**right, AstExpr::EndKeyword(_)) {
                ast_integer_offset_literal(left)
            } else {
                None
            }
        }
        AstExpr::Binary(left, BinOp::Sub, right, _)
            if matches!(&**left, AstExpr::EndKeyword(_)) =>
        {
            ast_integer_offset_literal(right).and_then(|offset| offset.checked_neg())
        }
        _ => None,
    }
}

fn ast_integer_offset_literal(expr: &AstExpr) -> Option<isize> {
    let value = static_numeric_literal(expr)?;
    if !value.is_finite() || value.fract() != 0.0 {
        return None;
    }
    if value < isize::MIN as f64 || value > isize::MAX as f64 {
        return None;
    }
    Some(value as isize)
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

fn parse_member_access_value(
    class_name: &str,
    section: &str,
    raw: &str,
) -> Result<crate::MemberAccess, HirError> {
    let normalized = raw.trim().trim_matches('\'').to_ascii_lowercase();
    match normalized.as_str() {
        "public" => Ok(crate::MemberAccess::Public),
        "private" => Ok(crate::MemberAccess::Private),
        other => Err(HirError::new(format!(
            "invalid access value '{other}' in class '{class_name}' {section} (allowed: public, private)"
        ))
        .with_identifier(IDENT_CLASS_ACCESS_VALUE_INVALID)),
    }
}

fn property_attributes(
    class_name: &str,
    attrs: &[runmat_parser::Attr],
) -> Result<crate::PropertyAttributes, HirError> {
    let mut result = crate::PropertyAttributes::default();
    let mut has_static = false;
    let mut has_constant = false;
    let mut has_dependent = false;
    for attr in attrs {
        if attr.name.eq_ignore_ascii_case("Static") {
            result.is_static = true;
            has_static = true;
        } else if attr.name.eq_ignore_ascii_case("Constant") {
            result.is_constant = true;
            has_constant = true;
        } else if attr.name.eq_ignore_ascii_case("Dependent") {
            result.is_dependent = true;
            has_dependent = true;
        } else if attr.name.eq_ignore_ascii_case("Transient") {
            result.is_transient = true;
        } else if attr.name.eq_ignore_ascii_case("Hidden") {
            result.is_hidden = true;
        } else if attr.name.eq_ignore_ascii_case("Access") {
            let raw = attr.value.as_deref().ok_or_else(|| {
                HirError::new(format!(
                    "Access requires value in class '{class_name}' properties block",
                ))
            })?;
            let access = parse_member_access_value(class_name, "properties", raw)?;
            result.access = access.clone();
            result.get_access = access.clone();
            result.set_access = access;
        } else if attr.name.eq_ignore_ascii_case("GetAccess") {
            let raw = attr.value.as_deref().ok_or_else(|| {
                HirError::new(format!(
                    "GetAccess requires value in class '{class_name}' properties block",
                ))
            })?;
            let access = parse_member_access_value(class_name, "properties", raw)?;
            result.get_access = access;
        } else if attr.name.eq_ignore_ascii_case("SetAccess") {
            let raw = attr.value.as_deref().ok_or_else(|| {
                HirError::new(format!(
                    "SetAccess requires value in class '{class_name}' properties block",
                ))
            })?;
            let access = parse_member_access_value(class_name, "properties", raw)?;
            result.set_access = access;
        }
    }
    if has_static && has_dependent {
        return Err(HirError::new(format!(
            "class '{class_name}' properties: attributes 'Static' and 'Dependent' cannot be combined"
        ))
        .with_identifier(IDENT_CLASS_PROPERTY_ATTRIBUTE_CONFLICT));
    }
    if has_constant && has_dependent {
        return Err(HirError::new(format!(
            "class '{class_name}' properties: attributes 'Constant' and 'Dependent' cannot be combined"
        ))
        .with_identifier(IDENT_CLASS_PROPERTY_ATTRIBUTE_CONFLICT));
    }
    Ok(result)
}

fn method_attributes(
    class_name: &str,
    attrs: &[runmat_parser::Attr],
) -> Result<crate::MethodAttributes, HirError> {
    let mut result = crate::MethodAttributes::default();
    let mut has_abstract = false;
    let mut has_sealed = false;
    for attr in attrs {
        if attr.name.eq_ignore_ascii_case("Access") {
            let raw = attr.value.as_deref().ok_or_else(|| {
                HirError::new(format!(
                    "Access requires value in class '{class_name}' methods block",
                ))
            })?;
            let access = parse_member_access_value(class_name, "methods", raw)?;
            result.access = access;
        } else if attr.name.eq_ignore_ascii_case("Hidden") {
            result.is_hidden = true;
        } else if attr.name.eq_ignore_ascii_case("Abstract") {
            result.is_abstract = true;
            has_abstract = true;
        } else if attr.name.eq_ignore_ascii_case("Sealed") {
            result.is_sealed = true;
            has_sealed = true;
        }
    }
    if has_abstract && has_sealed {
        return Err(HirError::new(format!(
            "class '{class_name}' methods: attributes 'Abstract' and 'Sealed' cannot be combined"
        ))
        .with_identifier(IDENT_CLASS_METHOD_ATTRIBUTE_CONFLICT));
    }
    Ok(result)
}

fn is_builtin(name: &str) -> bool {
    if matches!(name, NARGIN_BUILTIN_NAME | NARGOUT_BUILTIN_NAME) {
        return true;
    }
    runmat_builtins::builtin_functions()
        .iter()
        .any(|builtin| builtin.name == name)
        || runmat_builtins::builtin_semantics_for_name(name).is_some()
}

fn builtin_min_inputs_for_last_arg_expansion(name: &str) -> Option<usize> {
    match name {
        // These call sites support "missing-argument filled by last argument expansion"
        // when the final argument can supply a comma-list.
        "max" | "min" | "cat" | "atan2" => Some(2),
        _ => None,
    }
}

fn builtin_can_supply_requested_outputs(name: &str, requested: usize) -> bool {
    if requested <= 1 {
        return true;
    }
    matches!(
        name,
        "size"
            | "max"
            | "min"
            | "sort"
            | "unique"
            | "find"
            | "union"
            | "ismember"
            | "sortrows"
            | "chol"
            | "lu"
            | "qr"
            | "svd"
            | "eig"
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
