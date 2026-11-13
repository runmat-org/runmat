use crate::ir::{NativeAssembly, NativeRequirements};
use crate::{NativeCodegenError, NativeCodegenResult, NativeTarget, NATIVE_IR_SCHEMA_VERSION};

#[path = "assembly/analysis.rs"]
mod analysis;

pub use analysis::verify_against_analysis;

pub struct NativeLoweringInput<'a> {
    pub mir: &'a runmat_mir::MirAssembly,
    pub analysis: &'a runmat_mir::analysis::AnalysisStore,
    pub manifest: &'a runmat_execution::ExecutableUnitManifest,
    /// Selected compiler/MIR identity. The manifest retains the immutable
    /// portable identity of the same function.
    pub entrypoint: NativeEntrypointBinding,
    /// Canonical names for semantic HIR bindings retained by MIR locals.
    /// May be absent only when the supplied MIR contains no bound locals.
    pub binding_names: Option<&'a std::collections::BTreeMap<runmat_types::BindingId, String>>,
    pub target: NativeTarget,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeEntrypointBinding {
    compiled: runmat_types::FunctionId,
    portable: runmat_types::ProgramFunctionId,
    body_digest: runmat_execution::Digest,
}

impl NativeEntrypointBinding {
    pub fn new(
        mir: &runmat_mir::MirAssembly,
        compiled: runmat_hir::FunctionId,
        portable: runmat_types::ProgramFunctionId,
    ) -> NativeCodegenResult<Self> {
        let body_digest = entrypoint_body_digest(mir, compiled)?;
        Ok(Self {
            compiled,
            portable,
            body_digest,
        })
    }

    pub fn compiled(&self) -> runmat_types::FunctionId {
        self.compiled
    }

    pub fn portable(&self) -> runmat_types::ProgramFunctionId {
        self.portable
    }

    fn validate(
        &self,
        mir: &runmat_mir::MirAssembly,
        manifest: &runmat_execution::ExecutableUnitManifest,
    ) -> NativeCodegenResult<()> {
        if Some(self.portable) != manifest.identity.entrypoint_function
            || self.body_digest != entrypoint_body_digest(mir, self.compiled)?
        {
            return Err(NativeCodegenError::new(
                "native.lowering.entrypoint_binding",
                "selected native entrypoint does not match its admitted portable body",
            ));
        }
        Ok(())
    }
}

fn entrypoint_body_digest(
    mir: &runmat_mir::MirAssembly,
    function: runmat_types::FunctionId,
) -> NativeCodegenResult<runmat_execution::Digest> {
    let metadata = mir.functions.get(&function).ok_or_else(|| {
        NativeCodegenError::new(
            "native.lowering.entrypoint_identity",
            "selected native entrypoint has no canonical MIR metadata",
        )
    })?;
    let body = mir.bodies.get(&function).ok_or_else(|| {
        NativeCodegenError::new(
            "native.lowering.entrypoint_identity",
            "selected native entrypoint is absent from canonical MIR",
        )
    })?;
    let bytes = serde_json::to_vec(&(metadata, body)).map_err(|error| {
        NativeCodegenError::new("native.lowering.entrypoint_binding", error.to_string())
    })?;
    Ok(runmat_execution::Digest::sha256(&bytes))
}

pub fn lower_executable(input: NativeLoweringInput<'_>) -> NativeCodegenResult<NativeAssembly> {
    input
        .manifest
        .validate()
        .map_err(|error| NativeCodegenError::new("native.lowering.manifest", error.to_string()))?;
    input
        .manifest
        .identity
        .program
        .validate_current_compiler()
        .map_err(|error| {
            NativeCodegenError::new("native.lowering.compiler_compatibility", error.to_string())
        })?;
    input
        .analysis
        .revision
        .validate_current()
        .map_err(|error| {
            NativeCodegenError::new("native.lowering.analysis_revision", error.to_string())
        })?;
    input.target.validate()?;
    if input.manifest.revisions.analysis_schema != input.analysis.revision.schema_version {
        return Err(NativeCodegenError::new(
            "native.lowering.analysis_schema",
            "manifest analysis schema does not match the supplied AnalysisStore",
        ));
    }
    if input.manifest.revisions.mir_schema != runmat_mir::MIR_SCHEMA_VERSION {
        return Err(NativeCodegenError::new(
            "native.lowering.mir_schema",
            format!(
                "Native IR received MIR schema {}; expected {}. Rebuild the program with this RunMat version",
                input.manifest.revisions.mir_schema,
                runmat_mir::MIR_SCHEMA_VERSION
            ),
        ));
    }
    let analysis_component = analysis::verify_analysis_component(input.analysis, input.manifest)?;
    let analysis = analysis::project_analysis(input.analysis, input.mir)?;
    super::requirements::validate_requirements(input.mir, input.manifest)?;
    super::requirements::reject_predeclared_capabilities(input.mir, &analysis)?;

    let mut expected_operation_sites = Vec::new();
    for (function, body) in &input.mir.bodies {
        let portable = analysis
            .function_for_local(*function)
            .ok_or_else(|| {
                NativeCodegenError::new(
                    "native.lowering.operation_contract_coverage",
                    "MIR body has no canonical analysis function owner",
                )
                .at_function(*function)
            })?
            .function;
        expected_operation_sites.extend(super::inventory::expected_sites(portable, body)?);
    }
    expected_operation_sites.sort();
    if analysis
        .operation_contracts
        .iter()
        .map(|contract| contract.site.clone())
        .ne(expected_operation_sites)
    {
        return Err(NativeCodegenError::new(
            "native.lowering.operation_contract_coverage",
            "AnalysisStore operation contracts are missing, duplicated, or orphaned from MIR",
        ));
    }

    let mut functions = Vec::with_capacity(input.mir.bodies.len());
    for (function, body) in &input.mir.bodies {
        let function_id = *function;
        let metadata = input.mir.functions.get(function).ok_or_else(|| {
            NativeCodegenError::new(
                "native.lowering.function_metadata",
                "MIR body has no immutable function metadata",
            )
            .at_function(function_id)
        })?;
        let retained_callable_parent =
            input
                .mir
                .retained_callable_parent(*function)
                .map_err(|error| {
                    NativeCodegenError::new("native.lowering.lexical_parent", error)
                        .at_function(function_id)
                })?;
        functions.push(super::function::lower_function(
            function_id,
            entrypoint_body_digest(input.mir, function_id)?,
            &input,
            &analysis,
            metadata,
            retained_callable_parent,
            body,
        )?);
    }
    functions.sort_by_key(|function| function.id);

    input.entrypoint.validate(input.mir, input.manifest)?;
    let entrypoint = input.entrypoint.compiled();
    if !input.mir.bodies.contains_key(&entrypoint) {
        return Err(NativeCodegenError::new(
            "native.lowering.entrypoint_identity",
            "selected native entrypoint is absent from canonical MIR",
        ));
    }
    let mut entrypoints = input.mir.entrypoints.to_vec();
    entrypoints.push(input.entrypoint.compiled());
    entrypoints.sort_unstable();
    entrypoints.dedup();

    let executable_cache_key = input
        .manifest
        .cache_key()
        .map_err(|error| NativeCodegenError::new("native.lowering.cache_key", error.to_string()))?;
    let native_cache_key = input.target.cache_key(
        &executable_cache_key,
        input.entrypoint.compiled(),
        &input.entrypoint.body_digest,
    )?;
    let analysis_projection = analysis::analysis_digest(&analysis)?;
    let assembly = NativeAssembly {
        schema_version: NATIVE_IR_SCHEMA_VERSION,
        executable_identity: input.manifest.identity.clone(),
        program: input.manifest.identity.program.clone(),
        executable_cache_key,
        native_cache_key,
        analysis_component,
        analysis_projection,
        target: input.target,
        requirements: NativeRequirements {
            capabilities: input.manifest.capabilities.clone(),
            regions: input.manifest.regions.clone(),
            interop: input.manifest.interop.clone(),
            parallel: input.manifest.parallel.clone(),
        },
        selected_entrypoint: input.entrypoint.compiled(),
        selected_portable_entrypoint: input.entrypoint.portable(),
        selected_entrypoint_body: input.entrypoint.body_digest,
        entrypoints,
        functions,
        class_declarations: super::super_constructor::lower(input.mir)?,
    };
    assembly.verify()?;
    verify_against_manifest(&assembly, input.manifest)?;
    verify_against_mir(&assembly, input.mir, input.binding_names)?;
    Ok(assembly)
}

/// Binds a decoded Native assembly back to the complete executable manifest.
pub fn verify_against_manifest(
    assembly: &NativeAssembly,
    manifest: &runmat_execution::ExecutableUnitManifest,
) -> NativeCodegenResult<()> {
    manifest
        .validate()
        .map_err(|error| NativeCodegenError::new("native.ir.manifest", error.to_string()))?;
    let cache_key = manifest
        .cache_key()
        .map_err(|error| NativeCodegenError::new("native.ir.manifest", error.to_string()))?;
    if assembly.executable_identity != manifest.identity
        || assembly.executable_cache_key != cache_key
        || assembly.requirements.capabilities != manifest.capabilities
        || assembly.requirements.regions != manifest.regions
        || assembly.requirements.interop != manifest.interop
        || assembly.requirements.parallel != manifest.parallel
        || manifest
            .components
            .iter()
            .find(|component| component.kind == runmat_execution::ExecutableComponentKind::Analysis)
            .is_none_or(|component| component.digest != assembly.analysis_component)
    {
        return Err(NativeCodegenError::new(
            "native.ir.manifest_binding",
            "Native IR does not exactly match its executable manifest",
        ));
    }
    Ok(())
}

/// Proves that a structurally valid Native assembly is a complete lowering of
/// the supplied canonical MIR. Artifact/cache readers must run this check when
/// the MIR component is available; structural verification alone cannot infer
/// constructs that an untrusted producer removed from both IR and inventory.
pub fn verify_against_mir(
    assembly: &NativeAssembly,
    mir: &runmat_mir::MirAssembly,
    binding_names: Option<&std::collections::BTreeMap<runmat_types::BindingId, String>>,
) -> NativeCodegenResult<()> {
    if assembly.class_declarations != super::super_constructor::lower(mir)? {
        return Err(NativeCodegenError::new(
            "native.ir.mir_class_declarations",
            "Native IR class declarations differ from canonical MIR",
        ));
    }
    if assembly.selected_entrypoint_body
        != entrypoint_body_digest(mir, assembly.selected_entrypoint)?
    {
        return Err(NativeCodegenError::new(
            "native.ir.entrypoint_binding",
            "selected native entrypoint body differs from canonical MIR",
        ));
    }
    let functions = assembly
        .functions
        .iter()
        .map(|function| (function.id, function))
        .collect::<std::collections::BTreeMap<_, _>>();
    if functions.len() != mir.bodies.len() {
        return Err(NativeCodegenError::new(
            "native.ir.mir_functions",
            "Native IR function set differs from canonical MIR",
        ));
    }
    for (function, body) in &mir.bodies {
        let function_id = *function;
        let native = functions.get(&function_id).ok_or_else(|| {
            NativeCodegenError::new(
                "native.ir.mir_functions",
                "canonical MIR function is absent from Native IR",
            )
            .at_function(function_id)
        })?;
        if native.mir_body_digest != entrypoint_body_digest(mir, function_id)? {
            return Err(NativeCodegenError::new(
                "native.ir.mir_body_digest",
                "Native IR function body identity differs from canonical MIR",
            )
            .at_function(function_id));
        }
        if Some(&native.spawn_safety)
            != runmat_mir::analysis::function_spawn_safety(mir, function_id).as_ref()
        {
            return Err(NativeCodegenError::new(
                "native.ir.mir_spawn_contract",
                "Native IR spawn policy differs from the canonical MIR definition",
            )
            .at_function(function_id));
        }
        if native.locals.len() != body.locals.len() {
            return Err(NativeCodegenError::new(
                "native.ir.mir_locals",
                "Native IR local catalog differs from canonical MIR",
            )
            .at_function(function_id));
        }
        for (native_local, mir_local) in native.locals.iter().zip(&body.locals) {
            let expected_id = u32::try_from(mir_local.id.0)
                .map(crate::NativeLocalId)
                .map_err(|_| {
                    NativeCodegenError::new(
                        "native.ir.mir_local_identity",
                        "MIR local identity exceeds the Native IR schema",
                    )
                    .at_function(function_id)
                })?;
            let expected_name = mir_local
                .binding
                .map(|binding| {
                    binding_names
                        .and_then(|names| names.get(&binding))
                        .cloned()
                        .ok_or_else(|| {
                            NativeCodegenError::new(
                                "native.ir.mir_binding_name",
                                "bound MIR local has no canonical semantic name",
                            )
                            .at_function(function_id)
                        })
                })
                .transpose()?;
            if native_local.id != expected_id
                || native_local.binding != mir_local.binding
                || native_local.name != expected_name
                || native_local.kind != crate::NativeLocalKind::from(&mir_local.kind)
            {
                return Err(NativeCodegenError::new(
                    "native.ir.mir_locals",
                    "Native IR local metadata differs from canonical MIR and binding metadata",
                )
                .at_function(function_id));
            }
        }
        let metadata = mir.functions.get(function).ok_or_else(|| {
            NativeCodegenError::new(
                "native.ir.mir_function_metadata",
                "canonical MIR body has no immutable function metadata",
            )
            .at_function(function_id)
        })?;
        let expected_abi = super::function::lower_abi(body, function_id)?;
        if native.abi != expected_abi {
            return Err(NativeCodegenError::new(
                "native.ir.mir_function_abi",
                "Native IR function ABI differs from canonical MIR",
            )
            .at_function(function_id));
        }
        let expected_validations = super::function::lower_argument_validations(
            metadata,
            &native.locals,
            &native.abi,
            function_id,
        )?;
        if native.argument_validations != expected_validations {
            return Err(NativeCodegenError::new(
                "native.ir.mir_argument_validations",
                "Native IR argument-validation metadata differs from canonical MIR",
            )
            .at_function(function_id));
        }
        let expected = super::inventory::expected_sites(native.portable_id, body)?;
        if native.expected_sites != expected {
            return Err(NativeCodegenError::new(
                "native.ir.mir_construct_coverage",
                "Native IR inventory differs from canonical MIR",
            )
            .at_function(function_id));
        }
        verify_statement_operations(native, body, function_id)?;
    }
    Ok(())
}

fn verify_statement_operations(
    native: &crate::NativeFunction,
    body: &runmat_mir::MirBody,
    function: runmat_types::FunctionId,
) -> NativeCodegenResult<()> {
    for block in &body.blocks {
        let block_id = u32::try_from(block.id.0)
            .map(crate::NativeBlockId)
            .map_err(|_| {
                NativeCodegenError::new(
                    "native.ir.mir_operation",
                    "MIR block identity exceeds the Native IR schema",
                )
                .at_function(function)
            })?;
        let native_block = native
            .blocks
            .iter()
            .find(|candidate| candidate.id == block_id)
            .ok_or_else(|| {
                NativeCodegenError::new(
                    "native.ir.mir_operation",
                    "canonical MIR block is absent from Native IR",
                )
                .at_function(function)
            })?;
        for (position, statement) in block.statements.iter().enumerate() {
            let position = u32::try_from(position).map_err(|_| {
                NativeCodegenError::new(
                    "native.ir.mir_operation",
                    "MIR statement position exceeds the Native IR schema",
                )
                .at_function(function)
            })?;
            let point = runmat_types::ProgramPointId {
                function: native.portable_id,
                block: block_id.0,
                position,
            };
            let statement_instruction = native_block.instructions.iter().find(|instruction| {
                instruction.site.point == point
                    && instruction.site.phase == crate::NativeSitePhase::Statement
                    && instruction.site.ordinal == 0
            });
            if !matches!(
                statement_instruction.map(|instruction| &instruction.operation),
                Some(crate::NativeOperation::Statement(actual)) if actual == &statement.kind
            ) {
                return Err(NativeCodegenError::new(
                    "native.ir.mir_operation",
                    "Native IR statement payload differs from canonical MIR",
                )
                .at_point(point));
            }
            let expected_rvalue = match &statement.kind {
                runmat_mir::MirStmtKind::PreparedAssign { value, .. }
                | runmat_mir::MirStmtKind::Assign { value, .. }
                | runmat_mir::MirStmtKind::MultiAssign { value, .. }
                | runmat_mir::MirStmtKind::SequenceAssign { value, .. }
                | runmat_mir::MirStmtKind::Expr(value) => Some(value),
                runmat_mir::MirStmtKind::PrepareFixedAssignment(_)
                | runmat_mir::MirStmtKind::PlaceMutation(_)
                | runmat_mir::MirStmtKind::StatementOutput(_)
                | runmat_mir::MirStmtKind::CaptureSequence { .. }
                | runmat_mir::MirStmtKind::WorkspaceEffect { .. }
                | runmat_mir::MirStmtKind::EnvironmentEffect(_) => None,
            };
            if let Some(expected_rvalue) = expected_rvalue {
                let rvalue_instruction = native_block.instructions.iter().find(|instruction| {
                    instruction.site.point == point
                        && instruction.site.phase == crate::NativeSitePhase::Rvalue
                        && instruction.site.ordinal == 0
                });
                if !matches!(
                    rvalue_instruction.map(|instruction| &instruction.operation),
                    Some(crate::NativeOperation::Rvalue { value, .. }) if value == expected_rvalue
                ) {
                    return Err(NativeCodegenError::new(
                        "native.ir.mir_operation",
                        "Native IR rvalue payload differs from canonical MIR",
                    )
                    .at_point(point));
                }
            }
        }
        let terminator_point = runmat_types::ProgramPointId {
            function: native.portable_id,
            block: block_id.0,
            position: u32::try_from(block.statements.len()).map_err(|_| {
                NativeCodegenError::new(
                    "native.ir.mir_operation",
                    "MIR terminator position exceeds the Native IR schema",
                )
                .at_function(function)
            })?,
        };
        let expected_values = runmat_mir::terminator_rvalues(&block.terminator.kind);
        let actual_values = native_block
            .instructions
            .iter()
            .filter(|instruction| {
                instruction.site.point == terminator_point
                    && instruction.site.phase == crate::NativeSitePhase::TerminatorRvalue
            })
            .map(|instruction| match &instruction.operation {
                crate::NativeOperation::Rvalue { value, .. } => Some((instruction, value)),
                crate::NativeOperation::Statement(_) => None,
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                NativeCodegenError::new(
                    "native.ir.mir_terminator_operation",
                    "terminator rvalue site contains a statement payload",
                )
                .at_point(terminator_point)
            })?;
        if actual_values.len() != expected_values.len()
            || actual_values.iter().zip(&expected_values).enumerate().any(
                |(ordinal, ((instruction, actual), expected))| {
                    instruction.site.ordinal as usize != ordinal || *actual != expected
                },
            )
        {
            return Err(NativeCodegenError::new(
                "native.ir.mir_terminator_operation",
                "Native IR terminator operations differ from canonical MIR evaluation order",
            )
            .at_point(terminator_point));
        }
        if native_block.terminator.site.point != terminator_point
            || native_block.terminator.site.phase != crate::NativeSitePhase::Terminator
            || native_block.terminator.site.ordinal != 0
            || native_block.terminator.site.construct
                != runmat_mir::terminator_construct_kind(&block.terminator.kind)
        {
            return Err(NativeCodegenError::new(
                "native.ir.mir_terminator_operation",
                "Native IR terminator site differs from canonical MIR",
            )
            .at_point(terminator_point));
        }
    }
    Ok(())
}
