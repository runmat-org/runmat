use crate::ir::{NativeAssembly, NativeRequirements};
use crate::{NativeCodegenError, NativeCodegenResult, NativeTarget, NATIVE_IR_SCHEMA_VERSION};

pub struct NativeLoweringInput<'a> {
    pub mir: &'a runmat_mir::MirAssembly,
    pub analysis: &'a runmat_mir::analysis::AnalysisStore,
    pub manifest: &'a runmat_execution::ExecutableUnitManifest,
    /// Canonical names for semantic HIR bindings retained by MIR locals.
    /// May be absent only when the supplied MIR contains no bound locals.
    pub binding_names: Option<&'a std::collections::BTreeMap<runmat_types::BindingId, String>>,
    pub target: NativeTarget,
}

pub fn lower_executable(input: NativeLoweringInput<'_>) -> NativeCodegenResult<NativeAssembly> {
    input
        .manifest
        .validate()
        .map_err(|error| NativeCodegenError::new("native.lowering.manifest", error.to_string()))?;
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
            "Native IR requires MIR schema 2 with retained source identities",
        ));
    }
    super::requirements::validate_requirements(input.mir, input.manifest)?;
    super::requirements::reject_predeclared_capabilities(input.mir)?;

    let mut functions = Vec::with_capacity(input.mir.bodies.len());
    for (function, body) in &input.mir.bodies {
        let function_id = u32::try_from(function.0)
            .map(runmat_types::ProgramFunctionId)
            .map_err(|_| {
                NativeCodegenError::new(
                    "native.lowering.function_identity",
                    "MIR function identity exceeds the Native IR schema",
                )
            })?;
        let metadata = input.mir.functions.get(function).ok_or_else(|| {
            NativeCodegenError::new(
                "native.lowering.function_metadata",
                "MIR body has no immutable function metadata",
            )
            .at_function(function_id)
        })?;
        functions.push(super::function::lower_function(
            function_id,
            metadata,
            body,
            input.analysis,
            input.binding_names,
            &input.manifest.regions,
        )?);
    }
    functions.sort_by_key(|function| function.id);

    let mut entrypoints = input
        .mir
        .entrypoints
        .iter()
        .map(|function| {
            u32::try_from(function.0)
                .map(runmat_types::ProgramFunctionId)
                .map_err(|_| {
                    NativeCodegenError::new(
                        "native.lowering.entrypoint_identity",
                        "MIR entrypoint identity exceeds the Native IR schema",
                    )
                })
        })
        .collect::<NativeCodegenResult<Vec<_>>>()?;
    if !entrypoints.contains(&input.manifest.identity.entrypoint_function) {
        entrypoints.push(input.manifest.identity.entrypoint_function);
    }
    entrypoints.sort();
    entrypoints.dedup();

    let executable_cache_key = input
        .manifest
        .cache_key()
        .map_err(|error| NativeCodegenError::new("native.lowering.cache_key", error.to_string()))?;
    let native_cache_key = input.target.cache_key(&executable_cache_key)?;
    let assembly = NativeAssembly {
        schema_version: NATIVE_IR_SCHEMA_VERSION,
        executable_identity: input.manifest.identity.clone(),
        program: input.manifest.identity.program.clone(),
        executable_cache_key,
        native_cache_key,
        target: input.target,
        requirements: NativeRequirements {
            capabilities: input.manifest.capabilities.clone(),
            regions: input.manifest.regions.clone(),
            interop: input.manifest.interop.clone(),
            parallel: input.manifest.parallel.clone(),
        },
        entrypoints,
        functions,
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
        let function_id = u32::try_from(function.0)
            .map(runmat_types::ProgramFunctionId)
            .map_err(|_| {
                NativeCodegenError::new(
                    "native.ir.mir_function_identity",
                    "MIR function identity exceeds the Native IR schema",
                )
            })?;
        let native = functions.get(&function_id).ok_or_else(|| {
            NativeCodegenError::new(
                "native.ir.mir_functions",
                "canonical MIR function is absent from Native IR",
            )
            .at_function(function_id)
        })?;
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
        let expected = super::inventory::expected_sites(function_id, body)?;
        if native.expected_sites != expected {
            return Err(NativeCodegenError::new(
                "native.ir.mir_construct_coverage",
                "Native IR inventory differs from canonical MIR",
            )
            .at_function(function_id));
        }
    }
    Ok(())
}
