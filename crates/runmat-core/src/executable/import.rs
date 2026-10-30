//! Restore the compiler-owned product; never substitute a bare VM registry for a session unit.
use runmat_execution::{
    ExecutableComponentKind as Kind, ExecutableEntrypointKind, ExecutableUnitEnvelope,
};
use serde::de::DeserializeOwned;

use super::{ExecutableRevision, ExecutableUnit};
mod dependencies;
mod sources;
#[cfg(test)]
mod tests;

impl ExecutableUnit {
    /// Decode a current compiler product with its original source, MIR, analysis,
    /// layout, function inventory, coverage and extension metadata. This checks
    /// schema and component consistency; it does not recompile or attest source semantics.
    pub fn from_portable_envelope(envelope: &ExecutableUnitEnvelope) -> Result<Self, String> {
        envelope.validate().map_err(|error| error.to_string())?;
        let unit = dependencies::restore(envelope)?;
        let restored = match envelope.manifest.identity.entrypoint_kind {
            ExecutableEntrypointKind::Function => {
                unit.portable_envelope_for_selection(Some(imported_function(&unit, envelope)?))?
            }
            ExecutableEntrypointKind::Script => unit.portable_envelope_for_selection(None)?,
            ExecutableEntrypointKind::DeclarationOnly => {
                return Err("declaration-only products are not imported for invocation".into())
            }
        };
        if restored != *envelope {
            return Err("portable dependency closure differs from original inventory".into());
        }
        Ok(unit)
    }

    fn decode_local_product(
        envelope: &ExecutableUnitEnvelope,
        mut compiler_registry: runmat_vm::FunctionRegistry,
        owners: &dependencies::OriginalOwners,
    ) -> Result<Self, String> {
        envelope
            .manifest
            .validate_for(
                &runmat_execution::ExecutableSectionSupport::known([
                    runmat_execution::executable::KnownExecutableSection::StaticImports,
                ])
                .map_err(|error| error.to_string())?,
            )
            .map_err(|error| error.to_string())?;
        require_environment(envelope)?;
        let compiler_bytecode: runmat_vm::Bytecode = decode(envelope, Kind::Bytecode)?;
        let (source, source_map) = sources::decode(envelope, &compiler_bytecode)?;
        let program = envelope
            .manifest
            .identity
            .program
            .identity_digest()
            .map_err(|error| error.to_string())?;
        let mut originals = std::collections::HashMap::new();
        let mut original_owners = std::collections::HashMap::new();
        for (id, projection) in &compiler_registry.functions {
            let origin = projection
                .binding_origin
                .ok_or("compiler function projection lacks its original binding origin")?;
            if *id != projection.function || origin.function != projection.portable_function {
                return Err(
                    "compiler function projection has inconsistent original coordinates".into(),
                );
            }
            if origin.program == program {
                continue;
            }
            let owner = owners
                .get(&origin)
                .ok_or("foreign function projection has no admitted original product")?;
            let original =
                super::dependencies::original_projection(&source_map, owner, projection)?;
            originals.insert(*id, original);
            original_owners.insert(*id, std::rc::Rc::clone(owner));
        }
        let mut executable_registry = compiler_registry.clone();
        for id in originals.keys() {
            executable_registry.remove(*id);
        }
        // Foreign projections are compiler inventory, not executable bodies
        // of this program. Each was checked above against its original admitted
        // owner (including its instructions). Admit every current instruction
        // and import site through the ordinary strict worker boundary.
        let executable = current_product(envelope, &executable_registry)?;
        let canonical = executable
            .canonical_bytes()
            .map_err(|error| error.to_string())?;
        let admitted =
            runmat_vm::admit_executable_unit(&canonical).map_err(|error| error.to_string())?;
        let imports = admitted.static_imports().to_vec();
        let current = admitted.registry().functions.keys().copied().collect();
        let (mut bytecode, registry, layout) = admitted.into_compiler_components();
        // Only admitted current bodies and independently validated original
        // bodies enter the compiler registry. Never execute deserialized
        // projection flags as if they were MIR-issued evidence.
        originals.extend(registry.into_functions());
        let names = std::mem::take(&mut compiler_registry.names);
        let mut registry = runmat_vm::FunctionRegistry::new(originals);
        registry.names = names;
        drop(canonical);
        let mir: runmat_mir::MirAssembly = decode(envelope, Kind::Mir)?;
        let analysis: runmat_mir::analysis::AnalysisStore = decode(envelope, Kind::Analysis)?;
        analysis
            .revision
            .validate_current()
            .map_err(|error| error.to_string())?;
        let revision = ExecutableRevision::derive(
            &source,
            Some(envelope.manifest.identity.program.clone()),
            envelope.manifest.identity.program.environment(),
        );
        if revision
            .source_digest
            .parse::<runmat_execution::Digest>()
            .map_err(|error| error.to_string())?
            != envelope.manifest.identity.source_digest
        {
            return Err("portable primary source differs from its original identity".into());
        }
        bytecode.bound_functions = registry.functions.clone();
        bytecode.function_registry = registry;
        bytecode.layout = Some(layout);
        let mut unit = Self::new(source, revision, source_map, mir, analysis, bytecode)?
            .bind_original_dependencies(&original_owners)?;
        // A selected product can omit an inherited projection altogether.
        // Its admitted site still selects an exact original product by origin,
        // never by the numeric operand that happens to occupy that site.
        for import in &imports {
            let owner = owners
                .get(&import.target)
                .ok_or("static import has no admitted original compiler product")?;
            let function = owner
                .local_function_for_origin(import.target)?
                .ok_or("static import target is absent from its original compiler product")?;
            let code = original_code(owner)?;
            if code.content_digest() != import.target_content
                || code.body_digest(import.target) != Some(import.target_body)
            {
                return Err("static import differs from its original admitted content/body".into());
            }
            if let Some(previous) = unit.original_dependencies.insert(function, owner.clone()) {
                if !std::rc::Rc::ptr_eq(&previous, owner) {
                    return Err(
                        "static imports have conflicting original compiler coordinates".into(),
                    );
                }
            }
        }
        let mut unit = unit.with_admitted_portable_selection(current, imports)?;
        unit.reset_direct_code_cache();
        unit.imported_contract = Some(std::rc::Rc::new(super::unit::ImportedPortableContract {
            interop: envelope.manifest.interop.clone(),
            parallel: envelope.manifest.parallel.clone(),
            optional_sections: envelope.manifest.optional_sections.clone(),
        }));
        let selected = match envelope.manifest.identity.entrypoint_kind {
            ExecutableEntrypointKind::Function => Some(imported_function(&unit, envelope)?),
            ExecutableEntrypointKind::Script => None,
            ExecutableEntrypointKind::DeclarationOnly => {
                return Err("declaration-only products are not imported for invocation".into())
            }
        };
        let restored = unit.portable_local_envelope_for_selection(selected)?;
        let mut local_manifest = envelope.manifest.clone();
        local_manifest.dependencies.clear();
        if restored.manifest != local_manifest || restored.payloads != envelope.payloads {
            return Err("portable product differs from canonical source, coverage, region or component inventory".into());
        }
        Ok(unit)
    }
}

fn original_code(
    unit: &ExecutableUnit,
) -> Result<std::rc::Rc<runmat_vm::SelectedExecutableCode>, String> {
    let interop = unit
        .imported_contract
        .as_ref()
        .ok_or("original dependency has no admitted compiler contract")?
        .interop
        .clone();
    unit.admit_direct_code(interop.clone(), || {
        let envelope = unit.portable_primary_source_envelope_with_static_imports(
            &unit.compiled_current_body_ids()?,
            Some(unit.primary_source_id()?),
            interop,
            unit.original_static_imports().to_vec(),
        )?;
        envelope
            .canonical_bytes()
            .map(std::sync::Arc::<[u8]>::from)
            .map_err(|error| error.to_string())
    })
}

/// Produce the existing worker view of one complete compiler product. The
/// caller has already validated every excluded projection against an admitted
/// original owner; all executable references remain in this strict VM view.
fn current_product(
    envelope: &ExecutableUnitEnvelope,
    registry: &runmat_vm::FunctionRegistry,
) -> Result<ExecutableUnitEnvelope, String> {
    let mut manifest = envelope.manifest.clone();
    manifest.dependencies.clear();
    let mut payloads = envelope.payloads.clone();
    let payload = payloads
        .iter_mut()
        .find(|payload| payload.kind == Kind::FunctionRegistry)
        .ok_or("portable product lacks its function registry")?;
    payload.bytes = super::portable::canonical_json(registry)?;
    let descriptor = manifest
        .components
        .iter_mut()
        .find(|descriptor| descriptor.kind == Kind::FunctionRegistry)
        .ok_or("portable product lacks its registry descriptor")?;
    *descriptor = runmat_execution::ExecutableComponentDescriptor::from_payload(
        Kind::FunctionRegistry,
        descriptor.schema_version,
        &payload.bytes,
    )
    .map_err(|error| error.to_string())?;
    ExecutableUnitEnvelope::new(manifest, payloads).map_err(|error| error.to_string())
}

/// The manifest carries an original portable ordinal, not a compiler/session
/// function coordinate. Resolve it through the immutable binding origin;
/// entrypoint text is descriptive and cannot make a nested body name-callable.
fn imported_function(
    unit: &ExecutableUnit,
    envelope: &ExecutableUnitEnvelope,
) -> Result<(String, runmat_hir::FunctionId), String> {
    let identity = &envelope.manifest.identity;
    let expected = identity
        .entrypoint_function
        .ok_or("portable function product has no entrypoint function")?;
    let origin = runmat_execution::FunctionBindingOrigin::from_revision(
        &unit.revision().program_revision,
        expected,
    )
    .map_err(|error| error.to_string())?;
    let compiled = unit
        .local_function_for_origin(origin)?
        .ok_or("portable function product has no exact original body")?;
    let selected = super::portable_entrypoint::resolve_compiled_function_with_label(
        unit,
        compiled,
        &identity.entrypoint,
    )?;
    match selected {
        super::portable_entrypoint::PortableEntrypoint::Callable {
            name,
            compiled_function,
            function,
            kind: ExecutableEntrypointKind::Function,
        } if function == expected && name == identity.entrypoint => Ok((name, compiled_function)),
        _ => Err("portable entrypoint differs from its original function binding".into()),
    }
}

fn decode<T: DeserializeOwned>(envelope: &ExecutableUnitEnvelope, kind: Kind) -> Result<T, String> {
    let bytes = &envelope
        .component(kind)
        .ok_or_else(|| format!("portable product lacks {kind:?}"))?
        .bytes;
    serde_json::from_slice(bytes).map_err(|error| format!("invalid portable {kind:?}: {error}"))
}

fn require_environment(envelope: &ExecutableUnitEnvelope) -> Result<(), String> {
    let original = envelope.manifest.identity.program.environment();
    let mode = match original.compatibility_mode.as_str() {
        "matlab" => crate::CompatMode::Matlab,
        "runmat" => crate::CompatMode::RunMat,
        "strict" => crate::CompatMode::Strict,
        _ => return Err("portable language mode is unsupported".into()),
    };
    if original != crate::program_environment(mode) {
        return Err("portable compiler/runtime environment differs from this consumer".into());
    }
    Ok(())
}
