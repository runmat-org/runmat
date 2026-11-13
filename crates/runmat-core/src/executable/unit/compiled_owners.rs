use std::collections::BTreeSet;

/// Assembly owners are compiled here; registered bodies are also owned by
/// this immutable bytecode product. A synthetic script entrypoint may share
/// a numeric ID with an inherited selected callable without owning its body.
#[derive(Debug)]
pub(super) struct CompiledProductOwners {
    pub(super) assembly: BTreeSet<runmat_hir::FunctionId>,
    pub(super) registered_bodies: BTreeSet<runmat_hir::FunctionId>,
}

impl CompiledProductOwners {
    /// The only publication bridge between compiler coordinates and the
    /// current immutable revision. Inherited registry bodies are deliberately
    /// absent, even when their portable ordinals coincide with current ones.
    pub(super) fn analysis_owners(
        &self,
        analysis: &runmat_mir::analysis::AnalysisStore,
        bytecode: &runmat_vm::Bytecode,
    ) -> Result<
        std::collections::BTreeMap<runmat_hir::FunctionId, runmat_types::ProgramFunctionId>,
        String,
    > {
        self.assembly
            .iter()
            .map(|local| {
                let facts = analysis.function_for_local(*local).ok_or_else(|| {
                    format!("compiled function {local:?} has no unique analysis self binding")
                })?;
                let portable = if self.registered_bodies.contains(local) {
                    let body = bytecode
                        .function_registry
                        .get(*local)
                        .ok_or("compiled function lost its original bytecode owner")?;
                    let origin = body
                        .binding_origin
                        .ok_or("compiled function has no immutable origin")?;
                    if body.function != *local || origin.function != body.portable_function {
                        return Err(
                            "compiled function coordinates disagree with its original identity"
                                .into(),
                        );
                    }
                    body.portable_function
                } else {
                    // A synthetic script is owned by the exact MIR/layout
                    // entrypoint, never by a numerically overlapping inherited
                    // registry body. Its original analysis ordinal is unchanged.
                    facts.function
                };
                Ok((*local, portable))
            })
            .collect()
    }
}

/// Select only bodies authored by this immutable product. MIR retains bodies
/// and declarations imported for whole-program analysis, but those inherited
/// definitions keep their earlier region and parallel compilation metadata.
pub(super) fn compiled_function_owners(
    current_program: runmat_execution::Digest,
    mir: &runmat_mir::MirAssembly,
    bytecode: &runmat_vm::Bytecode,
) -> Result<CompiledProductOwners, String> {
    let entrypoints = mir.entrypoints.iter().copied().collect::<BTreeSet<_>>();
    if let Some(missing) = entrypoints
        .iter()
        .find(|function| !mir.bodies.contains_key(function))
    {
        return Err(format!("MIR entrypoint {missing:?} has no executable body"));
    }
    let mut assembly = BTreeSet::new();
    let mut registered_bodies = BTreeSet::new();
    for function in mir.bodies.keys().copied() {
        let synthetic_script = entrypoints.contains(&function)
            && mir.functions.get(&function).is_some_and(|metadata| {
                metadata.kind == runmat_hir::FunctionKind::SyntheticEntrypoint
            })
            && bytecode.layout.as_ref().is_some_and(|layout| {
                layout
                    .entrypoints
                    .values()
                    .any(|entry| entry.target == function)
                    && layout.functions.contains_key(&function)
            });
        if synthetic_script {
            if let Some(body) = bytecode.function_registry.functions.get(&function) {
                let origin = body.binding_origin.ok_or_else(|| {
                    format!(
                        "synthetic MIR entrypoint {function:?} collides with an unoriginated body"
                    )
                })?;
                if origin.program == current_program {
                    return Err(format!(
                        "synthetic MIR entrypoint {function:?} collides with a current product body"
                    ));
                }
                if let Some(bound) = bytecode.bound_functions.get(&function) {
                    if bound.binding_origin != body.binding_origin
                        || bound.source_id != body.source_id
                        || !std::sync::Arc::ptr_eq(&bound.definition_owner, &body.definition_owner)
                    {
                        return Err(format!(
                            "synthetic MIR entrypoint {function:?} has a different bound body"
                        ));
                    }
                }
            } else if bytecode.bound_functions.contains_key(&function) {
                return Err(format!(
                    "synthetic MIR entrypoint {function:?} has an orphan bound body"
                ));
            }
            assembly.insert(function);
            continue;
        }
        let Some(body) = bytecode.function_registry.functions.get(&function) else {
            return Err(if entrypoints.contains(&function) {
                format!("unregistered MIR entrypoint {function:?} is not synthetic")
            } else {
                format!("MIR function {function:?} has no executable body")
            });
        };
        let origin = body
            .binding_origin
            .ok_or_else(|| format!("MIR function {function:?} has no immutable binding origin"))?;
        if body.function != function || origin.function != body.portable_function {
            return Err(format!(
                "MIR function {function:?} has inconsistent original coordinates"
            ));
        }
        if origin.program == current_program {
            assembly.insert(function);
            registered_bodies.insert(function);
        } else if entrypoints.contains(&function) {
            return Err(format!(
                "MIR entrypoint {function:?} belongs to a different immutable program"
            ));
        }
    }
    Ok(CompiledProductOwners {
        assembly,
        registered_bodies,
    })
}
