//! Atomic source-domain projection for selected VM executable streams.

use super::*;
#[cfg(feature = "native-accel")]
use crate::bytecode::fusion_stage::FusionStageStream;
#[cfg(feature = "native-accel")]
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FunctionSourceProjectionError {
    Missing(BytecodeSourceRef),
    InvalidStage {
        function: FunctionId,
        reason: String,
    },
}

impl FunctionRegistry {
    /// Rebind current definitions without changing names, owners or retained
    /// old bodies. Neither bodies nor the source index publish on failure.
    pub fn project_current_source_ids(
        &mut self,
        mut project: impl FnMut(BytecodeSourceRef) -> Option<BytecodeSourceRef>,
    ) -> Result<(), FunctionSourceProjectionError> {
        let plan = PreparedRegistry::new(self, &mut |source| {
            project(source).ok_or(FunctionSourceProjectionError::Missing(source))
        })?;
        plan.commit(self);
        Ok(())
    }
}

impl Bytecode {
    /// Project the entrypoint and both executable function views under one
    /// selected source mapping. A malformed prestate or missing mapping leaves
    /// the caller's bytecode and source indexes untouched.
    pub fn project_source_ids(
        &mut self,
        mut project: impl FnMut(BytecodeSourceRef) -> Option<BytecodeSourceRef>,
    ) -> Result<(), String> {
        let mut mapped = HashMap::<BytecodeSourceRef, Option<BytecodeSourceRef>>::new();
        let mut selected =
            |source: BytecodeSourceRef| *mapped.entry(source).or_insert_with(|| project(source));
        let entry_source = self
            .source_id
            .map(|source| {
                selected(source).ok_or_else(|| format!("entrypoint source {source:?} is absent"))
            })
            .transpose()?;
        #[cfg(feature = "native-accel")]
        let entry_stage = prepare_entry_stage(self, entry_source)?;
        let registry = PreparedRegistry::new(&self.function_registry, &mut |source| {
            selected(source).ok_or(FunctionSourceProjectionError::Missing(source))
        })
        .map_err(|error| format!("function source projection: {error:?}"))?;
        let mut bound = Vec::with_capacity(self.bound_functions.len());
        for (id, body) in &self.bound_functions {
            let source = body
                .source_id
                .map(|source| {
                    selected(source)
                        .ok_or_else(|| format!("bound function {id:?} source {source:?} is absent"))
                })
                .transpose()?;
            if let Some(current) = registry.body(*id) {
                if source != current.source {
                    return Err(format!(
                        "bound function {id:?} disagrees with registry source"
                    ));
                }
            }
            #[cfg(feature = "native-accel")]
            let stage = if let Some(current) = registry.body(*id) {
                let current_body = &self.function_registry.functions[id];
                if body.source_id == current_body.source_id
                    && body
                        .fusion_stage_projection
                        .as_ref()
                        .zip(current_body.fusion_stage_projection.as_ref())
                        .is_some_and(|(bound, registered)| Arc::ptr_eq(bound, registered))
                {
                    // Both views are independently validated against their
                    // original streams. Shared selected facts and source mean
                    // the registry's prepared digest is also the bound digest.
                    body.validate_fusion_stage_projection()
                        .map_err(|reason| format!("function {id:?}: {reason}"))?;
                    current.stage.clone()
                } else {
                    prepare_body_stage(body, source)
                        .map_err(|reason| format!("function {id:?}: {reason}"))?
                }
            } else {
                prepare_body_stage(body, source)
                    .map_err(|reason| format!("function {id:?}: {reason}"))?
            };
            bound.push(PreparedBody {
                id: *id,
                source,
                #[cfg(feature = "native-accel")]
                stage,
            });
        }
        // No fallible operations below: publish all selected views together.
        self.source_id = entry_source;
        #[cfg(feature = "native-accel")]
        {
            self.fusion_metadata.stage_projection = entry_stage;
        }
        for prepared in bound {
            let body = self
                .bound_functions
                .get_mut(&prepared.id)
                .expect("planned bound body");
            body.source_id = prepared.source;
            #[cfg(feature = "native-accel")]
            {
                body.fusion_stage_projection = prepared.stage;
            }
        }
        registry.commit(&mut self.function_registry);
        Ok(())
    }
}

struct PreparedBody {
    id: FunctionId,
    source: Option<BytecodeSourceRef>,
    #[cfg(feature = "native-accel")]
    stage: Option<Arc<crate::bytecode::FusionStageProjection>>,
}

struct PreparedRegistry {
    bodies: Vec<PreparedBody>,
    positions: HashMap<FunctionId, usize>,
    index: HashMap<BytecodeSourceRef, Vec<FunctionId>>,
}

impl PreparedRegistry {
    fn new(
        registry: &FunctionRegistry,
        project: &mut impl FnMut(
            BytecodeSourceRef,
        ) -> Result<BytecodeSourceRef, FunctionSourceProjectionError>,
    ) -> Result<Self, FunctionSourceProjectionError> {
        let mut ids = registry.functions.keys().copied().collect::<Vec<_>>();
        ids.sort_by_key(|id| id.0);
        let mut bodies = Vec::with_capacity(ids.len());
        let mut positions = HashMap::with_capacity(ids.len());
        let mut index = HashMap::<BytecodeSourceRef, Vec<FunctionId>>::new();
        let mut mapped = HashMap::<BytecodeSourceRef, BytecodeSourceRef>::new();
        for id in ids {
            let body = &registry.functions[&id];
            let source = body
                .source_id
                .map(|source| {
                    if let Some(mapped_source) = mapped.get(&source) {
                        Ok(*mapped_source)
                    } else {
                        let mapped_source = project(source)?;
                        mapped.insert(source, mapped_source);
                        Ok(mapped_source)
                    }
                })
                .transpose()?;
            #[cfg(feature = "native-accel")]
            let stage = prepare_body_stage(body, source).map_err(|reason| {
                FunctionSourceProjectionError::InvalidStage {
                    function: id,
                    reason,
                }
            })?;
            if let Some(source) = source {
                index.entry(source).or_default().push(id);
            }
            positions.insert(id, bodies.len());
            bodies.push(PreparedBody {
                id,
                source,
                #[cfg(feature = "native-accel")]
                stage,
            });
        }
        Ok(Self {
            bodies,
            positions,
            index,
        })
    }

    fn body(&self, id: FunctionId) -> Option<&PreparedBody> {
        self.positions
            .get(&id)
            .map(|position| &self.bodies[*position])
    }

    fn commit(self, registry: &mut FunctionRegistry) {
        // DerefMut performs at most one registry COW; instruction vectors are
        // copied only when an existing shared snapshot must be preserved.
        let data = &mut **registry;
        for prepared in self.bodies {
            let body = data
                .functions
                .get_mut(&prepared.id)
                .expect("planned registry body");
            body.source_id = prepared.source;
            #[cfg(feature = "native-accel")]
            {
                body.fusion_stage_projection = prepared.stage;
            }
        }
        data.source_functions = self.index;
    }
}

#[cfg(feature = "native-accel")]
fn prepare_entry_stage(
    bytecode: &Bytecode,
    source: Option<BytecodeSourceRef>,
) -> Result<Option<Arc<crate::bytecode::FusionStageProjection>>, String> {
    bytecode.validate_fusion_stage_projection()?;
    let Some(original) = &bytecode.fusion_metadata.stage_projection else {
        return Ok(None);
    };
    if source == bytecode.source_id {
        return Ok(Some(original.clone()));
    }
    let mut prepared = original.clone();
    Arc::make_mut(&mut prepared)
        .reseal_owned_remap(FusionStageStream {
            instructions: &bytecode.instructions,
            catch_scopes: &bytecode.catch_scopes,
            resume_points: &bytecode.resume_points,
            source_id: source,
            var_count: bytecode.var_count,
        })
        .then_some(Some(prepared))
        .ok_or_else(|| "selected entrypoint MIR projection cannot be resealed".into())
}

#[cfg(feature = "native-accel")]
fn prepare_body_stage(
    body: &FunctionBytecode,
    source: Option<BytecodeSourceRef>,
) -> Result<Option<Arc<crate::bytecode::FusionStageProjection>>, String> {
    body.validate_fusion_stage_projection()?;
    let Some(original) = &body.fusion_stage_projection else {
        return Ok(None);
    };
    if source == body.source_id {
        return Ok(Some(original.clone()));
    }
    let mut prepared = original.clone();
    Arc::make_mut(&mut prepared)
        .reseal_owned_remap(FusionStageStream {
            instructions: &body.instructions,
            catch_scopes: &body.catch_scopes,
            resume_points: &body.resume_points,
            source_id: source,
            var_count: body.var_count,
        })
        .then_some(Some(prepared))
        .ok_or_else(|| "selected function MIR projection cannot be resealed".into())
}

#[cfg(all(test, feature = "native-accel"))]
mod tests {
    use super::*;
    use runmat_types::ProgramSourceId;
    use std::rc::Rc;

    fn selected_bytecode() -> Bytecode {
        selected_bytecode_from_source(
            "y=helper(1); function y=helper(x); mu=x+1; sigma=mu.*2; y=sigma-3; end",
        )
    }

    fn selected_bytecode_from_source(source: &str) -> Bytecode {
        let ast = runmat_parser::parse(source).unwrap();
        let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).unwrap();
        let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).unwrap();
        let mut bytecode =
            crate::bytecode::compile(&hir.assembly, &mir, hir.assembly.entrypoints[0].id).unwrap();
        bytecode.source_id = Some(BytecodeSourceRef::Program(ProgramSourceId(0)));
        bytecode.reseal_fusion_stage_after_owned_remap();
        for body in bytecode.bound_functions.values_mut() {
            function_remap::rebind_session_source(
                body,
                Some(BytecodeSourceRef::Program(ProgramSourceId(0))),
            )
            .unwrap();
        }
        bytecode.function_registry = FunctionRegistry::new(bytecode.bound_functions.clone());
        assert!(bytecode.validate_fusion_stage_projection().is_ok());
        bytecode
    }

    #[test]
    fn source_projection_retains_owned_instruction_buffers_across_many_bodies() {
        let mut source = String::from("y=helper0(1); ");
        for ordinal in 0..16 {
            source.push_str(&format!(
                "function y=helper{ordinal}(x); mu=x+1; sigma=mu.*2; y=sigma-3; end "
            ));
        }
        let mut bytecode = selected_bytecode_from_source(&source);
        assert!(bytecode.bound_functions.len() >= 16);
        assert!(bytecode
            .bound_functions
            .values()
            .any(|body| body.fusion_stage_projection.is_some()));
        let entry_buffer = bytecode.instructions.as_ptr();
        let bound_buffers = bytecode
            .bound_functions
            .iter()
            .map(|(id, body)| (*id, body.instructions.as_ptr()))
            .collect::<HashMap<_, _>>();
        let registry_buffers = bytecode
            .function_registry
            .functions
            .iter()
            .map(|(id, body)| (*id, body.instructions.as_ptr()))
            .collect::<HashMap<_, _>>();
        let mut source_map_calls = 0;
        bytecode
            .project_source_ids(|_| {
                source_map_calls += 1;
                Some(BytecodeSourceRef::Program(ProgramSourceId(1)))
            })
            .unwrap();
        assert_eq!(source_map_calls, 1);
        assert_eq!(bytecode.instructions.as_ptr(), entry_buffer);
        for (id, body) in &bytecode.bound_functions {
            assert_eq!(body.instructions.as_ptr(), bound_buffers[id]);
            assert!(body.validate_fusion_stage_projection().is_ok());
        }
        for (id, body) in &bytecode.function_registry.functions {
            assert_eq!(body.instructions.as_ptr(), registry_buffers[id]);
            assert!(body.validate_fusion_stage_projection().is_ok());
        }
    }

    #[test]
    fn program_to_runtime_source_projection_reseals_every_selected_view() {
        let mut bytecode = selected_bytecode();
        let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let lease = runmat_runtime::source_context::lease_source_context_in_context(
            &runtime,
            "selected.m",
            None,
            "selected source",
        )
        .unwrap();
        let selected = BytecodeSourceRef::Runtime(lease.source_ref());
        bytecode
            .project_source_ids(|source| {
                (source == BytecodeSourceRef::Program(ProgramSourceId(0))).then_some(selected)
            })
            .unwrap();
        assert_eq!(bytecode.source_id, Some(selected));
        assert!(bytecode.validate_fusion_stage_projection().is_ok());
        for (id, body) in &bytecode.bound_functions {
            assert_eq!(body.source_id, Some(selected));
            assert!(body.validate_fusion_stage_projection().is_ok());
            assert_eq!(
                bytecode.function_registry.get(*id).unwrap().source_id,
                Some(selected)
            );
            assert_eq!(
                bytecode
                    .function_registry
                    .functions_for_source_ref(selected),
                &[*id]
            );
        }
    }

    #[test]
    fn malformed_stage_or_missing_source_keeps_original_views_and_index() {
        let mut bytecode = selected_bytecode();
        let original = bytecode.clone();
        assert!(bytecode.project_source_ids(|_| None).is_err());
        assert_eq!(bytecode.source_id, original.source_id);
        assert_eq!(
            bytecode.function_registry.source_functions,
            original.function_registry.source_functions
        );

        std::sync::Arc::make_mut(bytecode.fusion_metadata.stage_projection.as_mut().unwrap())
            .source_fingerprint = "forged".into();
        let before = bytecode.clone();
        assert!(bytecode.project_source_ids(Some).is_err());
        assert_eq!(bytecode.source_id, before.source_id);
        assert_eq!(
            bytecode.function_registry.source_functions,
            before.function_registry.source_functions
        );
        assert_eq!(
            bytecode
                .fusion_metadata
                .stage_projection
                .as_ref()
                .unwrap()
                .source_fingerprint,
            "forged"
        );
    }

    #[test]
    fn registry_rejects_forged_selected_stage_without_reindexing_any_body() {
        let mut registry = selected_bytecode().function_registry;
        let id = *registry.functions.keys().next().expect("selected function");
        let original_source = registry.functions[&id].source_id;
        let original_index = registry.source_functions.clone();
        std::sync::Arc::make_mut(
            registry
                .functions
                .get_mut(&id)
                .unwrap()
                .fusion_stage_projection
                .as_mut()
                .unwrap(),
        )
        .source_fingerprint = "forged".into();
        assert!(matches!(
            registry.project_current_source_ids(Some),
            Err(FunctionSourceProjectionError::InvalidStage { function, .. }) if function == id
        ));
        assert_eq!(registry.functions[&id].source_id, original_source);
        assert_eq!(registry.source_functions, original_index);
    }
}
