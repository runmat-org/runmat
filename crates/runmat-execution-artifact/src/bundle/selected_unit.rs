use runmat_execution::{resource::accelerator_requirements_for_capabilities, OutputContract};
use runmat_execution::{
    Digest, ExecutableComponentKind, ExecutableSourceMapV2, ExecutableUnitEnvelope,
};

use super::{
    BuildResourceDeclaration, BundleCodeClosure, BundleManifest, CompiledPackageClosure,
    ExecutionBundle, ForeignArtifactClosure, ProjectRevisionRecord,
    EXECUTION_BUNDLE_SCHEMA_VERSION,
};
use crate::{
    ArtifactError, ArtifactResult, ExecutableForm, LogicalObject, ObjectNamespace, ProgramArtifact,
    ProgramBuildRecipe, ProgramTarget, PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
};

impl ExecutionBundle {
    /// Select one Direct V3 recipe's exact foreign closure from an admitted
    /// bundle, including a compiled package with no source-project handoff.
    /// The original bundle remains the authority for every selected object;
    /// unrelated recipes cannot contribute foreign bytes to this unit.
    pub fn selected_unit_bundle_for(
        &self,
        recipe: &ProgramBuildRecipe,
        canonical_v3: &[u8],
    ) -> ArtifactResult<Self> {
        let selection = self.selected_foreign_projection(recipe)?;
        let recipe_id = recipe.id()?;
        if self
            .manifest
            .artifacts
            .iter()
            .filter(|artifact| {
                artifact.form == ExecutableForm::ExecutableUnitV4
                    && artifact.recipe_id == recipe_id
                    && artifact.executable_bytes == canonical_v3
            })
            .count()
            != 1
        {
            return Err(ArtifactError::Identity(
                "selected Direct V3 is not the unique artifact of its recipe".into(),
            ));
        }
        let objects = selection.objects().cloned().collect();
        let closures = selection.closures().cloned().collect();
        let selected = Self::from_selected_executable_unit_with_policy(
            canonical_v3,
            objects,
            closures,
            selection.policy_object().cloned(),
            self.manifest.resources.clone(),
        )?;
        match self.project_runtime_policy_object()? {
            Some(policy) => selected.with_project_runtime_policy_object(policy.clone()),
            None => Ok(selected),
        }
    }

    /// Package one already-admitted V3 unit and its frozen foreign objects.
    /// The selected unit, not a later project handoff, supplies the program
    /// revision and compiled package identities. No source project or source
    /// file is opened by this constructor.
    pub fn from_selected_executable_unit(
        canonical_v3: &[u8],
        foreign_objects: Vec<LogicalObject>,
        foreign_artifact_closures: Vec<ForeignArtifactClosure>,
        resources: BuildResourceDeclaration,
    ) -> ArtifactResult<Self> {
        Self::from_selected_executable_unit_with_policy(
            canonical_v3,
            foreign_objects,
            foreign_artifact_closures,
            None,
            resources,
        )
    }

    pub fn from_selected_executable_unit_with_policy(
        canonical_v3: &[u8],
        mut foreign_objects: Vec<LogicalObject>,
        mut foreign_artifact_closures: Vec<ForeignArtifactClosure>,
        policy_object: Option<LogicalObject>,
        resources: BuildResourceDeclaration,
    ) -> ArtifactResult<Self> {
        let unit = ExecutableUnitEnvelope::from_canonical_bytes(canonical_v3)
            .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
        // A selected-unit foreign receipt is packaging metadata, not an
        // invocation. Zero outputs prevents a declaration-only unit from
        // inheriting an arbitrary caller's output contract.
        let recipe = ProgramBuildRecipe {
            schema_version: PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
            program_revision: unit.manifest.identity.program.clone(),
            entrypoint: unit.manifest.identity.entrypoint.clone(),
            outputs: OutputContract {
                requested_outputs: 0,
                owned_work: None,
            },
            execution_mode: "interpreter".into(),
            target: ProgramTarget::portable("selected-unit-foreign-receipt"),
            interop: unit.manifest.interop.clone(),
            accelerators: accelerator_requirements_for_capabilities(&unit.manifest.capabilities)
                .map_err(|error| ArtifactError::Invalid(error.to_string()))?,
            features: Default::default(),
            compile_options: Default::default(),
            source_objects: Vec::new(),
            foreign_policy: policy_object
                .as_ref()
                .map(|object| object.descriptor.clone()),
            expected_artifact_id: None,
        };
        let source_component = unit
            .component(ExecutableComponentKind::SourceMap)
            .ok_or_else(|| ArtifactError::Invalid("selected V3 has no source map".into()))?;
        let source_map = ExecutableSourceMapV2::decode(&source_component.bytes)
            .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
        let mut package_instances = source_map
            .entries
            .iter()
            .map(|record| record.owner_identity.clone())
            .collect::<Vec<_>>();
        package_instances.sort();
        package_instances.dedup();
        let artifact = ProgramArtifact::materialize(
            &recipe,
            ExecutableForm::ExecutableUnitV4,
            canonical_v3.to_vec(),
        )?;
        let revision = unit.manifest.identity.program;
        foreign_objects.sort_by(|left, right| left.descriptor.cmp(&right.descriptor));
        if foreign_objects
            .iter()
            .any(|object| object.descriptor.namespace != ObjectNamespace::ForeignArtifact)
        {
            return Err(ArtifactError::Invalid(
                "selected bundle contains a non-foreign object".into(),
            ));
        }
        foreign_artifact_closures.sort();
        let foreign_artifacts = foreign_objects
            .iter()
            .map(|object| object.descriptor.clone())
            .collect();
        if let Some(object) = policy_object {
            crate::object::validate_foreign_policy_object(&object)?;
            foreign_objects.push(object);
            foreign_objects.sort_by(|left, right| left.descriptor.cmp(&right.descriptor));
        }
        let graph_digest = Digest::from_bytes(*revision.graph_digest().bytes());
        let source_digest = Digest::from_bytes(*revision.source_digest().bytes());
        let bundle = Self {
            manifest: BundleManifest {
                schema_version: EXECUTION_BUNDLE_SCHEMA_VERSION,
                program_revision: revision,
                project_revision: ProjectRevisionRecord {
                    graph_digest,
                    source_digest,
                },
                code_closure: BundleCodeClosure::Compiled {
                    package: CompiledPackageClosure {
                        schema_version: CompiledPackageClosure::SCHEMA_VERSION,
                        graph_digest,
                        source_digest,
                        package_instances,
                    },
                },
                sources: Vec::new(),
                foreign_artifacts,
                foreign_artifact_closures,
                project_runtime_policy: None,
                callables: Vec::new(),
                recipes: vec![recipe],
                artifacts: vec![artifact],
                requested_capabilities: Default::default(),
                resources,
                portable_environment: Vec::new(),
            },
            objects: foreign_objects,
        };
        bundle.validate()?;
        Ok(bundle)
    }
}
