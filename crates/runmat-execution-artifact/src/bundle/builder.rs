use std::path::Path;

use runmat_execution::{Digest, ProgramRevision};
use runmat_package::{FrozenProject, FrozenProjectHandoff};

use crate::bundle::{
    BuildResourceDeclaration, BundleCallable, BundleCodeClosure, BundleManifest,
    CompiledPackageClosure, ExecutionBundle, ProjectRevisionRecord,
    EXECUTION_BUNDLE_SCHEMA_VERSION,
};
use crate::{
    ArtifactError, ArtifactResult, ExecutableForm, LogicalObject, ObjectNamespace, ProgramArtifact,
    ProgramBuildRecipe,
};

struct Materialization {
    recipe: ProgramBuildRecipe,
    form: ExecutableForm,
    executable_bytes: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CodeClosureMode {
    SourceProject,
    Compiled,
}

pub trait SourceReader {
    fn read(&self, path: &Path) -> ArtifactResult<Vec<u8>>;
}

impl<F> SourceReader for F
where
    F: Fn(&Path) -> ArtifactResult<Vec<u8>>,
{
    fn read(&self, path: &Path) -> ArtifactResult<Vec<u8>> {
        self(path)
    }
}

pub struct ExecutionBundleBuilder<'a, R> {
    project: &'a FrozenProject,
    revision: ProgramRevision,
    reader: R,
    recipes: Vec<ProgramBuildRecipe>,
    materializations: Vec<Materialization>,
    resources: BuildResourceDeclaration,
    code_closure: CodeClosureMode,
}

impl<'a, R: SourceReader> ExecutionBundleBuilder<'a, R> {
    pub fn new(
        project: &'a FrozenProject,
        revision: ProgramRevision,
        reader: R,
    ) -> ArtifactResult<Self> {
        let project_revision = project.revision();
        let exact_project_sources =
            revision.source_digest().bytes() == project_revision.source_revision.bytes();
        let exact_test_overlay = revision.domain_contribution("runmat.test.config").is_some();
        if revision.graph_digest().bytes() != project_revision.graph_digest.bytes()
            || (!exact_project_sources && !exact_test_overlay)
        {
            return Err(ArtifactError::Identity(
                "program and frozen-project revisions differ without an exact test overlay".into(),
            ));
        }
        Ok(Self {
            project,
            revision,
            reader,
            recipes: Vec::new(),
            materializations: Vec::new(),
            resources: BuildResourceDeclaration {
                cpu_millicores: 1000,
                memory_bytes: 1024 * 1024 * 1024,
                scratch_bytes: 1024 * 1024 * 1024,
            },
            code_closure: CodeClosureMode::SourceProject,
        })
    }

    pub fn with_recipe(mut self, recipe: ProgramBuildRecipe) -> Self {
        self.recipes.push(recipe);
        self
    }

    pub fn with_materialized_program(
        mut self,
        recipe: ProgramBuildRecipe,
        form: ExecutableForm,
        executable_bytes: Vec<u8>,
    ) -> Self {
        self.materializations.push(Materialization {
            recipe,
            form,
            executable_bytes,
        });
        self
    }

    pub fn with_resources(mut self, resources: BuildResourceDeclaration) -> Self {
        self.resources = resources;
        self
    }

    /// Package an already compiled program without source files or a project
    /// handoff. The immutable program revision and sorted package identities
    /// remain in the bundle closure, so workers can validate the exact graph
    /// without resolving or materializing it.
    pub fn with_compiled_package_closure(mut self) -> Self {
        self.code_closure = CodeClosureMode::Compiled;
        self
    }

    pub fn build(mut self) -> ArtifactResult<ExecutionBundle> {
        if self.code_closure == CodeClosureMode::Compiled
            && (self.materializations.is_empty()
                || self.materializations.iter().any(|materialization| {
                    !matches!(
                        materialization.form,
                        ExecutableForm::ExecutableUnitV3
                            | ExecutableForm::NativeObjectV1
                            | ExecutableForm::MeshingWorkload
                    )
                }))
        {
            return Err(ArtifactError::Invalid(
                "compiled package closure requires a compiled executable-unit or native-object artifact"
                    .into(),
            ));
        }
        let mut objects = Vec::new();
        let mut callables = Vec::new();
        let code_closure = match self.code_closure {
            CodeClosureMode::SourceProject => {
                let mut project_handoff = FrozenProjectHandoff::new(self.project.clone());
                project_handoff.project.manifest_path = "runmat.toml".into();
                project_handoff.project.workspace_root = ".".into();
                for package in self.project.sources.packages.values() {
                    for source in &package.sources {
                        let path = self.project.access_paths.get(&source.id).ok_or_else(|| {
                            ArtifactError::Invalid(format!(
                                "source {} has no frozen access path",
                                source.id.relative_path
                            ))
                        })?;
                        let bytes = self.reader.read(path)?;
                        if source.id.content_digest.bytes() != Digest::sha256(&bytes).bytes() {
                            return Err(ArtifactError::Identity(format!(
                                "source {} changed after project freeze",
                                source.id.relative_path
                            )));
                        }
                        let logical_name =
                            format!("{}/{}", package.mount.logical_root, source.id.relative_path);
                        project_handoff
                            .project
                            .access_paths
                            .insert(source.id.clone(), logical_name.clone().into());
                        objects.push(LogicalObject::new(
                            ObjectNamespace::ProgramSource,
                            logical_name,
                            "text/x-matlab",
                            bytes,
                        )?);
                        callables.push(BundleCallable {
                            owner_identity: package.package_instance.to_string(),
                            qualified_name: source.qualified_name.clone(),
                            source_digest: Digest::from_bytes(*source.id.content_digest.bytes()),
                        });
                    }
                }
                BundleCodeClosure::SourceProject {
                    handoff: project_handoff,
                }
            }
            CodeClosureMode::Compiled => {
                let mut package_instances = self
                    .project
                    .graph
                    .packages
                    .values()
                    .map(|package| package.instance.to_string())
                    .collect::<Vec<_>>();
                package_instances.sort();
                package_instances.dedup();
                BundleCodeClosure::Compiled {
                    package: CompiledPackageClosure {
                        schema_version: CompiledPackageClosure::SCHEMA_VERSION,
                        graph_digest: Digest::from_bytes(*self.project.graph.graph_digest.bytes()),
                        source_digest: Digest::from_bytes(*self.project.sources.revision.bytes()),
                        package_instances,
                    },
                }
            }
        };
        objects.sort_by(|left, right| left.descriptor.cmp(&right.descriptor));
        callables.sort();
        let source_descriptors = objects
            .iter()
            .map(|object| object.descriptor.clone())
            .collect::<Vec<_>>();
        for materialization in &mut self.materializations {
            attach_source_closure(
                &mut materialization.recipe,
                &self.revision,
                &source_descriptors,
            )?;
        }
        for recipe in &mut self.recipes {
            attach_source_closure(recipe, &self.revision, &source_descriptors)?;
        }
        let mut artifacts = Vec::with_capacity(self.materializations.len());
        for materialization in self.materializations {
            let artifact = ProgramArtifact::materialize(
                &materialization.recipe,
                materialization.form,
                materialization.executable_bytes,
            )?;
            self.recipes.push(materialization.recipe);
            artifacts.push(artifact);
        }
        let mut keyed_recipes = self
            .recipes
            .into_iter()
            .map(|recipe| Ok((recipe.id()?, recipe)))
            .collect::<ArtifactResult<Vec<_>>>()?;
        keyed_recipes.sort_by_key(|(id, _)| *id);
        let recipes = keyed_recipes
            .into_iter()
            .map(|(_, recipe)| recipe)
            .collect();
        artifacts.sort_by_key(|artifact| artifact.id);
        let project_revision = self.project.revision();
        let manifest = BundleManifest {
            schema_version: EXECUTION_BUNDLE_SCHEMA_VERSION,
            program_revision: self.revision,
            project_revision: ProjectRevisionRecord {
                graph_digest: Digest::from_bytes(*project_revision.graph_digest.bytes()),
                source_digest: Digest::from_bytes(*project_revision.source_revision.bytes()),
            },
            code_closure,
            sources: source_descriptors,
            callables,
            recipes,
            artifacts,
            requested_capabilities: Default::default(),
            resources: self.resources,
            portable_environment: Vec::new(),
        };
        let bundle = ExecutionBundle { manifest, objects };
        bundle.validate()?;
        Ok(bundle)
    }
}

fn attach_source_closure(
    recipe: &mut ProgramBuildRecipe,
    revision: &ProgramRevision,
    source_descriptors: &[crate::ObjectDescriptor],
) -> ArtifactResult<()> {
    if &recipe.program_revision != revision {
        return Err(ArtifactError::Identity(
            "program recipe revision differs from the bundle".into(),
        ));
    }
    if recipe.source_objects.is_empty() {
        recipe.source_objects = source_descriptors.to_vec();
    } else if recipe.source_objects != source_descriptors {
        return Err(ArtifactError::Identity(
            "program recipe source closure differs from the frozen project".into(),
        ));
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
impl<'a> ExecutionBundleBuilder<'a, fn(&Path) -> ArtifactResult<Vec<u8>>> {
    pub fn native(project: &'a FrozenProject, revision: ProgramRevision) -> ArtifactResult<Self> {
        Self::new(project, revision, native_read)
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn native_read(path: &Path) -> ArtifactResult<Vec<u8>> {
    std::fs::read(path).map_err(Into::into)
}
