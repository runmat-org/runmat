//! Immutable foreign closure selected from one authenticated Direct bundle.
//! Source paths and the worker's mutable adapter catalog are not inputs to a
//! later selected V3 publication.

use std::path::Path;
use std::sync::Arc;

use runmat_execution::Digest;
use runmat_execution_artifact::{
    ExecutionBundle, ForeignArtifactClosure, LogicalObject, ProgramBuildRecipe,
    SelectedForeignProjection,
};
use runmat_package::{ContentDigest, FrozenProjectHandoff};
use runmat_python::{PythonArtifactBundle, PythonExecutionMode, PYTHON_ARTIFACT_BUNDLE_MEDIA_TYPE};
use runmat_types::InteropManifest;

use super::PrivateArtifactRoot;

/// A Direct worker may carry several recipes in one bundle. This projection
/// retains only the exact request recipe's foreign identities and object bytes.
/// The Arc bundle is already validated; no object bytes are cloned until the
/// one selected-unit archive is built at publication.
pub struct PreparedSelectedForeignInput {
    selection: SelectedForeignProjection<'static>,
    recipe: ProgramBuildRecipe,
    rebased_handoff: Arc<FrozenProjectHandoff>,
    python_execution_mode: Option<PythonExecutionMode>,
    reuse_basis: Digest,
}

impl PreparedSelectedForeignInput {
    pub fn from_validated_bundle(
        bundle: Arc<ExecutionBundle>,
        recipe: &ProgramBuildRecipe,
        rebased_handoff: Arc<FrozenProjectHandoff>,
        private_root: &PrivateArtifactRoot,
    ) -> Result<Arc<Self>, String> {
        bundle.validate().map_err(|error| error.to_string())?;
        recipe.validate().map_err(|error| error.to_string())?;
        if bundle.manifest.program_revision != recipe.program_revision
            || !bundle
                .manifest
                .recipes
                .iter()
                .any(|member| member == recipe)
        {
            return Err("selected Direct recipe is not a member of its admitted bundle".into());
        }
        let mut expected_handoff = bundle
            .project_handoff_at(&rebased_handoff.project.workspace_root)
            .map_err(|error| error.to_string())?;
        // The bundle projects source paths, while the native materializer also
        // rebases foreign paths. Only private paths issued for the bundle's
        // authenticated bytes may account for those differences.
        for (source, path) in &rebased_handoff.project.access_paths {
            require_private_bytes(private_root, path, &source.content_digest)?;
        }
        if expected_handoff.project.native_interfaces.len()
            != rebased_handoff.project.native_interfaces.len()
            || expected_handoff.project.mex_artifacts.len()
                != rebased_handoff.project.mex_artifacts.len()
            || expected_handoff.project.java_artifacts.len()
                != rebased_handoff.project.java_artifacts.len()
            || expected_handoff.project.python_artifacts.len()
                != rebased_handoff.project.python_artifacts.len()
        {
            return Err("selected Direct handoff has a different foreign artifact set".into());
        }
        for (expected, materialized) in expected_handoff
            .project
            .native_interfaces
            .iter_mut()
            .zip(&rebased_handoff.project.native_interfaces)
        {
            require_private_bytes(
                private_root,
                &materialized.manifest_path,
                &expected.manifest_digest,
            )?;
            require_private_bytes(
                private_root,
                &materialized.library_path,
                &expected.library_digest,
            )?;
            expected.manifest_path = materialized.manifest_path.clone();
            expected.library_path = materialized.library_path.clone();
        }
        for (expected, materialized) in expected_handoff
            .project
            .mex_artifacts
            .iter_mut()
            .zip(&rebased_handoff.project.mex_artifacts)
        {
            require_private_bytes(
                private_root,
                &materialized.manifest_path,
                &expected.manifest_digest,
            )?;
            require_private_bytes(
                private_root,
                &materialized.module_path,
                &expected.module_digest,
            )?;
            expected.manifest_path = materialized.manifest_path.clone();
            expected.module_path = materialized.module_path.clone();
        }
        for (expected, materialized) in expected_handoff
            .project
            .java_artifacts
            .iter_mut()
            .zip(&rebased_handoff.project.java_artifacts)
        {
            require_private_bytes(private_root, &materialized.path, &expected.digest)?;
            expected.path = materialized.path.clone();
        }
        for (expected, materialized) in expected_handoff
            .project
            .python_artifacts
            .iter_mut()
            .zip(&rebased_handoff.project.python_artifacts)
        {
            require_private_bytes(private_root, &materialized.path, &expected.digest)?;
            expected.path = materialized.path.clone();
        }
        if expected_handoff != *rebased_handoff {
            return Err("selected Direct handoff differs from authenticated bundle bytes".into());
        }
        let selection = SelectedForeignProjection::from_owned(Arc::clone(&bundle), recipe)
            .map_err(|error| error.to_string())?;
        if let Some(policy) = selection.policy_object() {
            super::policy::decode_admitted_policy(policy)?;
        }
        let mut python_execution_mode = None;
        let mut seen_python_bundle = false;
        for object in selection.objects() {
            if object.descriptor.media_type == PYTHON_ARTIFACT_BUNDLE_MEDIA_TYPE {
                if seen_python_bundle {
                    return Err(
                        "selected Direct recipe has multiple Python artifact bundles".into(),
                    );
                }
                seen_python_bundle = true;
                let python = PythonArtifactBundle::from_canonical_bytes(&object.bytes)
                    .map_err(|error| error.to_string())?;
                // Artifacts-only bundles intentionally carry no interpreter
                // policy. Keep that absence typed; never supply a host default.
                python_execution_mode = python
                    .environment
                    .as_ref()
                    .map(|environment| environment.execution_mode);
            }
        }
        let mut objects = selection
            .objects()
            .map(|object| &object.descriptor)
            .collect::<Vec<_>>();
        objects.sort();
        let mut closures = selection.closures().collect::<Vec<_>>();
        closures.sort();
        let encoded = serde_json::to_vec(&(
            "runmat-selected-direct-foreign-basis-v1",
            recipe.id().map_err(|error| error.to_string())?,
            objects,
            closures,
        ))
        .map_err(|error| error.to_string())?;
        Ok(Arc::new(Self {
            selection,
            recipe: recipe.clone(),
            rebased_handoff,
            python_execution_mode,
            reuse_basis: Digest::sha256(&encoded),
        }))
    }

    pub fn matches_handoff(&self, handoff: &FrozenProjectHandoff) -> bool {
        self.rebased_handoff.as_ref() == handoff
    }

    pub fn interop(&self) -> &InteropManifest {
        &self.recipe.interop
    }

    pub fn selected_reuse_basis(&self) -> Digest {
        self.reuse_basis
    }

    /// This mode is read from the selected, validated bundle object, never a
    /// default host RuntimeConfig or mutable Python installation.
    pub fn python_execution_mode(&self) -> Option<PythonExecutionMode> {
        self.python_execution_mode
    }

    pub(crate) fn foreign_objects(&self) -> Vec<LogicalObject> {
        self.selection.objects().cloned().collect()
    }

    pub(crate) fn foreign_object_refs(&self) -> Vec<&LogicalObject> {
        self.selection.objects().collect()
    }

    pub(crate) fn policy_object(&self) -> Option<&LogicalObject> {
        self.selection.policy_object()
    }

    pub(crate) fn foreign_closures(&self) -> Vec<ForeignArtifactClosure> {
        self.selection.closures().cloned().collect()
    }

    pub(crate) fn foreign_closure_refs(&self) -> Vec<&ForeignArtifactClosure> {
        self.selection.closures().collect()
    }
}

fn require_private_bytes(
    root: &PrivateArtifactRoot,
    path: &Path,
    expected: &ContentDigest,
) -> Result<(), String> {
    if root
        .admitted_path(path)
        .is_some_and(|issued| issued.content_digest() == expected)
    {
        Ok(())
    } else {
        Err("selected Direct handoff path lacks exact admitted private bytes".into())
    }
}
