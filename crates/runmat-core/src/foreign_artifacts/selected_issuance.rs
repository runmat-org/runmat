use std::sync::Arc;

use anyhow::Result;
use runmat_execution_artifact::{
    ExecutionBundle, ForeignArtifactClosure, LogicalObject, PreparedForeignParentView,
    SelectedForeignProjection, SelectedPreparedForeignMaterial,
};
use runmat_types::InteropManifest;

use super::{
    prepare_runtime_only_for_selection, selected_artifact_basis, PreparedForeignArtifacts,
    PreparedProjectForeignArtifacts, PreparedSelectedForeignInput,
};

#[derive(Clone)]
pub(crate) enum SelectedForeignPreparation {
    Configured(Arc<PreparedProjectForeignArtifacts>),
    Admitted(Arc<PreparedSelectedForeignInput>),
    RuntimeOnly {
        config: Arc<runmat_config::runtime::RunMatRuntimeConfig>,
        artifacts: Arc<PreparedForeignArtifacts>,
        policy_object: LogicalObject,
    },
    /// A prior selected body's checked archive is the immutable foreign
    /// authority of an escaped callback, never the latest session project.
    Archived {
        bundle: Arc<ExecutionBundle>,
        selection: Arc<SelectedForeignProjection<'static>>,
        interop: InteropManifest,
    },
}

#[derive(Clone)]
pub(crate) enum ForeignIssuanceSource {
    Admitted(Arc<PreparedSelectedForeignInput>),
    Project {
        handoff: Arc<runmat_package::FrozenProjectHandoff>,
        config: Arc<runmat_config::runtime::RunMatRuntimeConfig>,
    },
    Runtime {
        config: Arc<runmat_config::runtime::RunMatRuntimeConfig>,
    },
}

impl ForeignIssuanceSource {
    pub(crate) fn capture(
        handoff: Option<&runmat_package::FrozenProjectHandoff>,
        config: Option<&Arc<runmat_config::runtime::RunMatRuntimeConfig>>,
        admitted: Option<&Arc<PreparedSelectedForeignInput>>,
    ) -> Result<Option<Self>> {
        if let Some(admitted) = admitted {
            let handoff = handoff.ok_or_else(|| {
                anyhow::anyhow!("admitted Direct foreign input has no frozen project handoff")
            })?;
            anyhow::ensure!(
                admitted.matches_handoff(handoff),
                "admitted Direct foreign input differs from its frozen project handoff"
            );
            return Ok(Some(Self::Admitted(Arc::clone(admitted))));
        }
        let Some(config) = config else {
            return Ok(None);
        };
        Ok(Some(match handoff {
            Some(handoff) => Self::Project {
                handoff: Arc::new(handoff.clone()),
                config: Arc::clone(config),
            },
            None => Self::Runtime {
                config: Arc::clone(config),
            },
        }))
    }

    pub(crate) fn prepare_for(
        &self,
        inferred: &InteropManifest,
    ) -> Result<SelectedForeignPreparation> {
        inferred.validate()?;
        let python = runmat_types::ForeignAdapterId::new(runmat_python::PYTHON_ADAPTER_ID)?;
        let python_required = inferred
            .adapters
            .iter()
            .any(|adapter| adapter.adapter == python);
        match self {
            Self::Admitted(input) => {
                anyhow::ensure!(
                    !python_required || input.python_execution_mode().is_some(),
                    "captured Direct bundle has no authenticated Python execution mode"
                );
                Ok(SelectedForeignPreparation::Admitted(Arc::clone(input)))
            }
            Self::Project { handoff, config } => Ok(SelectedForeignPreparation::Configured(
                Arc::new(PreparedProjectForeignArtifacts::prepare_selected(
                    Arc::clone(handoff),
                    Arc::clone(config),
                    inferred,
                )?),
            )),
            Self::Runtime { config } => {
                anyhow::ensure!(
                    *inferred != InteropManifest::empty(),
                    "runtime-only source has no selected foreign adapter"
                );
                SelectedForeignPreparation::runtime_only(
                    Arc::clone(config),
                    Arc::new(prepare_runtime_only_for_selection(config, inferred)?),
                )
            }
        }
    }
}

impl SelectedForeignPreparation {
    pub(crate) fn runtime_only(
        config: Arc<runmat_config::runtime::RunMatRuntimeConfig>,
        artifacts: Arc<PreparedForeignArtifacts>,
    ) -> Result<Self> {
        let policy_object = super::policy::capture_foreign_policy_object(&config.foreign)
            .map_err(anyhow::Error::msg)?;
        Ok(Self::RuntimeOnly {
            config,
            artifacts,
            policy_object,
        })
    }

    pub(crate) fn from_selected_receipt(
        receipt: &std::rc::Rc<runmat_vm::SelectedUnitInteropReceipt>,
        canonical_v3: &Arc<[u8]>,
        owner: &runmat_runtime::source_context::SelectedProgramSourceOwner,
    ) -> Result<Self> {
        anyhow::ensure!(
            receipt.matches_selected_unit(canonical_v3, owner),
            "selected foreign receipt differs from its exact body/source"
        );
        let bundle = runmat_execution_artifact::archive::read_bundle(
            receipt.archive().as_ref(),
            runmat_execution_artifact::archive::ArchiveLimits::default(),
        )?;
        bundle.validate()?;
        anyhow::ensure!(
            bundle.manifest.recipes.len() == 1,
            "selected foreign receipt lacks one exact recipe"
        );
        let bundle = Arc::new(bundle);
        let interop = bundle.manifest.recipes[0].interop.clone();
        let selection = SelectedForeignProjection::from_owned_subset(
            Arc::clone(&bundle),
            &bundle.manifest.recipes[0],
            &interop,
        )?;
        if let Some(policy) = selection.policy_object() {
            super::policy::decode_admitted_policy(policy).map_err(anyhow::Error::msg)?;
        }
        Ok(Self::Archived {
            bundle,
            selection: Arc::new(selection),
            interop,
        })
    }

    /// A child function-only V3 may consume less than its parent's exact
    /// foreign recipe, never a new capability or mutable project selection.
    pub(crate) fn scoped_to(&self, child: &InteropManifest) -> Result<Self> {
        if self.interop() == child {
            return Ok(self.clone());
        }
        match self {
            Self::Archived { bundle, .. } => {
                let recipe = bundle.manifest.recipes.first().ok_or_else(|| {
                    anyhow::anyhow!("selected archive has no exact parent recipe")
                })?;
                let selection = SelectedForeignProjection::from_owned_subset(
                    Arc::clone(bundle),
                    recipe,
                    child,
                )?;
                Ok(Self::Archived {
                    bundle: Arc::clone(bundle),
                    selection: Arc::new(selection),
                    interop: child.clone(),
                })
            }
            _ => anyhow::bail!("dynamic selected foreign source has no archived parent receipt"),
        }
    }

    /// Complete a compiler-proven function-only requirement with the exact
    /// captured adapter grant. The compiler may infer Python without knowing
    /// its private environment/wheel identities; those identities come only
    /// from this prepared source or selected body's checked archive.
    pub(crate) fn enrich_inferred_child(
        &self,
        inferred: &InteropManifest,
    ) -> Result<InteropManifest> {
        inferred.validate()?;
        let parent = self.interop();
        let python_adapter = runmat_types::ForeignAdapterId::new(runmat_python::PYTHON_ADAPTER_ID)?;
        let mut enriched = inferred.clone();
        for requirement in &mut enriched.adapters {
            let grant = parent
                .adapters
                .iter()
                .find(|candidate| candidate.adapter == requirement.adapter)
                .ok_or_else(|| {
                    anyhow::anyhow!("inferred foreign adapter has no exact selected grant")
                })?;
            // The compiler's typed capabilities and foreign type contract stay
            // unchanged. Only its missing byte pin and the parent's stricter
            // version/process constraints are inherited from the grant.
            requirement.minimum_version = requirement.minimum_version.max(grant.minimum_version);
            requirement.execution_stack = requirement.execution_stack.max(grant.execution_stack);
            if requirement.adapter == python_adapter && requirement.artifact_identities.is_empty() {
                requirement.artifact_identities = grant.artifact_identities.clone();
            }
        }
        enriched.validate()?;
        // The same Artifact-owned subset selector used for materialization
        // proves that merging supplied grants did not add another capability.
        if matches!(self, Self::Archived { .. }) {
            self.scoped_to(&enriched)?;
        }
        Ok(enriched)
    }

    pub(crate) fn issue_for_compiled_unit(
        &self,
        canonical_v3: Arc<[u8]>,
        source_owner: &runmat_runtime::source_context::SelectedProgramSourceOwner,
    ) -> Result<(
        runmat_execution::Digest,
        std::rc::Rc<runmat_vm::SelectedUnitInteropReceipt>,
    )> {
        anyhow::ensure!(
            !matches!(self, Self::Archived { .. }),
            "new foreign source requires captured prepared inputs"
        );
        let (objects, closures, policy) = match self {
            Self::Configured(prepared) => (
                prepared.artifacts().objects.iter().collect::<Vec<_>>(),
                prepared.artifacts().closures.iter().collect::<Vec<_>>(),
                Some(prepared.policy_object()),
            ),
            Self::Admitted(prepared) => (
                prepared.foreign_object_refs(),
                prepared.foreign_closure_refs(),
                prepared.policy_object(),
            ),
            Self::RuntimeOnly {
                artifacts,
                policy_object,
                ..
            } => (
                artifacts.objects.iter().collect::<Vec<_>>(),
                artifacts.closures.iter().collect::<Vec<_>>(),
                Some(policy_object),
            ),
            _ => anyhow::bail!("new foreign source requires captured prepared inputs"),
        };
        let parent = PreparedForeignParentView::new(
            self.interop(),
            policy.map(|object| &object.descriptor),
            policy,
            &objects,
            &closures,
        )?;
        let selection = SelectedPreparedForeignMaterial::for_compiled_unit(&canonical_v3, &parent)?;
        let (interop, objects, closures, policy_object) = selection.into_parts();
        let basis = self.selected_reuse_basis()?;
        let receipt = super::selected_receipt::selected_unit_interop_receipt_with_parts(
            canonical_v3,
            source_owner,
            &interop,
            objects.into_iter().cloned().collect(),
            closures.into_iter().cloned().collect(),
            policy_object.cloned(),
        )
        .map_err(anyhow::Error::msg)?
        .ok_or_else(|| anyhow::anyhow!("new foreign source has no checked receipt"))?;
        Ok((basis, receipt))
    }

    pub(crate) fn interop(&self) -> &InteropManifest {
        match self {
            Self::Configured(prepared) => &prepared.artifacts().interop,
            Self::Admitted(prepared) => prepared.interop(),
            Self::RuntimeOnly { artifacts, .. } => &artifacts.interop,
            Self::Archived { interop, .. } => interop,
        }
    }

    pub(crate) fn selected_reuse_basis(&self) -> Result<runmat_execution::Digest> {
        match self {
            Self::Configured(prepared) => prepared.selected_reuse_basis(),
            Self::Admitted(prepared) => Ok(prepared.selected_reuse_basis()),
            Self::RuntimeOnly {
                config, artifacts, ..
            } => {
                let python = runmat_types::ForeignAdapterId::new(runmat_python::PYTHON_ADAPTER_ID)?;
                let python_required = artifacts
                    .interop
                    .adapters
                    .iter()
                    .any(|requirement| requirement.adapter == python);
                selected_artifact_basis(artifacts, config, python_required)
            }
            Self::Archived { bundle, .. } => Ok(bundle.identity()?),
        }
    }

    pub(crate) fn foreign_objects(&self) -> Vec<LogicalObject> {
        match self {
            Self::Configured(prepared) => prepared.artifacts().objects.clone(),
            Self::Admitted(prepared) => prepared.foreign_objects(),
            Self::RuntimeOnly { artifacts, .. } => artifacts.objects.clone(),
            Self::Archived { selection, .. } => selection.objects().cloned().collect(),
        }
    }

    pub(crate) fn policy_object(&self) -> Option<&LogicalObject> {
        match self {
            Self::Configured(prepared) => Some(prepared.policy_object()),
            Self::Admitted(prepared) => prepared.policy_object(),
            Self::RuntimeOnly { policy_object, .. } => Some(policy_object),
            Self::Archived { selection, .. } => selection.policy_object(),
        }
    }

    pub(crate) fn foreign_closures(&self) -> Vec<ForeignArtifactClosure> {
        match self {
            Self::Configured(prepared) => prepared.artifacts().closures.clone(),
            Self::Admitted(prepared) => prepared.foreign_closures(),
            Self::RuntimeOnly { artifacts, .. } => artifacts.closures.clone(),
            Self::Archived { selection, .. } => selection.closures().cloned().collect(),
        }
    }
}
