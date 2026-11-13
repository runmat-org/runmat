//! One authenticated recipe-to-foreign-byte selection shared by Direct host
//! execution and later selected-unit publication.

use std::borrow::Borrow;
use std::collections::BTreeSet;
use std::sync::Arc;

use super::{ExecutionBundle, ForeignArtifactClosure};
use crate::{
    ArtifactError, ArtifactResult, LogicalObject, ObjectDescriptor, ObjectNamespace,
    ProgramBuildRecipe,
};
use runmat_execution::ExecutableUnitEnvelope;
use runmat_types::InteropManifest;

/// One immutable view of the invocation-captured preparation. The descriptor
/// comes from that preparation's parent receipt, never from child selection;
/// the object must be its exact bounded authenticated payload.
pub struct PreparedForeignParentView<'a> {
    interop: &'a InteropManifest,
    policy_object: Option<&'a LogicalObject>,
    objects: &'a [&'a LogicalObject],
    closures: &'a [&'a ForeignArtifactClosure],
}

impl<'a> PreparedForeignParentView<'a> {
    pub fn new(
        interop: &'a InteropManifest,
        policy_descriptor: Option<&ObjectDescriptor>,
        policy_object: Option<&'a LogicalObject>,
        objects: &'a [&'a LogicalObject],
        closures: &'a [&'a ForeignArtifactClosure],
    ) -> ArtifactResult<Self> {
        interop
            .validate()
            .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
        match (policy_descriptor, policy_object) {
            (Some(descriptor), Some(object)) if descriptor == &object.descriptor => {
                crate::object::validate_foreign_policy_object(object)?;
            }
            (None, None) if *interop == InteropManifest::empty() => {}
            _ => {
                return Err(ArtifactError::Identity(
                    "captured parent policy descriptor and object differ or are missing".into(),
                ))
            }
        }
        selected_indices(interop, objects, closures)?;
        Ok(Self {
            interop,
            policy_object,
            objects,
            closures,
        })
    }
}

/// Bytes selected from one captured, prepared project for a newly compiled
/// child. This is distinct from reuse of an already archived selected body:
/// only the new compiler-sealed unit may introduce additional type needs.
pub struct SelectedPreparedForeignMaterial<'a> {
    interop: InteropManifest,
    objects: Vec<&'a LogicalObject>,
    closures: Vec<&'a ForeignArtifactClosure>,
    policy_object: Option<&'a LogicalObject>,
}

impl<'a> SelectedPreparedForeignMaterial<'a> {
    pub fn for_compiled_unit(
        canonical_v3: &[u8],
        parent: &PreparedForeignParentView<'a>,
    ) -> ArtifactResult<Self> {
        let unit = ExecutableUnitEnvelope::from_canonical_bytes(canonical_v3)
            .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
        let child = &unit.manifest.interop;
        let merged = InteropManifest::merge([(*parent.interop).clone(), child.clone()])
            .map_err(|error| ArtifactError::Identity(error.to_string()))?;
        let mut permitted = (*parent.interop).clone();
        // Foreign type requirements come from the newly sealed executable,
        // never from an old archive. Adapter, contract, and byte grants still
        // must be contained by the captured prepared project.
        permitted.foreign_types = merged.foreign_types.clone();
        if merged != permitted {
            return Err(ArtifactError::Identity(
                "compiled child exceeds captured foreign adapter or artifact grants".into(),
            ));
        }
        let (closure_indices, object_indices) =
            selected_indices(child, parent.objects, parent.closures)?;
        Ok(Self {
            interop: child.clone(),
            objects: object_indices
                .into_iter()
                .map(|index| parent.objects[index])
                .collect(),
            closures: closure_indices
                .into_iter()
                .map(|index| parent.closures[index])
                .collect(),
            policy_object: parent.policy_object,
        })
    }

    pub fn into_parts(
        self,
    ) -> (
        InteropManifest,
        Vec<&'a LogicalObject>,
        Vec<&'a ForeignArtifactClosure>,
        Option<&'a LogicalObject>,
    ) {
        (
            self.interop,
            self.objects,
            self.closures,
            self.policy_object,
        )
    }
}

enum BundleOwner<'a> {
    Borrowed(&'a ExecutionBundle),
    Owned(Arc<ExecutionBundle>),
}

impl BundleOwner<'_> {
    fn bundle(&self) -> &ExecutionBundle {
        match self {
            Self::Borrowed(bundle) => bundle,
            Self::Owned(bundle) => bundle,
        }
    }
}

/// Indices are private and tied to the exact borrowed or Arc-owned bundle.
/// No caller-defined owner can change the indexed bundle after validation.
pub struct SelectedForeignProjection<'a> {
    owner: BundleOwner<'a>,
    closure_indices: Vec<usize>,
    object_indices: Vec<usize>,
    policy_object_index: Option<usize>,
}

impl ExecutionBundle {
    pub fn selected_foreign_projection(
        &self,
        recipe: &ProgramBuildRecipe,
    ) -> ArtifactResult<SelectedForeignProjection<'_>> {
        SelectedForeignProjection::new(BundleOwner::Borrowed(self), recipe)
    }
}

impl SelectedForeignProjection<'static> {
    pub fn from_owned(
        bundle: Arc<ExecutionBundle>,
        recipe: &ProgramBuildRecipe,
    ) -> ArtifactResult<Self> {
        Self::new(BundleOwner::Owned(bundle), recipe)
    }

    /// Select only sealed child requirements contained by one exact parent
    /// recipe. Its admitted bundle remains the authority for all byte indices.
    pub fn from_owned_subset(
        bundle: Arc<ExecutionBundle>,
        parent_recipe: &ProgramBuildRecipe,
        child_interop: &InteropManifest,
    ) -> ArtifactResult<Self> {
        bundle.validate()?;
        parent_recipe.validate()?;
        child_interop
            .validate()
            .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
        if !bundle.manifest.recipes.contains(parent_recipe) {
            return Err(ArtifactError::Identity(
                "selected parent recipe is absent from its admitted bundle".into(),
            ));
        }
        require_subset(&parent_recipe.interop, child_interop)?;
        Self::select(
            BundleOwner::Owned(bundle),
            child_interop,
            parent_recipe.foreign_policy.as_ref(),
        )
    }
}

impl<'a> SelectedForeignProjection<'a> {
    fn new(owner: BundleOwner<'a>, recipe: &ProgramBuildRecipe) -> ArtifactResult<Self> {
        let source = owner.bundle();
        source.validate()?;
        recipe.validate()?;
        if !source.manifest.recipes.contains(recipe) {
            return Err(ArtifactError::Identity(
                "selected Direct recipe is absent from its admitted bundle".into(),
            ));
        }
        Self::select(owner, &recipe.interop, recipe.foreign_policy.as_ref())
    }

    fn select(
        owner: BundleOwner<'a>,
        interop: &InteropManifest,
        policy: Option<&crate::ObjectDescriptor>,
    ) -> ArtifactResult<Self> {
        let source = owner.bundle();
        let (closure_indices, object_indices) = selected_indices(
            interop,
            &source.objects,
            &source.manifest.foreign_artifact_closures,
        )?;
        let policy_object_index = match policy {
            Some(descriptor) => Some(
                source
                    .objects
                    .iter()
                    .position(|object| &object.descriptor == descriptor)
                    .ok_or_else(|| {
                        ArtifactError::Identity("selected recipe policy object is absent".into())
                    })?,
            ),
            None if *interop == InteropManifest::empty() => None,
            None => {
                return Err(ArtifactError::Identity(
                    "selected foreign recipe has no policy".into(),
                ))
            }
        };
        Ok(Self {
            owner,
            closure_indices,
            object_indices,
            policy_object_index,
        })
    }

    pub fn objects(&self) -> impl Iterator<Item = &LogicalObject> {
        self.object_indices
            .iter()
            .map(|index| &self.owner.bundle().objects[*index])
    }

    pub fn closures(&self) -> impl Iterator<Item = &ForeignArtifactClosure> {
        self.closure_indices
            .iter()
            .map(|index| &self.owner.bundle().manifest.foreign_artifact_closures[*index])
    }

    pub fn policy_object(&self) -> Option<&LogicalObject> {
        self.policy_object_index
            .map(|index| &self.owner.bundle().objects[index])
    }
}

fn require_subset(parent: &InteropManifest, child: &InteropManifest) -> ArtifactResult<()> {
    let merged = InteropManifest::merge([parent.clone(), child.clone()])
        .map_err(|error| ArtifactError::Identity(error.to_string()))?;
    // The archived parent is the authority for the complete foreign type,
    // adapter, contract, and artifact requirement set. A selected child can
    // consume less, but cannot mint a new type contract through inference.
    if merged != *parent {
        return Err(ArtifactError::Identity(
            "child selected interop exceeds its exact parent recipe".into(),
        ));
    }
    Ok(())
}

fn selected_indices<O, C>(
    interop: &InteropManifest,
    objects: &[O],
    closures: &[C],
) -> ArtifactResult<(Vec<usize>, Vec<usize>)>
where
    O: Borrow<LogicalObject>,
    C: Borrow<ForeignArtifactClosure>,
{
    let required = interop
        .adapters
        .iter()
        .flat_map(|adapter| {
            adapter
                .artifact_identities
                .iter()
                .cloned()
                .map(|identity| (adapter.adapter.clone(), identity))
        })
        .collect::<BTreeSet<_>>();
    let closure_indices = closures
        .iter()
        .enumerate()
        .filter_map(|(index, closure)| {
            let closure = closure.borrow();
            required
                .contains(&(closure.adapter.clone(), closure.identity.clone()))
                .then_some(index)
        })
        .collect::<Vec<_>>();
    let selected = closure_indices
        .iter()
        .map(|index| {
            let closure: &ForeignArtifactClosure = closures[*index].borrow();
            (closure.adapter.clone(), closure.identity.clone())
        })
        .collect::<BTreeSet<_>>();
    if selected != required || closure_indices.len() != required.len() {
        return Err(ArtifactError::Identity(
            "selected Direct recipe has an incomplete foreign closure".into(),
        ));
    }
    let digests = closure_indices
        .iter()
        .flat_map(|index| closures[*index].borrow().object_digests.iter().copied())
        .collect::<BTreeSet<_>>();
    let object_indices = objects
        .iter()
        .enumerate()
        .filter_map(|(index, object)| {
            let object = object.borrow();
            (object.descriptor.namespace == ObjectNamespace::ForeignArtifact
                && digests.contains(&object.descriptor.digest))
            .then_some(index)
        })
        .collect::<Vec<_>>();
    let selected_digests = object_indices
        .iter()
        .map(|index| objects[*index].borrow().descriptor.digest)
        .collect::<BTreeSet<_>>();
    if selected_digests != digests || object_indices.len() != digests.len() {
        return Err(ArtifactError::Identity(
            "selected Direct foreign object bytes are incomplete".into(),
        ));
    }
    for index in &closure_indices {
        closures[*index].borrow().validate()?;
    }
    for index in &object_indices {
        objects[*index].borrow().validate()?;
    }
    Ok((closure_indices, object_indices))
}
