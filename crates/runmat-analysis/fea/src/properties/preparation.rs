//! Property selection, explicit identities and owned-materialization admission. Canonical mesh
//! validation remains a separate scratch phase; explicit identities are admitted and cached once.
use std::collections::BTreeMap;

use runmat_analysis_core::{AnalysisModel, MaterialModel};
use runmat_canonical_codec::{serialized_materialization_resident_bytes_estimate, CanonicalLimits};
use runmat_meshing_core::{SolverMeshArtifact, StableDigest};
use serde::{
    ser::{SerializeSeq, SerializeTuple},
    Serialize, Serializer,
};

use super::{
    compile::{material_inventory, validate_mesh, visit, Assignment},
    contract::CompiledPropertySource,
    selection::{OwnedProperties, SelectedProperty},
    CompiledBeamProperty, CompiledElementMaterial, CompiledLaminatePly, CompiledPropertyAudit,
    CompiledPropertyMap, CompiledShellProperty, PropertyCompilationError,
};

pub(super) mod explicit;
use explicit::PreparedExplicitTopology;

/// A validated borrowed input plus its admission evidence. Owned material/section/audit payloads
/// cannot be constructed through this path until `prepare` succeeds at the supplied ceiling.
pub(crate) struct PreparedProperties<'a> {
    model: &'a AnalysisModel,
    mesh: Option<&'a SolverMeshArtifact>,
    materials: BTreeMap<&'a str, &'a MaterialModel>,
    explicit: Option<PreparedExplicitTopology<'a>>,
    entry_count: usize,
    #[cfg(test)]
    pub(crate) selected_metadata_bytes: u64,
    pub(crate) estimated_resident_bytes: u64,
}

impl<'a> PreparedProperties<'a> {
    pub(crate) fn prepare(
        model: &'a AnalysisModel,
        mesh: Option<&'a SolverMeshArtifact>,
        maximum: u64,
    ) -> Result<Self, PropertyCompilationError> {
        // Check lookup/assignment inventory estimates before constructing those inventories.
        // BTree estimates include key/value slots and conservative spare-node/pointer storage;
        // these are resource estimates, not allocator-exact or native/RSS limits.
        let selection = selection_scratch(model)?;
        require_fits(selection, maximum)?;
        validate_mesh(model, mesh)?;
        let explicit = model
            .structural
            .as_ref()
            .map(|structural| {
                PreparedExplicitTopology::prepare(structural, maximum - selection).map_err(
                    |error| match error {
                        PropertyCompilationError::ResourceCeiling { required, .. } => {
                            match add(selection, required) {
                                Ok(required) => {
                                    PropertyCompilationError::ResourceCeiling { required, maximum }
                                }
                                Err(error) => error,
                            }
                        }
                        error => error,
                    },
                )
            })
            .transpose()?;
        let selection = add(
            selection,
            explicit
                .as_ref()
                .map_or(0, |prepared| prepared.estimated_resident_bytes),
        )?;
        let materials = material_inventory(model)?;
        let mut typed_bytes = std::mem::size_of::<CompiledPropertyMap>() as u64;
        let mut entry_count = 0_usize;
        // This pass preserves typed ownership errors before the serializer invokes the same
        // visitor. No owned constitutive, section, or audit metadata has been cloned.
        visit(
            model,
            mesh,
            &materials,
            explicit.as_ref(),
            &mut |selected| {
                entry_count = entry_count.checked_add(1).ok_or_else(overflow)?;
                let entry_bytes = match selected {
                    SelectedProperty::Solid { .. } => map_entry_bytes::<CompiledElementMaterial>()?,
                    SelectedProperty::Beam { .. } => map_entry_bytes::<CompiledBeamProperty>()?,
                    SelectedProperty::Shell { .. } => map_entry_bytes::<CompiledShellProperty>()?,
                    SelectedProperty::Laminate { section, .. } => add(
                        map_entry_bytes::<CompiledShellProperty>()?,
                        scale(
                            section.plies.len(),
                            std::mem::size_of::<CompiledLaminatePly>() as u64,
                        )?,
                    )?,
                };
                typed_bytes = add(
                    typed_bytes,
                    add(
                        entry_bytes,
                        (std::mem::size_of::<CompiledPropertyAudit>()
                            + 2 * std::mem::size_of::<StableDigest>())
                            as u64,
                    )?,
                )?;
                Ok(())
            },
        )?;
        require_fits(add(selection, typed_bytes)?, maximum)?;
        let metadata = serialized_materialization_resident_bytes_estimate(
            &Metadata {
                model,
                mesh,
                materials: &materials,
                explicit: explicit.as_ref(),
            },
            // Expanded selected occurrences are not bounded by the authored workload byte cap.
            CanonicalLimits::new(usize::MAX, usize::MAX, usize::MAX, 64),
        )
        .map_err(|error| PropertyCompilationError::Resource(error.to_string()))?;
        // Compose two shared-codec metadata estimates with the typed storage terms to cover
        // owned metadata plus concurrent identity materialization. The resource view is not the
        // canonical identity contract itself. Semantic/audit encodings and rehashes are sequential,
        // not four simultaneous buffers. Typed map/audit/ply slots are accounted independently.
        let estimated_resident_bytes = add(add(selection, typed_bytes)?, add(metadata, metadata)?)?;
        require_fits(estimated_resident_bytes, maximum)?;
        Ok(Self {
            model,
            mesh,
            materials,
            explicit,
            entry_count,
            #[cfg(test)]
            selected_metadata_bytes: metadata,
            estimated_resident_bytes,
        })
    }

    pub(crate) fn materialize(self) -> Result<CompiledPropertyMap, PropertyCompilationError> {
        let mut output = OwnedProperties::default();
        output
            .audit
            .try_reserve_exact(self.entry_count)
            .map_err(|error| PropertyCompilationError::Resource(error.to_string()))?;
        visit(
            self.model,
            self.mesh,
            &self.materials,
            self.explicit.as_ref(),
            &mut |selected| selected.materialize(&mut output),
        )?;
        CompiledPropertyMap::new(
            CompiledPropertySource {
                geometry_id: self.model.geometry_id.clone(),
                geometry_revision: self.model.geometry_revision,
                mesh_digest: self.mesh.map(|mesh| mesh.canonical_digest),
                explicit_topology_digest: self
                    .explicit
                    .as_ref()
                    .map(|prepared| prepared.topology_digest()),
            },
            output.solids,
            output.beams,
            output.shells,
            output.audit,
        )
        .map_err(PropertyCompilationError::Identity)
    }
}

struct Metadata<'a> {
    model: &'a AnalysisModel,
    mesh: Option<&'a SolverMeshArtifact>,
    materials: &'a BTreeMap<&'a str, &'a MaterialModel>,
    explicit: Option<&'a PreparedExplicitTopology<'a>>,
}
impl Serialize for Metadata<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut tuple = serializer.serialize_tuple(2)?;
        tuple.serialize_element(&self.model.geometry_id)?;
        tuple.serialize_element(&SelectedMetadata(self))?;
        tuple.end()
    }
}
struct SelectedMetadata<'a>(&'a Metadata<'a>);
impl Serialize for SelectedMetadata<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::Error;
        let mut sequence = serializer.serialize_seq(None)?;
        visit(
            self.0.model,
            self.0.mesh,
            self.0.materials,
            self.0.explicit,
            &mut |selected| {
                sequence
                    .serialize_element(&selected)
                    .map_err(|error| PropertyCompilationError::Resource(error.to_string()))
            },
        )
        .map_err(S::Error::custom)?;
        sequence.end()
    }
}

fn selection_scratch(model: &AnalysisModel) -> Result<u64, PropertyCompilationError> {
    let materials = scale(
        model.materials.len(),
        2 * (std::mem::size_of::<(&str, &MaterialModel)>() + 4 * std::mem::size_of::<usize>())
            as u64,
    )?;
    // Solid selection is followed by simultaneous beam/shell selectors. Three sets is a
    // conservative envelope including Vec growth and BTree matched-assignment bookkeeping.
    let assignments = scale(
        model.material_assignments.len(),
        3 * (2 * std::mem::size_of::<Assignment<'_>>() + 8 * std::mem::size_of::<usize>()) as u64,
    )?;
    add(64 * 1024, add(materials, assignments)?)
}
fn map_entry_bytes<T>() -> Result<u64, PropertyCompilationError> {
    // Include spare BTree node slots, not just one densely packed key/value per owner. Three
    // slots per entry plus pointer allowance conservatively covers partial node occupancy;
    // the fixed selection allowance covers small root allocations. Not allocator-exact.
    scale(
        3,
        (std::mem::size_of::<StableDigest>()
            + std::mem::size_of::<T>()
            + 4 * std::mem::size_of::<usize>()) as u64,
    )
}
fn scale(count: usize, width: u64) -> Result<u64, PropertyCompilationError> {
    u64::try_from(count)
        .ok()
        .and_then(|count| count.checked_mul(width))
        .ok_or_else(overflow)
}
fn add(left: u64, right: u64) -> Result<u64, PropertyCompilationError> {
    left.checked_add(right).ok_or_else(overflow)
}
fn overflow() -> PropertyCompilationError {
    PropertyCompilationError::Resource("property compilation residency overflow".into())
}
fn require_fits(required: u64, maximum: u64) -> Result<(), PropertyCompilationError> {
    if required > maximum {
        Err(PropertyCompilationError::ResourceCeiling { required, maximum })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests;
