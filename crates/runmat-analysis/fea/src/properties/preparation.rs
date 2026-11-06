//! Property selection, mesh validation, explicit identities and owned-materialization admission.
use std::collections::BTreeMap;

use runmat_analysis_core::{AnalysisModel, MaterialModel};
use runmat_meshing_core::{SolverMeshArtifact, StableDigest};

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
        // Selection inventories open the validation phase; that whole phase is admitted below.
        let selection = selection_scratch(model)?;
        let mesh_validation = mesh
            .map(SolverMeshArtifact::validation_resident_bytes_estimate)
            .transpose()
            .map_err(|error| PropertyCompilationError::InvalidMesh(error.to_string()))?
            .unwrap_or(0);
        // This phase precedes inventory construction. Reserve it conservatively alongside the
        // compiler's remaining phases so the thermal caller can compose one admission envelope.
        // Mesh validation is its own phase: selection inventories plus the validation scratch.
        // It is rejected before validation allocates, reporting this governing phase.
        let validation_phase = add(selection, mesh_validation)?;
        require_fits(validation_phase, maximum)?;
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
        // Identity and audit encodings run one element at a time (identity.rs `sequence_digest`)
        // and drop each buffer before the next, so their workspace is the largest single
        // element's direct-encoder profile, never the sum over the mesh. Beam, shell and laminate
        // owners clone their section, material and ply metadata per element; that owned heap is
        // bounded by the same per-element profile and summed. Solid and cell owners share their
        // material and assignment-id owners, charged once below.
        let mut element_workspace = 0_u64;
        let mut owned_occurrences = 0_u64;
        // This pass preserves typed ownership errors and allocates nothing: no owned
        // constitutive, section, or audit metadata has been cloned.
        visit(
            model,
            mesh,
            &materials,
            explicit.as_ref(),
            &mut |selected| {
                entry_count = entry_count.checked_add(1).ok_or_else(overflow)?;
                let element = runmat_canonical_codec::profile_canonical_encoding(
                    &selected,
                    // The selected record is a superset of each identity element's contents.
                    runmat_canonical_codec::CanonicalLimits::new(
                        usize::MAX,
                        usize::MAX,
                        usize::MAX,
                        64,
                    ),
                )
                .map_err(|error| PropertyCompilationError::Resource(error.to_string()))?
                .resident_workspace_bytes_estimate;
                element_workspace = element_workspace.max(element);
                let entry_bytes = match selected {
                    SelectedProperty::Solid { .. } | SelectedProperty::Cell { .. } => {
                        map_entry_bytes::<CompiledElementMaterial>()?
                    }
                    SelectedProperty::Beam { .. } => {
                        owned_occurrences = add(owned_occurrences, element)?;
                        map_entry_bytes::<CompiledBeamProperty>()?
                    }
                    SelectedProperty::Shell { .. } => {
                        owned_occurrences = add(owned_occurrences, element)?;
                        map_entry_bytes::<CompiledShellProperty>()?
                    }
                    SelectedProperty::Laminate { section, .. } => {
                        owned_occurrences = add(owned_occurrences, element)?;
                        add(
                            map_entry_bytes::<CompiledShellProperty>()?,
                            scale(
                                section.plies.len(),
                                std::mem::size_of::<CompiledLaminatePly>() as u64,
                            )?,
                        )?
                    }
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
        // The header contracts are refused above their limit; their encoder holds at most the
        // limit plus its copy slack.
        let header_workspace = scale(
            2,
            super::identity::HEADER_LIMITS.maximum_encoded_bytes as u64,
        )?;
        // One shared pair of id owners per assignment, plus its lookup slot.
        let assignment_ids =
            model
                .material_assignments
                .iter()
                .try_fold(0_u64, |total, assignment| {
                    add(
                        total,
                        (assignment.expected_material_id.len()
                            + assignment.assigned_material_id.len()
                            + 4 * std::mem::size_of::<usize>()
                            + 2 * (std::mem::size_of::<(
                                usize,
                                (std::sync::Arc<str>, std::sync::Arc<str>),
                            )>() + std::mem::size_of::<usize>())) as u64,
                    )
                })?;
        let metadata = add(
            add(element_workspace, header_workspace)?,
            add(owned_occurrences, assignment_ids)?,
        )?;
        // Elements share one owner per material; the table is charged once, not per element.
        let shared_materials = materials.values().try_fold(0_u64, |total, material| {
            add(
                total,
                add(
                    (std::mem::size_of::<MaterialModel>() + 2 * std::mem::size_of::<usize>())
                        as u64,
                    material.owned_buffer_bytes().ok_or_else(overflow)?,
                )?,
            )
        })?;
        // The reservation is the larger of the validation phase and the compilation phase.
        let estimated_resident_bytes = add(
            add(selection, typed_bytes)?,
            add(shared_materials, metadata)?,
        )?
        .max(validation_phase);
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
            output.cells,
            output.beams,
            output.shells,
            output.audit,
        )
        .map_err(PropertyCompilationError::Identity)
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
