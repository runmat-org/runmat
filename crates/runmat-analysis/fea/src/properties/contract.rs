#[path = "storage.rs"]
mod storage;
#[path = "wire.rs"]
mod wire;
use std::{collections::BTreeMap, sync::Arc};

use runmat_analysis_core::{
    BeamSectionModel, EvidenceConfidence, LaminatePlyModel, LaminateShellSectionModel,
    MaterialModel,
};
use runmat_execution::Digest;
use runmat_meshing_core::StableDigest;
use serde::{Deserialize, Serialize};

/// Placement-neutral semantic identity of one compiled property map.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompiledPropertyMapIdentity {
    pub geometry_id: String,
    pub geometry_revision: u32,
    pub mesh_digest: Option<StableDigest>,
    pub explicit_topology_digest: Option<Digest>,
    pub semantic_digest: Digest,
}

pub(crate) struct CompiledPropertySource {
    pub geometry_id: String,
    pub geometry_revision: u32,
    pub mesh_digest: Option<StableDigest>,
    pub explicit_topology_digest: Option<Digest>,
}

/// Audit-only provenance for one element assignment.
///
/// These fields explain how a constitutive owner was selected. They never weight or otherwise
/// modify a physical coefficient.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompiledPropertyAudit {
    pub element_identity: StableDigest,
    pub owner: CompiledPropertyAuditOwner,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum CompiledPropertyAuditOwner {
    MaterialAssignment {
        assignment_index: u64,
        /// Shared with every element of the same assignment: one owner per assignment.
        expected_material_id: Arc<str>,
        assigned_material_id: Arc<str>,
        confidence: EvidenceConfidence,
    },
    Laminate {
        section_id: String,
        ordered_ply_material_ids: Vec<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompiledElementMaterial {
    pub element_id: u64,
    pub element_identity: StableDigest,
    /// Shared with every element of the same material: one table entry, many references.
    #[serde(with = "shared_material")]
    pub material: Arc<MaterialModel>,
}

mod shared_material {
    use super::*;
    pub(super) fn serialize<S: serde::Serializer>(
        material: &Arc<MaterialModel>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        material.as_ref().serialize(serializer)
    }
    pub(super) fn deserialize<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Arc<MaterialModel>, D::Error> {
        MaterialModel::deserialize(deserializer).map(Arc::new)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompiledBeamProperty {
    pub element_id: String,
    pub element_identity: StableDigest,
    pub material: MaterialModel,
    pub section: BeamSectionModel,
    pub reference_axis: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompiledLaminatePly {
    pub definition: LaminatePlyModel,
    pub material: MaterialModel,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum CompiledShellPropertyOwner {
    Homogeneous {
        material: Box<MaterialModel>,
        section: runmat_analysis_core::HomogeneousShellSectionModel,
    },
    Laminate {
        section: LaminateShellSectionModel,
        plies: Vec<CompiledLaminatePly>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompiledShellProperty {
    pub element_id: String,
    pub element_identity: StableDigest,
    pub reference_axis: [f64; 3],
    pub owner: CompiledShellPropertyOwner,
}

/// Immutable material ownership after exact geometry-to-element projection.
/// Clones share original validated element and audit inventories. Sharing is
/// process-local; the persisted form is the compact table-and-column wire in `wire.rs`.
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledPropertyMap {
    cell_materials: Arc<BTreeMap<StableDigest, CompiledElementMaterial>>,
    pub identity: CompiledPropertyMapIdentity,
    pub audit_digest: Digest,
    solid_materials: Arc<BTreeMap<StableDigest, CompiledElementMaterial>>,
    beam_properties: Arc<BTreeMap<StableDigest, CompiledBeamProperty>>,
    shell_properties: Arc<BTreeMap<StableDigest, CompiledShellProperty>>,
    audit: Arc<Vec<CompiledPropertyAudit>>,
}

impl Serialize for CompiledPropertyMap {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        wire::serialize(self, serializer)
    }
}

impl<'de> Deserialize<'de> for CompiledPropertyMap {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        wire::deserialize(deserializer)
    }
}

impl CompiledPropertyMap {
    pub(crate) fn new(
        source: CompiledPropertySource,
        solid_materials: BTreeMap<StableDigest, CompiledElementMaterial>,
        cell_materials: BTreeMap<StableDigest, CompiledElementMaterial>,
        beam_properties: BTreeMap<StableDigest, CompiledBeamProperty>,
        shell_properties: BTreeMap<StableDigest, CompiledShellProperty>,
        audit: Vec<CompiledPropertyAudit>,
    ) -> Result<Self, String> {
        let semantic_digest = super::identity::semantic_digest(
            &source,
            &solid_materials,
            &cell_materials,
            &beam_properties,
            &shell_properties,
        )?;
        let audit_digest = super::identity::audit_digest(
            semantic_digest,
            &solid_materials,
            &cell_materials,
            &beam_properties,
            &shell_properties,
            &audit,
        )?;
        let compiled = Self {
            identity: CompiledPropertyMapIdentity {
                geometry_id: source.geometry_id,
                geometry_revision: source.geometry_revision,
                mesh_digest: source.mesh_digest,
                explicit_topology_digest: source.explicit_topology_digest,
                semantic_digest,
            },
            audit_digest,
            solid_materials: Arc::new(solid_materials),
            cell_materials: Arc::new(cell_materials),
            beam_properties: Arc::new(beam_properties),
            shell_properties: Arc::new(shell_properties),
            audit: Arc::new(audit),
        };
        // Both digests were derived above from these exact contents; only the structure
        // remains to check. Decoded or shared maps still run the full validate_integrity.
        super::identity::validate_structure(&compiled)?;
        Ok(compiled)
    }

    /// Requested storage introduced by cloning this shared immutable owner.
    /// Element/audit allocations remain shared and must be charged once by their
    /// retaining owner; only the root and cloned identity string are new here.
    pub fn shared_clone_storage_bytes(&self) -> Option<u64> {
        (std::mem::size_of::<Self>() as u64).checked_add(self.identity.geometry_id.len() as u64)
    }

    pub fn validate_integrity(&self) -> Result<(), String> {
        super::identity::validate(self)
    }

    pub fn solid_material(
        &self,
        element_identity: StableDigest,
    ) -> Option<&CompiledElementMaterial> {
        self.solid_materials.get(&element_identity)
    }

    pub fn solid_materials(&self) -> impl ExactSizeIterator<Item = &CompiledElementMaterial> + '_ {
        self.solid_materials.values()
    }

    pub fn cell_material(&self, identity: StableDigest) -> Option<&CompiledElementMaterial> {
        self.cell_materials.get(&identity)
    }

    pub fn cell_materials(&self) -> impl ExactSizeIterator<Item = &CompiledElementMaterial> + '_ {
        self.cell_materials.values()
    }

    pub(super) fn cell_entries(&self) -> &BTreeMap<StableDigest, CompiledElementMaterial> {
        &self.cell_materials
    }

    pub fn beam_property(&self, element_identity: StableDigest) -> Option<&CompiledBeamProperty> {
        self.beam_properties.get(&element_identity)
    }

    pub fn beam_properties(&self) -> impl ExactSizeIterator<Item = &CompiledBeamProperty> + '_ {
        self.beam_properties.values()
    }

    pub fn shell_property(&self, element_identity: StableDigest) -> Option<&CompiledShellProperty> {
        self.shell_properties.get(&element_identity)
    }

    pub fn shell_properties(&self) -> impl ExactSizeIterator<Item = &CompiledShellProperty> + '_ {
        self.shell_properties.values()
    }

    pub fn audit(&self) -> &[CompiledPropertyAudit] {
        &self.audit
    }

    pub(super) fn solid_entries(&self) -> &BTreeMap<StableDigest, CompiledElementMaterial> {
        &self.solid_materials
    }

    pub(super) fn beam_entries(&self) -> &BTreeMap<StableDigest, CompiledBeamProperty> {
        &self.beam_properties
    }

    pub(super) fn shell_entries(&self) -> &BTreeMap<StableDigest, CompiledShellProperty> {
        &self.shell_properties
    }
}
