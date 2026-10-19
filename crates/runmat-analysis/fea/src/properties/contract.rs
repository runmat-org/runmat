#[path = "storage.rs"]
mod storage;
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
        expected_material_id: String,
        assigned_material_id: String,
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
    pub material: MaterialModel,
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
/// process-local; the canonical wire contract and cold integrity checks are unchanged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "CompiledPropertyMapData")]
pub struct CompiledPropertyMap {
    #[serde(with = "ordered_map_serde", skip_serializing_if = "BTreeMap::is_empty")]
    cell_materials: Arc<BTreeMap<StableDigest, CompiledElementMaterial>>,
    pub identity: CompiledPropertyMapIdentity,
    pub audit_digest: Digest,
    #[serde(with = "ordered_map_serde")]
    solid_materials: Arc<BTreeMap<StableDigest, CompiledElementMaterial>>,
    #[serde(with = "ordered_map_serde")]
    beam_properties: Arc<BTreeMap<StableDigest, CompiledBeamProperty>>,
    #[serde(with = "ordered_map_serde")]
    shell_properties: Arc<BTreeMap<StableDigest, CompiledShellProperty>>,
    #[serde(serialize_with = "serialize_shared_audit")]
    audit: Arc<Vec<CompiledPropertyAudit>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CompiledPropertyMapData {
    #[serde(default, with = "ordered_map_serde")]
    cell_materials: BTreeMap<StableDigest, CompiledElementMaterial>,
    identity: CompiledPropertyMapIdentity,
    audit_digest: Digest,
    #[serde(with = "ordered_map_serde")]
    solid_materials: BTreeMap<StableDigest, CompiledElementMaterial>,
    #[serde(with = "ordered_map_serde")]
    beam_properties: BTreeMap<StableDigest, CompiledBeamProperty>,
    #[serde(with = "ordered_map_serde")]
    shell_properties: BTreeMap<StableDigest, CompiledShellProperty>,
    audit: Vec<CompiledPropertyAudit>,
}

fn serialize_shared_audit<S: serde::Serializer>(
    audit: &Arc<Vec<CompiledPropertyAudit>>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    audit.as_ref().serialize(serializer)
}

mod ordered_map_serde {
    use std::collections::BTreeMap;

    use serde::{ser::SerializeSeq, Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S, K, V>(map: &BTreeMap<K, V>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
        K: Serialize + Ord,
        V: Serialize,
    {
        let mut sequence = serializer.serialize_seq(Some(map.len()))?;
        for entry in map {
            sequence.serialize_element(&entry)?;
        }
        sequence.end()
    }

    pub fn deserialize<'de, D, K, V>(deserializer: D) -> Result<BTreeMap<K, V>, D::Error>
    where
        D: Deserializer<'de>,
        K: Deserialize<'de> + Ord,
        V: Deserialize<'de>,
    {
        let entries = Vec::<(K, V)>::deserialize(deserializer)?;
        let entry_count = entries.len();
        let map = entries.into_iter().collect::<BTreeMap<_, _>>();
        if map.len() != entry_count {
            return Err(serde::de::Error::custom(
                "compiled property map contains duplicate element identities",
            ));
        }
        Ok(map)
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
        compiled.validate_integrity()?;
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

impl TryFrom<CompiledPropertyMapData> for CompiledPropertyMap {
    type Error = String;

    fn try_from(data: CompiledPropertyMapData) -> Result<Self, Self::Error> {
        let compiled = Self {
            identity: data.identity,
            audit_digest: data.audit_digest,
            solid_materials: Arc::new(data.solid_materials),
            cell_materials: Arc::new(data.cell_materials),
            beam_properties: Arc::new(data.beam_properties),
            shell_properties: Arc::new(data.shell_properties),
            audit: Arc::new(data.audit),
        };
        compiled.validate_integrity()?;
        Ok(compiled)
    }
}
