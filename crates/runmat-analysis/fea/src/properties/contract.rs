use std::collections::BTreeMap;

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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "CompiledPropertyMapData")]
pub struct CompiledPropertyMap {
    pub identity: CompiledPropertyMapIdentity,
    pub audit_digest: Digest,
    #[serde(with = "ordered_map_serde")]
    solid_materials: BTreeMap<StableDigest, CompiledElementMaterial>,
    #[serde(with = "ordered_map_serde")]
    beam_properties: BTreeMap<StableDigest, CompiledBeamProperty>,
    #[serde(with = "ordered_map_serde")]
    shell_properties: BTreeMap<StableDigest, CompiledShellProperty>,
    audit: Vec<CompiledPropertyAudit>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CompiledPropertyMapData {
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

mod ordered_map_serde {
    use std::collections::BTreeMap;

    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S, K, V>(map: &BTreeMap<K, V>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
        K: Serialize + Ord,
        V: Serialize,
    {
        map.iter().collect::<Vec<_>>().serialize(serializer)
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
        beam_properties: BTreeMap<StableDigest, CompiledBeamProperty>,
        shell_properties: BTreeMap<StableDigest, CompiledShellProperty>,
        audit: Vec<CompiledPropertyAudit>,
    ) -> Result<Self, String> {
        let semantic_digest = super::identity::semantic_digest(
            &source,
            &solid_materials,
            &beam_properties,
            &shell_properties,
        )?;
        let audit_digest = super::identity::audit_digest(semantic_digest, &audit)?;
        let compiled = Self {
            identity: CompiledPropertyMapIdentity {
                geometry_id: source.geometry_id,
                geometry_revision: source.geometry_revision,
                mesh_digest: source.mesh_digest,
                explicit_topology_digest: source.explicit_topology_digest,
                semantic_digest,
            },
            audit_digest,
            solid_materials,
            beam_properties,
            shell_properties,
            audit,
        };
        compiled.validate_integrity()?;
        Ok(compiled)
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
            solid_materials: data.solid_materials,
            beam_properties: data.beam_properties,
            shell_properties: data.shell_properties,
            audit: data.audit,
        };
        compiled.validate_integrity()?;
        Ok(compiled)
    }
}
