use std::collections::BTreeMap;

use runmat_analysis_core::{EvidenceConfidence, MaterialModel};
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

/// Audit-only provenance for one element assignment.
///
/// These fields explain how a constitutive owner was selected. They never weight or otherwise
/// modify a physical coefficient.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompiledPropertyAudit {
    pub element_identity: StableDigest,
    pub assignment_index: u64,
    pub expected_material_id: String,
    pub assigned_material_id: String,
    pub confidence: EvidenceConfidence,
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
pub struct CompiledExplicitElementMaterial {
    pub element_id: String,
    pub element_identity: StableDigest,
    pub material: MaterialModel,
}

/// Immutable material ownership after exact geometry-to-element projection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompiledPropertyMap {
    pub identity: CompiledPropertyMapIdentity,
    pub audit_digest: Digest,
    solid_materials: BTreeMap<StableDigest, CompiledElementMaterial>,
    beam_materials: BTreeMap<StableDigest, CompiledExplicitElementMaterial>,
    shell_materials: BTreeMap<StableDigest, CompiledExplicitElementMaterial>,
    audit: Vec<CompiledPropertyAudit>,
}

impl CompiledPropertyMap {
    pub(crate) fn new(
        identity: CompiledPropertyMapIdentity,
        audit_digest: Digest,
        solid_materials: BTreeMap<StableDigest, CompiledElementMaterial>,
        beam_materials: BTreeMap<StableDigest, CompiledExplicitElementMaterial>,
        shell_materials: BTreeMap<StableDigest, CompiledExplicitElementMaterial>,
        audit: Vec<CompiledPropertyAudit>,
    ) -> Self {
        Self {
            identity,
            audit_digest,
            solid_materials,
            beam_materials,
            shell_materials,
            audit,
        }
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

    pub fn beam_material(
        &self,
        element_identity: StableDigest,
    ) -> Option<&CompiledExplicitElementMaterial> {
        self.beam_materials.get(&element_identity)
    }

    pub fn beam_materials(
        &self,
    ) -> impl ExactSizeIterator<Item = &CompiledExplicitElementMaterial> + '_ {
        self.beam_materials.values()
    }

    pub fn shell_material(
        &self,
        element_identity: StableDigest,
    ) -> Option<&CompiledExplicitElementMaterial> {
        self.shell_materials.get(&element_identity)
    }

    pub fn shell_materials(
        &self,
    ) -> impl ExactSizeIterator<Item = &CompiledExplicitElementMaterial> + '_ {
        self.shell_materials.values()
    }

    pub fn audit(&self) -> &[CompiledPropertyAudit] {
        &self.audit
    }
}
