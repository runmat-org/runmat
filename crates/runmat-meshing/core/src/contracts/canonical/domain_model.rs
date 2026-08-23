use runmat_geometry_core::{ExactBRepTopology, PersistentEntityId, PersistentEntityKind};
use serde::{Deserialize, Serialize};

use super::{validate_token, MeshingContractError};

pub const MESHING_DOMAIN_MODEL_SCHEMA_VERSION: u16 = 1;
// The canonical REQUEST codec admits at most 100,000 aggregate collection items. Keep the two
// top-level inventories below that ceiling so nested persistent identities remain bounded too.
const MAX_REGION_MATERIALS: usize = 65_536;
const MAX_CONTACTS: usize = 32_768;

/// Solver-facing classifications that are authored outside geometry but participate in mesh
/// identity. Exact contact pairing remains in the geometry topology; this model selects the
/// complete admitted contact inventory without duplicating its face-to-face definition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshingDomainModel {
    pub schema_version: u16,
    pub region_materials: Vec<RegionMaterialAssignment>,
    pub contact_ids: Vec<PersistentEntityId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegionMaterialAssignment {
    pub region_id: PersistentEntityId,
    pub material_id: String,
}

impl MeshingDomainModel {
    pub fn validate(&self) -> Result<(), MeshingContractError> {
        if self.schema_version != MESHING_DOMAIN_MODEL_SCHEMA_VERSION {
            return Err(invalid(
                "meshing domain model schema",
                "unsupported version",
            ));
        }
        if self.region_materials.len() > MAX_REGION_MATERIALS
            || self.contact_ids.len() > MAX_CONTACTS
        {
            return Err(invalid(
                "meshing domain model inventory",
                "region-material or contact inventory exceeds its hard bound",
            ));
        }
        for assignment in &self.region_materials {
            assignment.region_id.validate()?;
            if assignment.region_id.kind != PersistentEntityKind::Region {
                return Err(invalid(
                    "region material",
                    "assignment source must be a persistent region identity",
                ));
            }
            validate_token("material id", &assignment.material_id, 256)?;
        }
        if !strictly_increasing_by(&self.region_materials, |value| &value.region_id) {
            return Err(invalid(
                "region materials",
                "assignments must be unique and canonically ordered by region identity",
            ));
        }
        for contact_id in &self.contact_ids {
            contact_id.validate()?;
            if contact_id.kind != PersistentEntityKind::Contact {
                return Err(invalid(
                    "meshing contact",
                    "contact inventory may contain only persistent contact identities",
                ));
            }
        }
        if !strictly_increasing_by(&self.contact_ids, |value| value) {
            return Err(invalid(
                "meshing contacts",
                "contact identities must be unique and canonically ordered",
            ));
        }
        Ok(())
    }

    /// Binds authored classifications to one independently admitted exact topology.
    pub fn validate_against_exact_topology(
        &self,
        topology: &ExactBRepTopology,
    ) -> Result<(), MeshingContractError> {
        self.validate()?;
        if self
            .region_materials
            .iter()
            .map(|assignment| &assignment.region_id)
            .ne(topology.regions.iter().map(|region| &region.id))
        {
            return Err(invalid(
                "region materials",
                "every exact region must have exactly one canonical material assignment",
            ));
        }
        if self
            .contact_ids
            .iter()
            .ne(topology.contacts.iter().map(|contact| &contact.id))
        {
            return Err(invalid(
                "meshing contacts",
                "contact inventory must exactly match the admitted exact topology",
            ));
        }
        Ok(())
    }
}

fn strictly_increasing_by<T, K: Ord>(values: &[T], key: impl Fn(&T) -> &K) -> bool {
    values.windows(2).all(|pair| key(&pair[0]) < key(&pair[1]))
}

fn invalid(field: &str, reason: &str) -> MeshingContractError {
    MeshingContractError::invalid(field, reason)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CanonicalMeshingContract;

    #[test]
    fn domain_model_is_canonical_and_bound_to_exact_inventories() {
        let (_, topology, _) = runmat_geometry_fixtures::exact_tetrahedron();
        let model = MeshingDomainModel {
            schema_version: MESHING_DOMAIN_MODEL_SCHEMA_VERSION,
            region_materials: vec![RegionMaterialAssignment {
                region_id: topology.regions[0].id.clone(),
                material_id: "steel".into(),
            }],
            contact_ids: Vec::new(),
        };
        model.validate_against_exact_topology(&topology).unwrap();
        let encoded = model.canonical_encode().unwrap();
        let decoded = MeshingDomainModel::canonical_decode(&encoded).unwrap();
        assert_eq!(decoded, model);
        assert_eq!(decoded.canonical_encode().unwrap(), encoded);

        let mut trailing = encoded;
        trailing.push(0);
        assert!(MeshingDomainModel::canonical_decode(&trailing).is_err());
        let oversized = vec![0; MeshingDomainModel::LIMITS.maximum_encoded_bytes + 1];
        assert!(MeshingDomainModel::canonical_decode(&oversized).is_err());
    }

    #[test]
    fn domain_model_rejects_wrong_order_kind_inventory_and_material() {
        let (_, mut topology, _) = runmat_geometry_fixtures::exact_tetrahedron();
        let region = topology.regions[0].id.clone();
        let mut model = MeshingDomainModel {
            schema_version: MESHING_DOMAIN_MODEL_SCHEMA_VERSION,
            region_materials: vec![RegionMaterialAssignment {
                region_id: region.clone(),
                material_id: "steel".into(),
            }],
            contact_ids: Vec::new(),
        };

        model
            .region_materials
            .push(model.region_materials[0].clone());
        assert!(model.validate().is_err());
        model.region_materials.pop();
        model.region_materials[0].material_id = " bad".into();
        assert!(model.validate().is_err());
        model.region_materials[0].material_id = "steel".into();
        model.region_materials[0].region_id.kind = PersistentEntityKind::Face;
        assert!(model.validate().is_err());
        model.region_materials[0].region_id = region;
        let mut unknown_contact = model.region_materials[0].region_id.clone();
        unknown_contact.kind = PersistentEntityKind::Contact;
        unknown_contact.source_topology_id = "unknown-contact".into();
        model.contact_ids.push(unknown_contact);
        assert!(model.validate_against_exact_topology(&topology).is_err());
        model.contact_ids.clear();
        topology.regions.clear();
        assert!(model.validate_against_exact_topology(&topology).is_err());
    }
}
