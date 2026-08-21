use std::collections::BTreeMap;

use runmat_geometry_core::{
    ExactBRepTopology, ExactRegion, ExactSharedInterface, PersistentEntityId, PersistentEntityKind,
    TopologicalOrientation,
};
use sha2::{Digest, Sha256};

use crate::import::GeometryImportError;

pub(super) struct ExactRegionProjection {
    pub regions: Vec<ExactRegion>,
    pub interfaces: Vec<ExactSharedInterface>,
}

/// Projects one authoritative region per exact solid and only topologically conformal interfaces.
/// Geometrically coincident independent faces remain independent and require an explicit contact
/// contract.
pub(super) fn project_regions_and_interfaces(
    topology: &ExactBRepTopology,
) -> Result<ExactRegionProjection, GeometryImportError> {
    let mut regions = topology
        .solids
        .iter()
        .map(|solid| ExactRegion {
            id: region_id(&solid.id),
            solid_id: solid.id.clone(),
        })
        .collect::<Vec<_>>();
    regions.sort_by(|left, right| left.id.cmp(&right.id));
    let region_by_solid = regions
        .iter()
        .map(|region| (&region.solid_id, &region.id))
        .collect::<BTreeMap<_, _>>();

    let mut shell_owners = BTreeMap::<&PersistentEntityId, &PersistentEntityId>::new();
    for solid in &topology.solids {
        for shell_id in std::iter::once(&solid.outer_shell_id).chain(&solid.void_shell_ids) {
            if shell_owners.insert(shell_id, &solid.id).is_some() {
                return Err(invalid("an exact shell belongs to multiple solids"));
            }
        }
    }

    let mut face_uses =
        BTreeMap::<&PersistentEntityId, Vec<(&PersistentEntityId, TopologicalOrientation)>>::new();
    for shell in &topology.shells {
        let owner = shell_owners.get(&shell.id).copied();
        for face_use in &shell.face_uses {
            let Some(owner) = owner else {
                continue;
            };
            face_uses
                .entry(&face_use.entity_id)
                .or_default()
                .push((owner, face_use.orientation));
        }
    }

    let mut interfaces = Vec::new();
    for (face_id, uses) in face_uses {
        match uses.as_slice() {
            [_] => {}
            [(left_solid, left_orientation), (right_solid, right_orientation)] => {
                if left_solid == right_solid {
                    return Err(invalid(
                        "an exact face cannot be an interface between one solid and itself",
                    ));
                }
                if left_orientation == right_orientation {
                    return Err(invalid(
                        "shared exact face uses must have opposite orientations",
                    ));
                }
                let mut sides = [
                    (
                        region_by_solid
                            .get(left_solid)
                            .copied()
                            .cloned()
                            .ok_or_else(|| invalid("exact region projection is incomplete"))?,
                        *left_orientation,
                    ),
                    (
                        region_by_solid
                            .get(right_solid)
                            .copied()
                            .cloned()
                            .ok_or_else(|| invalid("exact region projection is incomplete"))?,
                        *right_orientation,
                    ),
                ];
                sides.sort_by(|left, right| left.0.cmp(&right.0));
                interfaces.push(ExactSharedInterface {
                    face_id: face_id.clone(),
                    side_a_region_id: sides[0].0.clone(),
                    side_b_region_id: sides[1].0.clone(),
                    side_a_orientation: sides[0].1,
                    side_b_orientation: sides[1].1,
                });
            }
            _ => {
                return Err(invalid("an exact face has more than two solid-region uses"));
            }
        }
    }
    Ok(ExactRegionProjection {
        regions,
        interfaces,
    })
}

fn region_id(solid_id: &PersistentEntityId) -> PersistentEntityId {
    let mut digest = Sha256::new();
    digest.update(b"runmat.exact-solid-region\0");
    digest.update((solid_id.source_topology_id.len() as u64).to_be_bytes());
    digest.update(solid_id.source_topology_id.as_bytes());
    for segment in &solid_id.assembly_path {
        digest.update((segment.len() as u64).to_be_bytes());
        digest.update(segment.as_bytes());
    }
    PersistentEntityId {
        kind: PersistentEntityKind::Region,
        source_topology_id: format!("solid-region:{:x}", digest.finalize()),
        assembly_path: solid_id.assembly_path.clone(),
    }
}

fn invalid(reason: &str) -> GeometryImportError {
    GeometryImportError::InvalidGeometry(reason.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_geometry_core::{
        ExactShell, ExactSolid, OrientedEntityUse, EXACT_BREP_TOPOLOGY_SCHEMA_VERSION,
    };

    #[test]
    fn shared_face_projects_one_canonical_oriented_interface() {
        let face = id(PersistentEntityKind::Face, "face");
        let mut topology = empty_topology();
        topology.shells = vec![
            shell("shell-b", face.clone(), TopologicalOrientation::Reversed),
            shell("shell-a", face.clone(), TopologicalOrientation::Forward),
        ];
        topology.solids = vec![solid("solid-b", "shell-b"), solid("solid-a", "shell-a")];

        let projected = project_regions_and_interfaces(&topology).unwrap();
        assert_eq!(projected.regions.len(), 2);
        assert_eq!(projected.interfaces.len(), 1);
        assert_eq!(projected.interfaces[0].face_id, face);
        assert!(
            projected.interfaces[0].side_a_region_id < projected.interfaces[0].side_b_region_id
        );
        assert_ne!(
            projected.interfaces[0].side_a_orientation,
            projected.interfaces[0].side_b_orientation
        );
        assert!(projected
            .regions
            .iter()
            .any(|region| region.id == projected.interfaces[0].side_a_region_id));
        assert!(projected
            .regions
            .iter()
            .any(|region| region.id == projected.interfaces[0].side_b_region_id));
    }

    #[test]
    fn nonmanifold_or_inconsistently_oriented_shared_faces_fail_closed() {
        let face = id(PersistentEntityKind::Face, "face");
        let mut topology = empty_topology();
        topology.shells = vec![
            shell("shell-a", face.clone(), TopologicalOrientation::Forward),
            shell("shell-b", face.clone(), TopologicalOrientation::Forward),
        ];
        topology.solids = vec![solid("solid-a", "shell-a"), solid("solid-b", "shell-b")];
        assert!(project_regions_and_interfaces(&topology).is_err());

        topology
            .shells
            .push(shell("shell-c", face, TopologicalOrientation::Reversed));
        topology.solids.push(solid("solid-c", "shell-c"));
        assert!(project_regions_and_interfaces(&topology).is_err());
    }

    fn empty_topology() -> ExactBRepTopology {
        ExactBRepTopology {
            schema_version: EXACT_BREP_TOPOLOGY_SCHEMA_VERSION,
            root_assembly_id: id(PersistentEntityKind::Assembly, "root"),
            assemblies: Vec::new(),
            instances: Vec::new(),
            bodies: Vec::new(),
            lumps: Vec::new(),
            solids: Vec::new(),
            regions: Vec::new(),
            shells: Vec::new(),
            faces: Vec::new(),
            wires: Vec::new(),
            coedges: Vec::new(),
            edges: Vec::new(),
            vertices: Vec::new(),
            interfaces: Vec::new(),
            contacts: Vec::new(),
        }
    }

    fn solid(name: &str, shell: &str) -> ExactSolid {
        ExactSolid {
            id: id(PersistentEntityKind::Solid, name),
            outer_shell_id: id(PersistentEntityKind::Shell, shell),
            void_shell_ids: Vec::new(),
        }
    }

    fn shell(
        name: &str,
        face_id: PersistentEntityId,
        orientation: TopologicalOrientation,
    ) -> ExactShell {
        ExactShell {
            id: id(PersistentEntityKind::Shell, name),
            orientation: TopologicalOrientation::Forward,
            face_uses: vec![OrientedEntityUse {
                entity_id: face_id,
                orientation,
            }],
        }
    }

    fn id(kind: PersistentEntityKind, name: &str) -> PersistentEntityId {
        PersistentEntityId {
            kind,
            source_topology_id: name.into(),
            assembly_path: vec!["part".into()],
        }
    }
}
