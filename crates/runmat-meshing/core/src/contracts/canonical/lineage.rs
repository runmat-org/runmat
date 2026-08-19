use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::{
    solver_volume_element_identity, CanonicalMeshingContract, GeometryRevisionRef,
    MeshingContractError, PersistentEntityId, PersistentEntityKind, SolverMeshArtifact,
    SolverMeshTransferMap, StableDigest,
};

pub const SOLVER_MESH_ADAPTATION_LINEAGE_SCHEMA_VERSION: u16 = 1;
const MAX_ADAPTATION_ENTITIES: usize = 20_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SolverMeshAdaptationKind {
    HRefinement,
    HCoarsening,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverMeshAdaptationMark {
    pub element_stable_identity: StableDigest,
    pub indicator_value: f64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverMeshAdaptationCell {
    pub stable_identity: StableDigest,
    pub node_identities: [StableDigest; 4],
    pub region_id: PersistentEntityId,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverMeshAdaptationMutation {
    pub source_mark_identity: Option<StableDigest>,
    pub node_identity: StableDigest,
    pub node_coordinates_m: [f64; 3],
    pub removed_cells: Vec<SolverMeshAdaptationCell>,
    pub created_cells: Vec<SolverMeshAdaptationCell>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverMeshAdaptationLineage {
    pub schema_version: u16,
    pub source_artifact_digest: StableDigest,
    pub target_artifact_digest: StableDigest,
    pub transfer_map_digest: StableDigest,
    pub geometry: GeometryRevisionRef,
    pub kind: SolverMeshAdaptationKind,
    pub marks: Vec<SolverMeshAdaptationMark>,
    pub requested_removal_node_identities: Vec<StableDigest>,
    pub mutations: Vec<SolverMeshAdaptationMutation>,
}

impl SolverMeshAdaptationLineage {
    pub fn validate(&self) -> Result<(), MeshingContractError> {
        if self.schema_version != SOLVER_MESH_ADAPTATION_LINEAGE_SCHEMA_VERSION {
            return Err(invalid(
                "solver adaptation lineage schema version",
                format!("expected {SOLVER_MESH_ADAPTATION_LINEAGE_SCHEMA_VERSION}"),
            ));
        }
        self.source_artifact_digest
            .validate_nonzero("lineage.source_artifact_digest")?;
        self.target_artifact_digest
            .validate_nonzero("lineage.target_artifact_digest")?;
        self.transfer_map_digest
            .validate_nonzero("lineage.transfer_map_digest")?;
        if self.source_artifact_digest == self.target_artifact_digest {
            return Err(invalid(
                "solver adaptation lineage",
                "source and target artifacts must be distinct",
            ));
        }
        self.geometry.validate()?;
        validate_marks(&self.marks)?;
        validate_removals(&self.requested_removal_node_identities)?;
        match self.kind {
            SolverMeshAdaptationKind::HRefinement
                if self.marks.is_empty() || !self.requested_removal_node_identities.is_empty() =>
            {
                return Err(invalid(
                    "solver refinement lineage",
                    "refinement requires marks and forbids removal requests",
                ));
            }
            SolverMeshAdaptationKind::HCoarsening
                if !self.marks.is_empty() || self.requested_removal_node_identities.is_empty() =>
            {
                return Err(invalid(
                    "solver coarsening lineage",
                    "coarsening requires removals and forbids marks",
                ));
            }
            _ => {}
        }
        if self.mutations.is_empty() || self.mutations.len() > MAX_ADAPTATION_ENTITIES {
            return Err(invalid(
                "solver adaptation mutations",
                "mutation inventory must be nonempty and bounded",
            ));
        }
        validate_mutations(self)
    }

    pub fn validate_against(
        &self,
        source: &SolverMeshArtifact,
        target: &SolverMeshArtifact,
        transfer: &SolverMeshTransferMap,
    ) -> Result<(), MeshingContractError> {
        self.validate()?;
        source.validate()?;
        target.validate()?;
        transfer.validate_against(source, target)?;
        if self.source_artifact_digest != source.canonical_digest
            || self.target_artifact_digest != target.canonical_digest
            || self.transfer_map_digest != transfer.canonical_digest()?
            || self.geometry != source.geometry
            || self.geometry != target.geometry
        {
            return Err(invalid(
                "solver adaptation lineage binding",
                "artifact, transfer, and geometry identities must match",
            ));
        }
        validate_artifact_replay(self, source, target)
    }
}

fn validate_marks(marks: &[SolverMeshAdaptationMark]) -> Result<(), MeshingContractError> {
    if marks.len() > MAX_ADAPTATION_ENTITIES {
        return Err(invalid(
            "solver adaptation marks",
            "inventory exceeds its bound",
        ));
    }
    let mut previous = None;
    let mut identities = BTreeSet::new();
    for mark in marks {
        if mark.element_stable_identity == StableDigest::ZERO
            || !mark.indicator_value.is_finite()
            || mark.indicator_value <= 0.0
            || !identities.insert(mark.element_stable_identity)
            || previous.is_some_and(|prior: &SolverMeshAdaptationMark| {
                prior.indicator_value < mark.indicator_value
                    || (prior.indicator_value == mark.indicator_value
                        && prior.element_stable_identity >= mark.element_stable_identity)
            })
        {
            return Err(invalid(
                "solver adaptation marks",
                "marks must be positive, unique, and ordered by descending indicator then identity",
            ));
        }
        previous = Some(mark);
    }
    Ok(())
}

fn validate_removals(removals: &[StableDigest]) -> Result<(), MeshingContractError> {
    if removals.len() > MAX_ADAPTATION_ENTITIES
        || removals.contains(&StableDigest::ZERO)
        || !removals.windows(2).all(|pair| pair[0] < pair[1])
    {
        return Err(invalid(
            "solver adaptation removal requests",
            "identities must be nonzero, unique, canonical, and bounded",
        ));
    }
    Ok(())
}

fn validate_mutations(lineage: &SolverMeshAdaptationLineage) -> Result<(), MeshingContractError> {
    let marks = lineage
        .marks
        .iter()
        .enumerate()
        .map(|(index, mark)| (mark.element_stable_identity, index))
        .collect::<BTreeMap<_, _>>();
    let removals = lineage
        .requested_removal_node_identities
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut nodes = BTreeSet::new();
    let mut mutation_marks = BTreeSet::new();
    let mut previous_mark_index = None;
    for mutation in &lineage.mutations {
        if mutation.node_identity == StableDigest::ZERO
            || mutation
                .node_coordinates_m
                .iter()
                .any(|coordinate| !coordinate.is_finite())
            || !nodes.insert(mutation.node_identity)
            || mutation.removed_cells.is_empty()
            || mutation.created_cells.is_empty()
        {
            return Err(invalid(
                "solver adaptation mutation",
                "node and nonempty cell delta must be finite, nonzero, and unique",
            ));
        }
        validate_cells("removed adaptation cells", &mutation.removed_cells)?;
        validate_cells("created adaptation cells", &mutation.created_cells)?;
        let removed = mutation
            .removed_cells
            .iter()
            .map(|cell| cell.stable_identity)
            .collect::<BTreeSet<_>>();
        if mutation
            .created_cells
            .iter()
            .any(|cell| removed.contains(&cell.stable_identity))
        {
            return Err(invalid(
                "solver adaptation mutation",
                "removed and created cell inventories must be disjoint",
            ));
        }
        match lineage.kind {
            SolverMeshAdaptationKind::HRefinement => {
                let mark = mutation.source_mark_identity.ok_or_else(|| {
                    invalid("solver refinement mutation", "source mark is required")
                })?;
                let Some(mark_index) = marks.get(&mark).copied() else {
                    return Err(invalid(
                        "solver refinement mutation",
                        "source mark is not in the admitted mark inventory",
                    ));
                };
                if !mutation_marks.insert(mark)
                    || previous_mark_index.is_some_and(|previous| previous >= mark_index)
                {
                    return Err(invalid(
                        "solver refinement mutation",
                        "source marks must be unique and follow canonical mark order",
                    ));
                }
                previous_mark_index = Some(mark_index);
            }
            SolverMeshAdaptationKind::HCoarsening => {
                if mutation.source_mark_identity.is_some()
                    || !removals.contains(&mutation.node_identity)
                {
                    return Err(invalid(
                        "solver coarsening mutation",
                        "mutations must correspond exactly to requested removal nodes",
                    ));
                }
            }
        }
    }
    if lineage.kind == SolverMeshAdaptationKind::HCoarsening && nodes != removals {
        return Err(invalid(
            "solver coarsening mutation",
            "every requested removal must have exactly one mutation",
        ));
    }
    Ok(())
}

fn validate_cells(
    field: &str,
    cells: &[SolverMeshAdaptationCell],
) -> Result<(), MeshingContractError> {
    if cells.len() > MAX_ADAPTATION_ENTITIES
        || !cells
            .windows(2)
            .all(|pair| pair[0].stable_identity < pair[1].stable_identity)
    {
        return Err(invalid(
            field,
            "cells must be unique, canonical, and bounded",
        ));
    }
    for cell in cells {
        if cell.stable_identity == StableDigest::ZERO
            || cell.node_identities.contains(&StableDigest::ZERO)
            || cell
                .node_identities
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
                .len()
                != 4
            || cell.stable_identity != solver_volume_element_identity(cell.node_identities)
            || cell.region_id.kind != PersistentEntityKind::Region
        {
            return Err(invalid(
                field,
                "cell identity, corner inventory, or region is invalid",
            ));
        }
        cell.region_id.validate()?;
    }
    Ok(())
}

fn validate_artifact_replay(
    lineage: &SolverMeshAdaptationLineage,
    source: &SolverMeshArtifact,
    target: &SolverMeshArtifact,
) -> Result<(), MeshingContractError> {
    let source_cells = artifact_cells(source);
    let target_cells = artifact_cells(target);
    let source_nodes = artifact_coordinates(source);
    let target_nodes = artifact_coordinates(target);
    let region_materials = |artifact: &SolverMeshArtifact| {
        artifact
            .topology
            .regions
            .iter()
            .map(|region| (region.region_id.clone(), region.material_id.clone()))
            .collect::<BTreeMap<_, _>>()
    };
    if region_materials(source) != region_materials(target) {
        return Err(invalid(
            "solver adaptation region materials",
            "interior adaptation must preserve region material assignments",
        ));
    }
    if lineage.kind == SolverMeshAdaptationKind::HRefinement
        && lineage
            .marks
            .iter()
            .any(|mark| !source_cells.contains_key(&mark.element_stable_identity))
    {
        return Err(invalid(
            "solver adaptation marks",
            "every mark must identify a source artifact element",
        ));
    }
    let mut state = source_cells;
    for mutation in &lineage.mutations {
        let expected_coordinates = match lineage.kind {
            SolverMeshAdaptationKind::HRefinement => target_nodes.get(&mutation.node_identity),
            SolverMeshAdaptationKind::HCoarsening => source_nodes.get(&mutation.node_identity),
        };
        if expected_coordinates.copied() != Some(mutation.node_coordinates_m)
            || match lineage.kind {
                SolverMeshAdaptationKind::HRefinement => {
                    source_nodes.contains_key(&mutation.node_identity)
                }
                SolverMeshAdaptationKind::HCoarsening => {
                    target_nodes.contains_key(&mutation.node_identity)
                }
            }
        {
            return Err(invalid(
                "solver adaptation node replay",
                "mutation node presence and coordinates do not bind source and target artifacts",
            ));
        }
        for removed in &mutation.removed_cells {
            if state.remove(&removed.stable_identity).as_ref() != Some(removed) {
                return Err(invalid(
                    "solver adaptation cell replay",
                    "removed cell is absent or differs from the current state",
                ));
            }
        }
        for created in &mutation.created_cells {
            if state
                .insert(created.stable_identity, created.clone())
                .is_some()
            {
                return Err(invalid(
                    "solver adaptation cell replay",
                    "created cell already exists in the current state",
                ));
            }
        }
    }
    if state != target_cells {
        return Err(invalid(
            "solver adaptation cell replay",
            "mutation sequence does not reconstruct the target artifact",
        ));
    }
    Ok(())
}

fn artifact_cells(
    artifact: &SolverMeshArtifact,
) -> BTreeMap<StableDigest, SolverMeshAdaptationCell> {
    let nodes = artifact_nodes(artifact);
    artifact
        .topology
        .volume_elements
        .iter()
        .map(|element| {
            let cell = SolverMeshAdaptationCell {
                stable_identity: element.stable_identity,
                node_identities: std::array::from_fn(|index| nodes[&element.node_ids[index]].0),
                region_id: element.region_id.clone(),
            };
            (cell.stable_identity, cell)
        })
        .collect()
}

fn artifact_nodes(artifact: &SolverMeshArtifact) -> BTreeMap<u64, (StableDigest, [f64; 3])> {
    artifact
        .topology
        .nodes
        .iter()
        .map(|node| (node.node_id, (node.stable_identity, node.coordinates_m)))
        .collect()
}

fn artifact_coordinates(artifact: &SolverMeshArtifact) -> BTreeMap<StableDigest, [f64; 3]> {
    artifact
        .topology
        .nodes
        .iter()
        .map(|node| (node.stable_identity, node.coordinates_m))
        .collect()
}

fn invalid(field: &str, reason: impl Into<String>) -> MeshingContractError {
    MeshingContractError::invalid(field, reason)
}
