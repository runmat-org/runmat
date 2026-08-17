use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::{
    BodyMassProperties, ExactAssembly, ExactBody, ExactInstance, ExactMassPropertiesImplementation,
    ExactMassPropertiesRecord, GeometryTransform, PersistentEntityId, PersistentEntityKind,
};
use sha2::{Digest, Sha256};

use super::{exact_projection_identity::*, ffi::bridge};
use crate::import::GeometryImportError;

pub(super) struct OccurrenceIndex<'a> {
    occurrences: BTreeMap<u64, &'a bridge::OcctExactOccurrencePayload>,
    definition_digests: BTreeMap<u64, [u8; 32]>,
    representation_digest: [u8; 32],
}

pub(super) struct AssemblyProjection {
    pub root_id: PersistentEntityId,
    pub assemblies: Vec<ExactAssembly>,
    pub instances: Vec<ExactInstance>,
}

impl<'a> OccurrenceIndex<'a> {
    pub fn new(
        payload: &'a bridge::OcctExactShapePayload,
        representation_digest: [u8; 32],
    ) -> Result<Self, GeometryImportError> {
        let mut definitions = BTreeMap::new();
        for definition in &payload.definitions {
            if definition.definition_index == 0 || definition.representation.is_empty() {
                return Err(invalid("OCCT exact definition inventory is malformed"));
            }
            if definitions
                .insert(
                    definition.definition_index,
                    digest(
                        b"runmat.exact-geometry.occt-definition\0",
                        &definition.representation,
                    ),
                )
                .is_some()
            {
                return Err(invalid("OCCT exact definition index is duplicated"));
            }
        }
        if definitions.keys().copied().ne(1..=definitions.len() as u64) {
            return Err(invalid("OCCT exact definition indices are not contiguous"));
        }

        let mut occurrences = BTreeMap::new();
        for occurrence in &payload.occurrences {
            if occurrence.occurrence_index == 0
                || occurrence.path_segments.is_empty()
                || occurrence.transform.len() != 16
                || (occurrence.definition_index != 0
                    && !definitions.contains_key(&occurrence.definition_index))
            {
                return Err(invalid("OCCT exact occurrence inventory is malformed"));
            }
            if occurrences
                .insert(occurrence.occurrence_index, occurrence)
                .is_some()
            {
                return Err(invalid("OCCT exact occurrence index is duplicated"));
            }
        }
        if occurrences.keys().copied().ne(1..=occurrences.len() as u64) {
            return Err(invalid("OCCT exact occurrence indices are not contiguous"));
        }
        let root = occurrences
            .get(&1)
            .ok_or_else(|| invalid("OCCT exact occurrence root is missing"))?;
        if root.parent_occurrence_index != 0
            || root.path_segments != ["root"]
            || root.definition_index != 0
        {
            return Err(invalid("OCCT exact occurrence root is not canonical"));
        }
        for occurrence in occurrences.values().skip(1) {
            let parent = occurrences
                .get(&occurrence.parent_occurrence_index)
                .ok_or_else(|| invalid("OCCT exact occurrence parent is missing"))?;
            if occurrence.parent_occurrence_index >= occurrence.occurrence_index
                || occurrence.path_segments.len() != parent.path_segments.len() + 1
                || !occurrence.path_segments.starts_with(&parent.path_segments)
            {
                return Err(invalid("OCCT exact occurrence path is not parent-scoped"));
            }
        }
        Ok(Self {
            occurrences,
            definition_digests: definitions,
            representation_digest,
        })
    }

    pub fn path(&self, occurrence_index: u64) -> Result<&[String], GeometryImportError> {
        self.occurrences
            .get(&occurrence_index)
            .map(|occurrence| occurrence.path_segments.as_slice())
            .ok_or_else(|| invalid("OCCT exact topology references an unknown occurrence"))
    }

    pub fn project_assemblies(&self) -> Result<AssemblyProjection, GeometryImportError> {
        let mut child_instances = BTreeMap::<u64, Vec<PersistentEntityId>>::new();
        let mut instances = Vec::new();
        for occurrence in self.occurrences.values().skip(1) {
            let path = occurrence.path_segments.as_slice();
            let id = instance_id(path);
            child_instances
                .entry(occurrence.parent_occurrence_index)
                .or_default()
                .push(id.clone());
            let transform: [f64; 16] = occurrence
                .transform
                .as_slice()
                .try_into()
                .map_err(|_| invalid("OCCT exact occurrence transform is malformed"))?;
            instances.push(ExactInstance {
                id,
                parent_assembly_id: assembly_id(self.path(occurrence.parent_occurrence_index)?),
                instantiated_assembly_id: assembly_id(path),
                transform: GeometryTransform(transform),
            });
        }
        instances.sort_by(|left, right| left.id.cmp(&right.id));

        let topology_occurrences = self.topology_occurrences();
        let mut assemblies = Vec::new();
        for occurrence in self.occurrences.values() {
            let mut children = child_instances
                .remove(&occurrence.occurrence_index)
                .unwrap_or_default();
            children.sort();
            let path = occurrence.path_segments.as_slice();
            assemblies.push(ExactAssembly {
                id: assembly_id(path),
                definition_digest: if occurrence.definition_index == 0 {
                    digest(
                        b"runmat.exact-geometry.occt-assembly-definition\0",
                        &self.representation_digest,
                    )
                } else {
                    self.definition_digests[&occurrence.definition_index]
                },
                body_ids: topology_occurrences
                    .contains(&occurrence.occurrence_index)
                    .then(|| body_id(path))
                    .into_iter()
                    .collect(),
                child_instance_ids: children,
            });
        }
        assemblies.sort_by(|left, right| left.id.cmp(&right.id));
        Ok(AssemblyProjection {
            root_id: assembly_id(self.path(1)?),
            assemblies,
            instances,
        })
    }

    pub fn project_bodies(
        &self,
        payload: &bridge::OcctExactShapePayload,
        solid_mass_properties: Option<&BodyMassProperties>,
        representation_digest: [u8; 32],
    ) -> Result<(Vec<ExactBody>, Vec<ExactMassPropertiesRecord>), GeometryImportError> {
        let mut bodies = Vec::new();
        let mut records = Vec::new();
        for occurrence in self.occurrences.values() {
            let index = occurrence.occurrence_index;
            let path = occurrence.path_segments.as_slice();
            let lump_ids = payload
                .lumps
                .iter()
                .filter(|lump| lump.occurrence_index == index)
                .map(|lump| exact_lump_id(lump, path))
                .collect::<Vec<_>>();
            let owned_shells = payload
                .solids
                .iter()
                .filter(|solid| solid.occurrence_index == index)
                .flat_map(|solid| {
                    std::iter::once(solid.outer_shell_key)
                        .chain(solid.void_shell_keys.iter().copied())
                })
                .collect::<BTreeSet<_>>();
            let sheet_shell_ids = payload
                .shells
                .iter()
                .filter(|shell| {
                    shell.occurrence_index == index && !owned_shells.contains(&shell.shape_key)
                })
                .map(|shell| shape_id(PersistentEntityKind::Shell, shell.shape_key, path))
                .collect::<Vec<_>>();
            if lump_ids.is_empty() && sheet_shell_ids.is_empty() {
                if occurrence.body_shape_key != 0 {
                    return Err(invalid(
                        "OCCT exact body occurrence contains no body topology",
                    ));
                }
                continue;
            }
            if occurrence.body_shape_key == 0 {
                return Err(invalid("OCCT exact body occurrence has no evaluator shape"));
            }
            if !lump_ids.is_empty() && !sheet_shell_ids.is_empty() {
                return Err(invalid(
                    "mixed solid and sheet topology requires distinct OCCT body definitions",
                ));
            }
            let mass_id = mass_id(path);
            let can_use_root_evidence = self.occurrences.len() == 1 && !lump_ids.is_empty();
            let implementation = if can_use_root_evidence {
                solid_mass_properties
                    .copied()
                    .map(
                        |properties| ExactMassPropertiesImplementation::KernelValidated {
                            properties,
                            validation_digest: super::exact_projection::mass_validation_digest(
                                representation_digest,
                                &properties,
                            ),
                        },
                    )
                    .unwrap_or_else(|| ExactMassPropertiesImplementation::Kernel {
                        reference: kernel_reference(
                            representation_digest,
                            occurrence.body_shape_key,
                        ),
                    })
            } else {
                ExactMassPropertiesImplementation::Kernel {
                    reference: kernel_reference(representation_digest, occurrence.body_shape_key),
                }
            };
            records.push(ExactMassPropertiesRecord {
                id: mass_id.clone(),
                implementation,
            });
            bodies.push(ExactBody {
                id: body_id(path),
                mass_properties_evaluator_id: mass_id,
                lump_ids,
                is_sheet_body: !sheet_shell_ids.is_empty(),
                sheet_shell_ids,
            });
        }
        bodies.sort_by(|left, right| left.id.cmp(&right.id));
        records.sort_by(|left, right| left.id.cmp(&right.id));
        Ok((bodies, records))
    }

    fn topology_occurrences(&self) -> BTreeSet<u64> {
        self.occurrences
            .values()
            .filter(|occurrence| occurrence.body_shape_key != 0)
            .map(|occurrence| occurrence.occurrence_index)
            .collect()
    }
}

fn kernel_reference(
    representation_digest: [u8; 32],
    body_shape_key: u64,
) -> runmat_geometry_core::KernelEvaluatorRef {
    runmat_geometry_core::KernelEvaluatorRef {
        entity_token: format!("body:{body_shape_key:020}"),
        representation_digest,
    }
}

fn digest(domain: &[u8], bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(bytes);
    hasher.finalize().into()
}

fn invalid(reason: impl Into<String>) -> GeometryImportError {
    GeometryImportError::InvalidGeometry(reason.into())
}
