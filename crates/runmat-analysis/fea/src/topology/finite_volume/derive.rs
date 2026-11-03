use super::{
    super::{
        element::{dot, subtract},
        AnalysisProjection,
    },
    geometry::{self, FACE_CORNERS, RELATIVE_TOLERANCE},
    FiniteVolumeBoundarySource, FiniteVolumeCell, FiniteVolumeCellSelection, FiniteVolumeFace,
    FiniteVolumeIncidence, FiniteVolumeProjectionError as Error,
    FiniteVolumeProjectionPolicy as Policy, FiniteVolumeReservation, FiniteVolumeTopology,
};
use runmat_execution::Digest;
use runmat_meshing_core::{BoundaryFaceRole, StableDigest};
use sha2::{Digest as _, Sha256};
use std::collections::BTreeMap;

pub(super) fn reserve(
    source: &AnalysisProjection,
    policy: Policy,
    selection: &FiniteVolumeCellSelection,
) -> Result<FiniteVolumeReservation, Error> {
    let selected = || {
        source
            .elements
            .iter()
            .filter(|e| selection.includes(&e.region_id))
    };
    reserve_regions(
        selected().count(),
        source.boundary_faces.len(),
        selected().map(|e| &e.region_id),
        policy,
    )
}

pub(super) fn reserve_mesh(
    source: &runmat_meshing_core::SolverMeshArtifact,
    policy: Policy,
) -> Result<FiniteVolumeReservation, Error> {
    reserve_regions(
        source.topology.volume_elements.len(),
        source.topology.boundary_faces.len(),
        source.topology.volume_elements.iter().map(|e| &e.region_id),
        policy,
    )
}

fn reserve_regions<'a>(
    cells: usize,
    boundaries: usize,
    regions: impl Iterator<Item = &'a runmat_meshing_core::PersistentEntityId>,
    policy: Policy,
) -> Result<FiniteVolumeReservation, Error> {
    let (incidences, mut bytes, base_operations) = checked_counts(cells, boundaries)?;
    if cells == 0 {
        return Err(Error::Invalid("empty cell inventory"));
    }
    for (count, maximum, name) in [
        (cells, policy.max_cells, "cells"),
        (incidences, policy.max_face_incidences, "face incidences"),
        (boundaries, policy.max_boundary_faces, "boundary faces"),
    ] {
        if maximum.is_some_and(|maximum| count > maximum) {
            return Err(Error::Limit(name));
        }
    }
    // Two ordered maps, face output, cells/incidences, sorting and closure sums.
    // 1024 B/incidence exceeds key/value + B-tree node overhead; 512 B/cell
    // covers fixed records and sort/index/sum arrays. Region allocations are additional.
    let mut region_visits = 0usize;
    let mut region_bytes = 0usize;
    for region in regions {
        region_bytes = region_bytes
            .checked_add(region.source_topology_id.len())
            .ok_or(Error::Overflow)?;
        bytes = bytes
            .checked_add(region.source_topology_id.len())
            .and_then(|v| {
                region
                    .assembly_path
                    .len()
                    .checked_mul(std::mem::size_of::<String>())
                    .and_then(|w| v.checked_add(w))
            })
            .ok_or(Error::Overflow)?;
        for component in &region.assembly_path {
            region_visits = region_visits.checked_add(1).ok_or(Error::Overflow)?;
            if policy
                .max_traversal_operations
                .is_some_and(|m| region_visits > m)
            {
                return Err(Error::Limit("traversal operations"));
            }
            bytes = bytes.checked_add(component.len()).ok_or(Error::Overflow)?;
            region_bytes = region_bytes
                .checked_add(component.len())
                .ok_or(Error::Overflow)?;
        }
        if policy.max_scratch_bytes.is_some_and(|m| bytes > m) {
            return Err(Error::Limit("scratch bytes"));
        }
    }
    let operations = base_operations
        .checked_add(region_visits)
        .and_then(|v| v.checked_add(region_bytes))
        .ok_or(Error::Overflow)?;
    if policy.max_scratch_bytes.is_some_and(|m| bytes > m) {
        return Err(Error::Limit("scratch bytes"));
    }
    if policy
        .max_traversal_operations
        .is_some_and(|m| operations > m)
    {
        return Err(Error::Limit("traversal operations"));
    }
    Ok(FiniteVolumeReservation {
        cell_count: cells,
        face_incidence_count: incidences,
        boundary_face_count: boundaries,
        scratch_bytes: bytes,
        traversal_operations: operations,
    })
}

pub(super) fn checked_counts(
    cells: usize,
    boundaries: usize,
) -> Result<(usize, usize, usize), Error> {
    let incidences = cells.checked_mul(4).ok_or(Error::Overflow)?;
    let bytes = incidences
        .checked_mul(1024)
        .and_then(|v| cells.checked_mul(512).and_then(|w| v.checked_add(w)))
        .and_then(|v| boundaries.checked_mul(512).and_then(|w| v.checked_add(w)))
        .ok_or(Error::Overflow)?;
    let total = incidences
        .checked_add(boundaries)
        .and_then(|v| v.checked_add(cells))
        .ok_or(Error::Overflow)?;
    let levels = usize::BITS as usize - total.leading_zeros() as usize + 1;
    let operations = total
        .checked_mul(levels)
        .and_then(|v| v.checked_mul(128))
        .ok_or(Error::Overflow)?;
    Ok((incidences, bytes, operations))
}

pub(super) fn derive(
    source: &AnalysisProjection,
    policy: Policy,
    selection: FiniteVolumeCellSelection,
) -> Result<FiniteVolumeTopology, Error> {
    let reservation = reserve(source, policy, &selection)?;
    let mut order = (0..source.elements.len())
        .filter(|&i| selection.includes(&source.elements[i].region_id))
        .collect::<Vec<_>>();
    if order.is_empty() {
        return Err(Error::Invalid("region selection contains no cells"));
    }
    order.sort_by_key(|&i| source.elements[i].stable_identity);
    if order
        .windows(2)
        .any(|w| source.elements[w[0]].stable_identity == source.elements[w[1]].stable_identity)
        || order
            .iter()
            .any(|&i| source.elements[i].stable_identity == StableDigest::ZERO)
    {
        return Err(Error::Invalid("duplicate or absent element identity"));
    }
    let mut cells = Vec::with_capacity(order.len());
    let mut faces = BTreeMap::<[StableDigest; 3], FiniteVolumeFace>::new();
    for (cell_index, &source_index) in order.iter().enumerate() {
        let element = &source.elements[source_index];
        let geometry::CellGeometry {
            points,
            identities,
            volume,
            centroid,
        } = geometry::corners(source, source_index)?;
        cells.push(FiniteVolumeCell {
            element_identity: element.stable_identity,
            element_id: element.element_id,
            source_element_index: source_index,
            geometry_order: element.order,
            region: element.region_id.clone(),
            volume_m3: volume,
            centroid_m: centroid,
            faces: [FiniteVolumeIncidence {
                face_index: usize::MAX,
                sign: 0,
            }; 4],
        });
        for local in FACE_CORNERS {
            let mut key = local.map(|i| identities[i]);
            key.sort();
            let (face_centroid, area, magnitude) =
                geometry::face_geometry(local.map(|i| points[i]), centroid)?;
            if let Some(face) = faces.get_mut(&key) {
                if face.neighbor.is_some() {
                    return Err(Error::Invalid("nonmanifold face"));
                }
                if geometry::length(std::array::from_fn(|i| {
                    face.owner_area_vector_m2[i] + area[i]
                })) > RELATIVE_TOLERANCE * magnitude
                {
                    return Err(Error::Invalid("shared face has nonopposite incidence"));
                }
                if geometry::length(subtract(face.centroid_m, face_centroid))
                    > RELATIVE_TOLERANCE * magnitude.sqrt()
                {
                    return Err(Error::Invalid("shared face coordinates disagree"));
                }
                face.neighbor = Some(cell_index);
            } else {
                faces.insert(
                    key,
                    FiniteVolumeFace {
                        corner_identities: key,
                        centroid_m: face_centroid,
                        owner_area_vector_m2: area,
                        area_m2: runmat_types::physical::Area::new(magnitude),
                        owner: cell_index,
                        neighbor: None,
                        boundary_source: None,
                    },
                );
            }
        }
    }
    let selected = |element: usize| selection.includes(&source.elements[element].region_id);
    for (boundary_index, boundary) in source.boundary_faces.iter().enumerate() {
        geometry::validate_boundary_nodes(source, &boundary.node_indices)?;
        let key = geometry::key(source, &boundary.node_indices)?;
        let included = boundary
            .adjacent_element_indices
            .iter()
            .filter(|&&e| selected(e))
            .count();
        if included == 0 {
            // Wholly outside the selected cell complex: neither its face nor its law.
            continue;
        }
        let face = faces
            .get_mut(&key)
            .ok_or(Error::Invalid("boundary without volume face"))?;
        if face.boundary_source.is_some() {
            return Err(Error::Invalid("duplicate boundary face"));
        }
        let exterior = face.neighbor.is_none();
        let role_is_exterior = matches!(
            boundary.role,
            BoundaryFaceRole::Exterior
                | BoundaryFaceRole::ContactPrimary
                | BoundaryFaceRole::ContactSecondary
        );
        // An interface face whose other side is excluded bounds the selected complex.
        let selected_interface =
            boundary.role == BoundaryFaceRole::ConformalInterface && included == 1;
        if (role_is_exterior || selected_interface) != exterior {
            return Err(Error::Invalid("boundary role/incidence mismatch"));
        }
        let expected = if role_is_exterior { 1 } else { 2 };
        if boundary.adjacent_element_indices.len() != expected
            || !boundary
                .adjacent_element_indices
                .contains(&cells[face.owner].source_element_index)
            || face.neighbor.is_some_and(|n| {
                !boundary
                    .adjacent_element_indices
                    .contains(&cells[n].source_element_index)
            })
        {
            return Err(Error::Invalid("boundary original adjacency mismatch"));
        }
        // Face geometry has one source: the finite-volume face computed from its cell's
        // nodes, which closes every cell exactly. The mesh boundary record supplies identity,
        // role and adjacency only; its own geometry may carry curved-surface data, so it is
        // checked for orientation, never for equality with the flat face.
        if role_is_exterior
            && !(dot(boundary.oriented_area_vector_m2, face.owner_area_vector_m2) > 0.0)
        {
            return Err(Error::Invalid("exterior boundary face is not outward"));
        }
        face.boundary_source = Some(FiniteVolumeBoundarySource {
            projection_boundary_index: boundary_index,
            face_id: boundary.face_id,
            face_identity: boundary.stable_identity,
        });
    }
    let faces = faces.into_values().collect::<Vec<_>>();
    let mut counts = vec![0usize; cells.len()];
    let mut area_sums = vec![[0.0; 3]; cells.len()];
    let mut area_scales = vec![0.0; cells.len()];
    let mut volumes = vec![0.0; cells.len()];
    // Forward rounding bound of each divergence-theorem sum: the magnitude of its terms plus the
    // representation error of the absolute face and cell centroids it subtracts. A flat cell's
    // terms nearly cancel (|c_f − c|·|A_f| ≫ V), so the sum's error is relative to that
    // magnitude, not to V; a missing, duplicated or reversed face still moves the sum by O(V).
    let mut volume_scales = vec![0.0; cells.len()];
    for (face_index, face) in faces.iter().enumerate() {
        let delta = subtract(
            face.neighbor
                .map(|n| cells[n].centroid_m)
                .unwrap_or(face.centroid_m),
            cells[face.owner].centroid_m,
        );
        let conductance_geometry = dot(face.owner_area_vector_m2, delta);
        if !conductance_geometry.is_finite() || conductance_geometry <= 0.0 {
            return Err(Error::Invalid(
                "nonpositive owner-neighbor geometric conductance",
            ));
        }
        if face.neighbor.is_none() && face.boundary_source.is_none() {
            return Err(Error::Invalid("missing original exterior boundary"));
        }
        for (cell, sign) in std::iter::once((face.owner, 1i8)).chain(face.neighbor.map(|n| (n, -1)))
        {
            let slot = cells[cell]
                .faces
                .get_mut(counts[cell])
                .ok_or(Error::Invalid("cell face count"))?;
            *slot = FiniteVolumeIncidence { face_index, sign };
            counts[cell] += 1;
            let area = face.owner_area_vector_m2.map(|v| v * f64::from(sign));
            for (sum, value) in area_sums[cell].iter_mut().zip(area) {
                *sum += value;
            }
            area_scales[cell] += face.area_m2.canonical_value();
            // Translation invariant divergence theorem relative to this cell centroid.
            volumes[cell] += dot(subtract(face.centroid_m, cells[cell].centroid_m), area) / 3.0;
            volume_scales[cell] += geometry::divergence_term_scale(
                face.centroid_m,
                cells[cell].centroid_m,
                face.area_m2.canonical_value(),
            );
        }
    }
    for (i, cell) in cells.iter().enumerate() {
        if counts[i] != 4
            || geometry::length(area_sums[i]) > RELATIVE_TOLERANCE * area_scales[i]
            || !geometry::close_scalar(volumes[i], cell.volume_m3, volume_scales[i])
        {
            return Err(Error::Invalid("cell closure or conservative volume"));
        }
    }
    let geometry_digest = identity(source, &selection, &cells, &faces);
    Ok(FiniteVolumeTopology {
        mesh_digest: source.mesh_digest,
        projection_digest: source.projection_digest,
        geometry_digest,
        policy,
        selection,
        reservation,
        cells,
        faces,
    })
}

fn identity(
    source: &AnalysisProjection,
    selection: &FiniteVolumeCellSelection,
    cells: &[FiniteVolumeCell],
    faces: &[FiniteVolumeFace],
) -> Digest {
    let mut hash = Sha256::new();
    hash.update(b"runmat-finite-volume-source-bound-geometry/v1\0");
    hash.update(source.mesh_digest.bytes());
    hash.update(source.projection_digest.bytes());
    // The whole-mesh complex keeps its established identity; a subset binds its regions.
    if let FiniteVolumeCellSelection::Regions(regions) = selection {
        hash.update(b"region-selection\0");
        hash.update((regions.len() as u64).to_le_bytes());
        for region in regions {
            hash.update(format!("{:?}", region.kind).as_bytes());
            for part in std::iter::once(&region.source_topology_id).chain(&region.assembly_path) {
                hash.update((part.len() as u64).to_le_bytes());
                hash.update(part.as_bytes());
            }
        }
    }
    hash.update((cells.len() as u64).to_le_bytes());
    hash.update((faces.len() as u64).to_le_bytes());
    for cell in cells {
        hash.update(cell.element_identity.bytes());
        for value in [cell.volume_m3].into_iter().chain(cell.centroid_m) {
            hash.update(value.to_bits().to_le_bytes());
        }
    }
    for face in faces {
        for corner in face.corner_identities {
            hash.update(corner.bytes());
        }
        hash.update(cells[face.owner].element_identity.bytes());
        hash.update(
            face.neighbor
                .map(|n| cells[n].element_identity)
                .unwrap_or(StableDigest::ZERO)
                .bytes(),
        );
        hash.update(
            face.boundary_source
                .map(|b| b.face_identity)
                .unwrap_or(StableDigest::ZERO)
                .bytes(),
        );
        for value in face
            .centroid_m
            .into_iter()
            .chain(face.owner_area_vector_m2)
            .chain([face.area_m2.canonical_value()])
        {
            hash.update(value.to_bits().to_le_bytes());
        }
    }
    Digest::from_bytes(hash.finalize().into())
}
