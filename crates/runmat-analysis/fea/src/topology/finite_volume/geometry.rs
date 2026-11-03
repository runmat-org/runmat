use runmat_meshing_core::{ElementOrder, StableDigest, TETRAHEDRON_MIDSIDE_EDGE_CORNERS};

use super::{
    super::{
        element::{cross, dot, subtract, tetrahedron_geometry},
        AnalysisProjection,
    },
    FiniteVolumeProjectionError as Error,
};

pub(super) const FACE_CORNERS: [[usize; 3]; 4] = [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]];
pub(super) const RELATIVE_TOLERANCE: f64 = 4096.0 * f64::EPSILON;

pub(super) struct CellGeometry {
    pub points: [[f64; 3]; 4],
    pub identities: [StableDigest; 4],
    pub volume: f64,
    pub centroid: [f64; 3],
}

pub(super) fn close_scalar(left: f64, right: f64, scale: f64) -> bool {
    left.is_finite() && right.is_finite() && (left - right).abs() <= RELATIVE_TOLERANCE * scale
}

/// Forward rounding bound of one divergence-theorem volume term `(c_f − c)·A_f / 3`: the term's
/// magnitude plus the representation error of the absolute centroids it subtracts.
pub(super) fn divergence_term_scale(
    face_centroid: [f64; 3],
    cell_centroid: [f64; 3],
    area_m2: f64,
) -> f64 {
    let coordinate = face_centroid
        .iter()
        .chain(&cell_centroid)
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    area_m2 * (length(super::super::element::subtract(face_centroid, cell_centroid)) + coordinate)
        / 3.0
}

pub(super) fn length(vector: [f64; 3]) -> f64 {
    vector[0].hypot(vector[1]).hypot(vector[2])
}

pub(super) fn corners(
    source: &AnalysisProjection,
    element_index: usize,
) -> Result<CellGeometry, Error> {
    let element = &source.elements[element_index];
    let expected = if element.order == ElementOrder::Tet4 {
        4
    } else {
        10
    };
    if element.node_indices.len() != expected {
        return Err(Error::Invalid("element arity"));
    }
    let mut coordinates = [[0.0; 3]; 4];
    let mut identities = [StableDigest::ZERO; 4];
    for local in 0..4 {
        let node = source
            .nodes
            .get(element.node_indices[local])
            .ok_or(Error::Invalid("node index"))?;
        if node
            .coordinates_m
            .canonical_value()
            .iter()
            .any(|v| !v.is_finite())
            || node.stable_identity == StableDigest::ZERO
        {
            return Err(Error::Invalid(
                "nonfinite coordinate or absent node identity",
            ));
        }
        coordinates[local] = node.coordinates_m.canonical_value();
        identities[local] = node.stable_identity;
        if identities[..local].contains(&node.stable_identity) {
            return Err(Error::Invalid("repeated corner identity"));
        }
    }
    let (volume, centroid, _) =
        tetrahedron_geometry(coordinates).ok_or(Error::Invalid("nonpositive volume"))?;
    let scale = TETRAHEDRON_MIDSIDE_EDGE_CORNERS
        .iter()
        .map(|[a, b]| length(subtract(coordinates[*a], coordinates[*b])))
        .fold(0.0_f64, f64::max);
    if !scale.is_finite() || scale <= 0.0 {
        return Err(Error::Invalid("cell length"));
    }
    if element.order == ElementOrder::Tet10 {
        for (local, [a, b]) in TETRAHEDRON_MIDSIDE_EDGE_CORNERS.into_iter().enumerate() {
            let node = source
                .nodes
                .get(element.node_indices[4 + local])
                .ok_or(Error::Invalid("midside index"))?;
            let midpoint = std::array::from_fn(|axis| {
                coordinates[a][axis] + (coordinates[b][axis] - coordinates[a][axis]) * 0.5
            });
            if length(subtract(node.coordinates_m.canonical_value(), midpoint))
                > RELATIVE_TOLERANCE * scale
                || node
                    .coordinates_m
                    .canonical_value()
                    .iter()
                    .any(|v| !v.is_finite())
            {
                return Err(Error::CurvedTet10(element.element_id));
            }
        }
    }
    if !close_scalar(element.signed_volume_m3, volume, volume) {
        return Err(Error::Invalid("source volume mismatch"));
    }
    if element.centroid_m.iter().any(|v| !v.is_finite())
        || length(subtract(element.centroid_m, centroid)) > RELATIVE_TOLERANCE * scale
    {
        return Err(Error::Invalid("source centroid mismatch"));
    }
    Ok(CellGeometry {
        points: coordinates,
        identities,
        volume,
        centroid,
    })
}

pub(super) fn face_geometry(
    points: [[f64; 3]; 3],
    cell_centroid: [f64; 3],
) -> Result<([f64; 3], [f64; 3], f64), Error> {
    let centroid = std::array::from_fn(|axis| {
        points[0][axis]
            + ((points[1][axis] - points[0][axis]) + (points[2][axis] - points[0][axis])) / 3.0
    });
    let mut area = cross(
        subtract(points[1], points[0]),
        subtract(points[2], points[0]),
    )
    .map(|v| v * 0.5);
    if dot(area, subtract(centroid, cell_centroid)) < 0.0 {
        area = area.map(|v| -v);
    }
    let magnitude = length(area);
    if magnitude <= 0.0
        || !magnitude.is_finite()
        || dot(area, subtract(centroid, cell_centroid)) <= 0.0
    {
        return Err(Error::Invalid("degenerate face"));
    }
    Ok((centroid, area, magnitude))
}

pub(super) fn key(
    source: &AnalysisProjection,
    indices: &[usize],
) -> Result<[StableDigest; 3], Error> {
    let corners = indices.get(..3).ok_or(Error::Invalid("boundary arity"))?;
    let mut key = [StableDigest::ZERO; 3];
    for (out, &index) in key.iter_mut().zip(corners) {
        *out = source
            .nodes
            .get(index)
            .ok_or(Error::Invalid("boundary node index"))?
            .stable_identity;
    }
    key.sort();
    if key[0] == key[1] || key[1] == key[2] {
        return Err(Error::Invalid("repeated boundary corner"));
    }
    Ok(key)
}

pub(super) fn validate_boundary_nodes(
    source: &AnalysisProjection,
    indices: &[usize],
) -> Result<(), Error> {
    if indices.len() != 3 && indices.len() != 6 {
        return Err(Error::Invalid("boundary arity"));
    }
    let coordinates = indices
        .iter()
        .map(|&i| {
            source
                .nodes
                .get(i)
                .map(|n| n.coordinates_m.canonical_value())
                .ok_or(Error::Invalid("boundary index"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if coordinates.iter().flatten().any(|v| !v.is_finite()) {
        return Err(Error::Invalid("boundary coordinate"));
    }
    let scale = length(subtract(coordinates[0], coordinates[1]))
        .max(length(subtract(coordinates[1], coordinates[2])))
        .max(length(subtract(coordinates[2], coordinates[0])));
    if indices.len() == 6 {
        for (local, [a, b]) in [[0, 1], [1, 2], [2, 0]].into_iter().enumerate() {
            let midpoint = std::array::from_fn(|axis| {
                coordinates[a][axis] + (coordinates[b][axis] - coordinates[a][axis]) * 0.5
            });
            if length(subtract(coordinates[3 + local], midpoint)) > RELATIVE_TOLERANCE * scale {
                return Err(Error::Invalid("curved boundary"));
            }
        }
    }
    Ok(())
}
