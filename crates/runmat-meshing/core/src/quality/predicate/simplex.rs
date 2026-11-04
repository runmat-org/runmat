use super::{
    types::{Point3, Tetrahedron3, Triangle3},
    vector::{add, cross, distance, norm, scale, sub},
};
use robust::Coord3D;
use runmat_simulation_usage::{
    LedgerError, OperationCount, UsageCategory, UsageContext, WorkReservation,
};

pub fn triangle_centroid(points: Triangle3) -> Point3 {
    scale(add(add(points[0], points[1]), points[2]), 1.0 / 3.0)
}

pub fn tetrahedron_centroid(points: Tetrahedron3) -> Point3 {
    std::array::from_fn(|axis| {
        points[0][axis]
            + ((points[1][axis] - points[0][axis])
                + (points[2][axis] - points[0][axis])
                + (points[3][axis] - points[0][axis]))
                * 0.25
    })
}

pub fn triangle_area(points: Triangle3) -> f64 {
    0.5 * norm(cross(sub(points[1], points[0]), sub(points[2], points[0])))
}

pub fn tetrahedron_signed_volume(points: Tetrahedron3) -> f64 {
    -robust::orient3d(
        Coord3D {
            x: points[0][0],
            y: points[0][1],
            z: points[0][2],
        },
        Coord3D {
            x: points[1][0],
            y: points[1][1],
            z: points[1][2],
        },
        Coord3D {
            x: points[2][0],
            y: points[2][1],
            z: points[2][2],
        },
        Coord3D {
            x: points[3][0],
            y: points[3][1],
            z: points[3][2],
        },
    ) / 6.0
}

pub fn tetrahedron_volume(points: Tetrahedron3) -> f64 {
    tetrahedron_signed_volume(points).abs()
}

pub fn orient_tetrahedron_node_ids(
    mut node_ids: [u32; 4],
    points: Tetrahedron3,
) -> ([u32; 4], f64) {
    let mut signed_volume = tetrahedron_signed_volume(points);
    if signed_volume < 0.0 {
        node_ids.swap(1, 2);
        signed_volume = -signed_volume;
    }
    (node_ids, signed_volume)
}

pub fn tetrahedron_edge_aspect_ratio(points: Tetrahedron3) -> f64 {
    let mut min_edge = f64::INFINITY;
    let mut max_edge = 0.0_f64;
    for left_index in 0..4 {
        for right_index in (left_index + 1)..4 {
            let length = distance(points[left_index], points[right_index]);
            min_edge = min_edge.min(length);
            max_edge = max_edge.max(length);
        }
    }
    max_edge / min_edge.max(f64::EPSILON)
}

/// Reference work per valid corner: three edge subtractions (9), three
/// normalized vectors (36), one four-point determinant (23), and sqrt(2)
/// followed by its product (2), for 70 operations. The adaptive expansion
/// backend implements the determinant's reference value; its extra-precision
/// instruction sequence is not a separate reference charge.
pub fn tetrahedron_scaled_jacobian(points: Tetrahedron3) -> Result<f64, LedgerError> {
    let usage = UsageContext::capture();
    let mut batch = usage
        .as_ref()
        .map(|usage| {
            usage
                .ledger()
                .reserve(UsageCategory::Meshing, OperationCount::from_operations(280))
        })
        .transpose()?;
    let (minimum, completed) = tetrahedron_scaled_jacobian_reference(points);
    if let Some(batch) = &mut batch {
        batch.complete(OperationCount::from_operations(completed))?;
    }
    Ok(minimum)
}

/// Evaluate within an already admitted meshing owner. The caller settles the
/// accumulated prefix before it relinquishes the owner. This permits a larger
/// score kernel to use one reservation and one completion transaction.
pub fn tetrahedron_scaled_jacobian_with_owner(
    points: Tetrahedron3,
    _owner: &mut WorkReservation<'_>,
    completed: &mut u64,
) -> Result<f64, LedgerError> {
    let (value, work) = tetrahedron_scaled_jacobian_reference(points);
    *completed = completed
        .checked_add(work)
        .ok_or(LedgerError::CountOverflow)?;
    Ok(value)
}

fn tetrahedron_scaled_jacobian_reference(points: Tetrahedron3) -> (f64, u64) {
    let corners = [
        (points[0], points[1], points[2], points[3]),
        (points[1], points[0], points[3], points[2]),
        (points[2], points[0], points[1], points[3]),
        (points[3], points[0], points[2], points[1]),
    ];
    let mut minimum = f64::INFINITY;
    let mut completed = 0;
    for (origin, first, second, third) in corners {
        let mut unit = [[0.0; 3]; 3];
        let mut valid = true;
        for (slot, point) in [first, second, third].into_iter().enumerate() {
            let (normal, work) = normalized(sub(point, origin));
            completed += 3 + work;
            let Some(normal) = normal else {
                valid = false;
                break;
            };
            unit[slot] = normal;
        }
        if !valid {
            minimum = minimum.min(0.0);
            continue;
        }
        // Normalize before evaluating the dimensionless determinant so the
        // cubic edge product cannot overflow or underflow. The adaptive
        // determinant retains the sign and magnitude of nearly coplanar unit
        // vectors far beyond a direct dot/cross evaluation.
        let determinant = -robust::orient3d(
            Coord3D {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            coord(unit[0]),
            coord(unit[1]),
            coord(unit[2]),
        );
        minimum = minimum.min((2.0_f64.sqrt() * determinant).abs());
        completed += 25;
    }
    (minimum, completed)
}

fn normalized(vector: Point3) -> (Option<Point3>, u64) {
    let scale = vector.into_iter().map(f64::abs).fold(0.0_f64, f64::max);
    if scale == 0.0 || !scale.is_finite() {
        return (None, 0);
    }
    let scaled = vector.map(|value| value / scale);
    let length = scaled
        .into_iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if !length.is_finite() || length <= 0.0 {
        return (None, 9);
    }
    (Some(scaled.map(|value| value / length)), 12)
}

fn coord(point: Point3) -> Coord3D<f64> {
    Coord3D {
        x: point[0],
        y: point[1],
        z: point[2],
    }
}
