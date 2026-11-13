use runmat_meshing_core::quality::predicate::{
    tetrahedron_scaled_jacobian, tetrahedron_scaled_jacobian_with_owner,
};
use runmat_simulation_usage::{
    LedgerError, OperationCount, UsageCategory, UsageContext, WorkReservation,
};

use super::LocalCapApexCandidate;

pub(super) fn optimized_normal_cap_apex_candidate(
    face_points: [[f64; 3]; 3],
    surface_point: [f64; 3],
    direction: [f64; 3],
    max_edge_length: f64,
    source: &'static str,
) -> Result<LocalCapApexCandidate, LedgerError> {
    // The fixed 28-step search evaluates two candidates per step. Each has
    // distance (1), apex coordinates (6), and a four-corner predicate (280).
    // The brackets cost six per step; final point construction costs nine.
    // Maximum = 28 * (6 + 2 * 287) + 9 = 16,249.
    let usage = UsageContext::capture();
    let mut batch = usage
        .as_ref()
        .map(|usage| {
            usage.ledger().reserve(
                UsageCategory::Meshing,
                OperationCount::from_operations(16_249),
            )
        })
        .transpose()?;
    let mut completed = 0;
    let result = (|| {
        let quality_at =
            |scale: f64, batch: &mut Option<WorkReservation<'_>>, completed: &mut u64| {
                let distance = max_edge_length * scale;
                let apex = [
                    surface_point[0] + direction[0] * distance,
                    surface_point[1] + direction[1] * distance,
                    surface_point[2] + direction[2] * distance,
                ];
                quality_with_owner(
                    [face_points[0], face_points[1], face_points[2], apex],
                    7,
                    batch.as_mut(),
                    completed,
                )
            };
        let mut low = 0.02_f64;
        let mut high = 1.75_f64;
        for _ in 0..28 {
            let left = low + (high - low) / 3.0;
            let right = high - (high - low) / 3.0;
            completed += 6;
            if quality_at(left, &mut batch, &mut completed)?
                < quality_at(right, &mut batch, &mut completed)?
            {
                low = left;
            } else {
                high = right;
            }
        }
        let scale = (low + high) * 0.5;
        let distance = max_edge_length * scale;
        let candidate = LocalCapApexCandidate {
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new([
                surface_point[0] + direction[0] * distance,
                surface_point[1] + direction[1] * distance,
                surface_point[2] + direction[2] * distance,
            ]),
            source,
        };
        completed += 9;
        Ok(candidate)
    })();
    if let Some(batch) = &mut batch {
        batch.complete(OperationCount::from_operations(completed))?;
    }
    result
}

pub(super) fn optimized_inplane_inward_cap_apex_candidate(
    face_points: [[f64; 3]; 3],
    surface_point: [f64; 3],
    inward_direction: [f64; 3],
    in_plane_direction: [f64; 3],
    max_edge_length: f64,
) -> Result<LocalCapApexCandidate, LedgerError> {
    // One seed plus 7 * 6 lattice probes. Each apex uses 18 real operations
    // and the predicate up to 280; final apex construction uses 18.
    // Maximum = 43 * 298 + 18 = 12,832.
    let usage = UsageContext::capture();
    let mut batch = usage
        .as_ref()
        .map(|usage| {
            usage.ledger().reserve(
                UsageCategory::Meshing,
                OperationCount::from_operations(12_832),
            )
        })
        .transpose()?;
    let mut completed = 0;
    let result = (|| {
        let quality_at = |inward_scale: f64,
                          lateral_scale: f64,
                          batch: &mut Option<WorkReservation<'_>>,
                          completed: &mut u64| {
            let apex = [
                surface_point[0]
                    + inward_direction[0] * max_edge_length * inward_scale
                    + in_plane_direction[0] * max_edge_length * lateral_scale,
                surface_point[1]
                    + inward_direction[1] * max_edge_length * inward_scale
                    + in_plane_direction[1] * max_edge_length * lateral_scale,
                surface_point[2]
                    + inward_direction[2] * max_edge_length * inward_scale
                    + in_plane_direction[2] * max_edge_length * lateral_scale,
            ];
            quality_with_owner(
                [face_points[0], face_points[1], face_points[2], apex],
                18,
                batch.as_mut(),
                completed,
            )
        };
        let mut best = (
            0.16_f64,
            0.24_f64,
            quality_at(0.16, 0.24, &mut batch, &mut completed)?,
        );
        for inward_scale in [0.04, 0.08, 0.14, 0.22, 0.34, 0.5, 0.72] {
            for lateral_scale in [0.04, 0.1, 0.18, 0.28, 0.42, 0.6] {
                let quality = quality_at(inward_scale, lateral_scale, &mut batch, &mut completed)?;
                if quality > best.2 {
                    best = (inward_scale, lateral_scale, quality);
                }
            }
        }
        let (inward_scale, lateral_scale, _) = best;
        let candidate = LocalCapApexCandidate {
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new([
                surface_point[0]
                    + inward_direction[0] * max_edge_length * inward_scale
                    + in_plane_direction[0] * max_edge_length * lateral_scale,
                surface_point[1]
                    + inward_direction[1] * max_edge_length * inward_scale
                    + in_plane_direction[1] * max_edge_length * lateral_scale,
                surface_point[2]
                    + inward_direction[2] * max_edge_length * inward_scale
                    + in_plane_direction[2] * max_edge_length * lateral_scale,
            ]),
            source: "inplane_inward_optimized",
        };
        completed += 18;
        Ok(candidate)
    })();
    if let Some(batch) = &mut batch {
        batch.complete(OperationCount::from_operations(completed))?;
    }
    result
}

fn quality_with_owner(
    points: [[f64; 3]; 4],
    geometry_work: u64,
    owner: Option<&mut WorkReservation<'_>>,
    completed: &mut u64,
) -> Result<f64, LedgerError> {
    *completed += geometry_work;
    match owner {
        Some(owner) => tetrahedron_scaled_jacobian_with_owner(points, owner, completed),
        None => tetrahedron_scaled_jacobian(points),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_simulation_usage::UsageLedger;
    use std::sync::Arc;

    #[test]
    fn fixed_normal_search_settles_one_bounded_owner() {
        let budget = OperationCount::from_operations(16_249);
        let ledger = Arc::new(UsageLedger::new(budget, budget).unwrap());
        let context = UsageContext::new(Arc::clone(&ledger));
        context.run_sync(|| {
            optimized_normal_cap_apex_candidate(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [0.25, 0.25, 0.0],
                [0.0, 0.0, 1.0],
                1.0,
                "normal_optimized_positive",
            )
            .unwrap();
        });
        let snapshot = ledger.snapshot().unwrap();
        assert_eq!(snapshot.completed.meshing, budget);
        assert_eq!(snapshot.reserved, OperationCount::ZERO);
    }
}
