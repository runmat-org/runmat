use super::*;

mod focus;

pub(super) use focus::append_geometry_focus_sizing_samples;

pub(super) fn structured_grid(
    input: &BoundaryMeshInput,
    options: &VolumeMeshingOptions,
    sizing: Option<&mut MeshSizingField>,
) -> Result<StructuredGrid, MeshingError> {
    let max_by_budget = ((options.max_elements / 6).max(1) as f64)
        .cbrt()
        .floor()
        .max(1.0) as usize;
    let requested = match options.target_size {
        MeshTargetSize::Auto => match options.profile {
            MeshProfile::Coarse => 1,
            MeshProfile::AnalysisReady => 2,
            MeshProfile::Adaptive | MeshProfile::Fine => 3,
        },
        MeshTargetSize::LengthM(length_m) => {
            if !length_m.is_finite() || length_m <= 0.0 {
                return Err(MeshingError::InvalidTargetSize);
            }
            let max_span = (0..3)
                .map(|axis| input.bounds_max_m[axis] - input.bounds_min_m[axis])
                .fold(0.0_f64, f64::max);
            (max_span / length_m).ceil().max(1.0) as usize
        }
    };
    let requested = sizing
        .as_deref()
        .and_then(global_sizing_target_size_m)
        .map(|length_m| requested.max(divisions_for_target_size(input, length_m)))
        .unwrap_or(requested);
    let divisions = requested.clamp(1, max_by_budget);
    let mut grid = StructuredGrid::uniform(input, divisions);
    if let Some(sizing) = sizing {
        insert_local_sizing_breakpoints(input, options.max_elements, sizing, &mut grid);
    }
    Ok(grid)
}

fn divisions_for_target_size(input: &BoundaryMeshInput, length_m: f64) -> usize {
    let max_span = (0..3)
        .map(|axis| input.bounds_max_m[axis] - input.bounds_min_m[axis])
        .fold(0.0_f64, f64::max);
    (max_span / length_m).ceil().max(1.0) as usize
}

fn global_sizing_target_size_m(sizing: &MeshSizingField) -> Option<f64> {
    [sizing.min_size_m, sizing.global_target_size_m]
        .into_iter()
        .flatten()
        .filter(|value| value.is_finite() && *value > 0.0)
        .reduce(f64::min)
}

fn insert_local_sizing_breakpoints(
    input: &BoundaryMeshInput,
    max_elements: usize,
    sizing: &mut MeshSizingField,
    grid: &mut StructuredGrid,
) {
    let mut samples = Vec::new();
    for sample in sizing.samples.clone() {
        let Some(target_size_m) = clamped_sample_target_size(sample.target_size_m, sizing) else {
            sizing.rejected_samples.push(sizing_rejection(
                sample.position_m,
                sample.target_size_m,
                sample.reason,
                "skipped_invalid",
                "sizing sample target size was not finite and positive after bounds were applied",
            ));
            continue;
        };
        if !sample.position_m.iter().all(|value| value.is_finite()) {
            sizing.rejected_samples.push(sizing_rejection(
                sample.position_m,
                target_size_m,
                sample.reason,
                "skipped_invalid",
                "sizing sample position contained a non-finite coordinate",
            ));
            continue;
        }
        samples.push((sample.position_m, target_size_m, sample.reason));
    }
    for sample in sizing.anisotropic_samples.clone() {
        let Some(target_size_m) = anisotropic_sample_target_size(&sample, sizing) else {
            sizing.rejected_samples.push(sizing_rejection(
                sample.position_m,
                sample
                    .target_sizes_m
                    .iter()
                    .copied()
                    .fold(f64::INFINITY, f64::min),
                sample.reason,
                "skipped_invalid",
                "anisotropic sizing sample did not define a valid metric",
            ));
            continue;
        };
        samples.push((sample.position_m, target_size_m, sample.reason));
    }
    samples.sort_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.0[0].total_cmp(&right.0[0]))
            .then_with(|| left.0[1].total_cmp(&right.0[1]))
            .then_with(|| left.0[2].total_cmp(&right.0[2]))
    });

    for (position_m, target_size_m, reason) in samples {
        let mut inserted_breakpoint_count = 0_usize;
        let mut duplicate_or_boundary_count = 0_usize;
        for axis in 0..3 {
            for coordinate in
                local_breakpoint_candidates(input, axis, position_m[axis], target_size_m)
            {
                let mut candidate = grid.clone();
                if !candidate.insert_axis_coordinate(axis, coordinate) {
                    duplicate_or_boundary_count += 1;
                    continue;
                }
                if candidate.element_count() > max_elements {
                    sizing.rejected_samples.push(sizing_rejection(
                        position_m,
                        target_size_m,
                        reason.clone(),
                        "skipped_budget",
                        "element budget prevented local sizing breakpoint",
                    ));
                } else if !candidate.satisfies_quality_guard() {
                    sizing.rejected_samples.push(sizing_rejection(
                        position_m,
                        target_size_m,
                        reason.clone(),
                        "skipped_quality",
                        "mesh quality guard prevented local sizing breakpoint",
                    ));
                } else {
                    *grid = candidate;
                    inserted_breakpoint_count += 1;
                }
            }
        }
        if inserted_breakpoint_count > 0 {
            sizing.applied_samples.push(sizing_application(
                position_m,
                target_size_m,
                inserted_breakpoint_count,
                reason.clone(),
                duplicate_or_boundary_count,
            ));
        } else if duplicate_or_boundary_count > 0 {
            sizing.rejected_samples.push(sizing_rejection(
                position_m,
                target_size_m,
                reason,
                "skipped_duplicate",
                "sizing sample only produced duplicate or boundary-clamped breakpoints",
            ));
        }
    }
}

fn sizing_application(
    position_m: [f64; 3],
    target_size_m: f64,
    inserted_breakpoint_count: usize,
    reason: Option<String>,
    duplicate_or_boundary_count: usize,
) -> SizingSampleApplication {
    let detail = if duplicate_or_boundary_count > 0 {
        Some(format!(
            "inserted {inserted_breakpoint_count} local sizing breakpoints; skipped {duplicate_or_boundary_count} duplicate or boundary-clamped candidates"
        ))
    } else {
        Some(format!(
            "inserted {inserted_breakpoint_count} local sizing breakpoints"
        ))
    };
    SizingSampleApplication {
        position_m,
        target_size_m,
        inserted_breakpoint_count,
        reason,
        detail,
    }
}

fn sizing_rejection(
    position_m: [f64; 3],
    target_size_m: f64,
    reason: Option<String>,
    status: &str,
    detail: &str,
) -> SizingSampleRejection {
    SizingSampleRejection {
        position_m,
        target_size_m,
        status: status.to_string(),
        reason,
        detail: Some(detail.to_string()),
    }
}

fn clamped_sample_target_size(target_size_m: f64, sizing: &MeshSizingField) -> Option<f64> {
    if !target_size_m.is_finite() || target_size_m <= 0.0 {
        return None;
    }
    let mut target_size_m = target_size_m;
    if let (Some(global_target_size_m), Some(growth_rate)) = (
        sizing
            .global_target_size_m
            .filter(|value| value.is_finite() && *value > 0.0),
        sizing
            .growth_rate
            .filter(|value| value.is_finite() && *value >= 1.0),
    ) {
        target_size_m = target_size_m.max(global_target_size_m / growth_rate);
    }
    if let Some(min_size_m) = sizing
        .min_size_m
        .filter(|value| value.is_finite() && *value > 0.0)
    {
        target_size_m = target_size_m.max(min_size_m);
    }
    if let Some(max_size_m) = sizing
        .max_size_m
        .filter(|value| value.is_finite() && *value > 0.0)
    {
        target_size_m = target_size_m.min(max_size_m);
    }
    (target_size_m.is_finite() && target_size_m > 0.0).then_some(target_size_m)
}

fn anisotropic_sample_target_size(
    sample: &AnisotropicSizingSample,
    sizing: &MeshSizingField,
) -> Option<f64> {
    if !sample.is_valid_metric() {
        return None;
    }
    let target_size_m = sample
        .target_sizes_m
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    clamped_sample_target_size(target_size_m, sizing)
}

fn local_breakpoint_candidates(
    input: &BoundaryMeshInput,
    axis: usize,
    coordinate: f64,
    target_size_m: f64,
) -> [f64; 3] {
    [
        coordinate,
        coordinate - target_size_m,
        coordinate + target_size_m,
    ]
    .map(|value| value.clamp(input.bounds_min_m[axis], input.bounds_max_m[axis]))
}

impl StructuredGrid {
    fn insert_axis_coordinate(&mut self, axis: usize, coordinate: f64) -> bool {
        let coordinates = match axis {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => &mut self.z,
        };
        if !coordinate.is_finite() {
            return false;
        }
        let span = coordinates
            .last()
            .zip(coordinates.first())
            .map(|(max, min)| max - min)
            .unwrap_or(0.0)
            .abs();
        let tolerance = span.max(1.0) * 1.0e-10;
        if coordinates
            .iter()
            .any(|existing| (*existing - coordinate).abs() <= tolerance)
        {
            return false;
        }
        coordinates.push(coordinate);
        coordinates.sort_by(f64::total_cmp);
        true
    }

    fn satisfies_quality_guard(&self) -> bool {
        let thresholds = QualityThresholds::default();
        let aspect_limit = thresholds
            .max_aspect_ratio
            .min(1.0 / thresholds.min_scaled_jacobian.max(f64::EPSILON));
        self.max_cell_aspect_ratio()
            .is_some_and(|ratio| ratio.is_finite() && ratio <= aspect_limit)
    }
}
