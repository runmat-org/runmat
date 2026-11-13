/// Automatic studies share the available whole-core worker capacity. Both
/// admission and execution use this plan; a sub-core ceiling cannot fund a worker.
pub(crate) fn study_allocation(
    max_workers: u32,
    cpu_capacity_millicores: Option<u64>,
    study_count: usize,
) -> Option<(u32, usize)> {
    if study_count == 0 {
        return None;
    }
    let maximum = max_workers.max(1);
    let available = cpu_capacity_millicores.map_or(maximum, |capacity| {
        u32::try_from((capacity / 1000).min(u64::from(maximum)))
            .expect("worker count is bounded by a u32 maximum")
    });
    if available == 0 {
        return None;
    }
    Some((
        if study_count == 1 { available } else { 1 },
        (available as usize).min(study_count),
    ))
}

#[cfg(test)]
mod tests {
    use super::study_allocation;

    #[test]
    fn whole_core_capacity_bounds_single_studies_and_sweeps() {
        for (maximum, capacity, studies, expected) in [
            (8, None, 1, Some((8, 1))),
            (8, Some(1000), 1, Some((1, 1))),
            (8, Some(2000), 5, Some((1, 2))),
            (8, Some(3500), 1, Some((3, 1))),
            (2, Some(u64::MAX), 8, Some((1, 2))),
            (8, Some(999), 1, None),
            (8, Some(0), 1, None),
            (8, Some(1000), 0, None),
        ] {
            assert_eq!(study_allocation(maximum, capacity, studies), expected);
        }
    }
}
