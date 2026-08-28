use runmat_test::coverage::{CoverageAggregate, CoverageFilter};

pub(super) fn filtered(coverage: &CoverageAggregate, filter: &CoverageFilter) -> CoverageAggregate {
    let sites = coverage
        .sites
        .iter()
        .filter(|site| filter.includes(site))
        .cloned()
        .collect::<Vec<_>>();
    let counts = sites
        .iter()
        .filter_map(|site| {
            coverage
                .counts
                .get(&site.id)
                .copied()
                .map(|count| (site.id.clone(), count))
        })
        .collect();
    CoverageAggregate {
        program_revision: coverage.program_revision.clone(),
        sites,
        counts,
    }
}
