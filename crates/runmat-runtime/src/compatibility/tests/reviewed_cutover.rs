//! Fixed review receipts, not a second production catalog or live-derived oracle.
//! New declarations still fail the exhaustive comparisons until reviewed here.
use std::collections::BTreeSet;

fn rows(source: &'static str) -> impl Iterator<Item = &'static str> {
    let mut previous = "";
    source
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .inspect(move |line| {
            assert!(
                *line > previous,
                "reviewed fixture must be sorted and duplicate-free: {line}"
            );
            previous = *line;
        })
}
pub(super) fn audited_names() -> BTreeSet<&'static str> {
    rows(include_str!("reviewed/audit-additions.txt")).collect()
}
pub(super) fn extensions() -> impl Iterator<Item = (&'static str, &'static str)> {
    rows(include_str!("reviewed/extension-additions.tsv")).map(|row| {
        let (name, id) = row
            .split_once('\t')
            .expect("reviewed extension requires a name and ID");
        assert!(!name.is_empty() && !id.is_empty() && !id.contains('\t'));
        (name, id)
    })
}
