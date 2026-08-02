use std::collections::{BTreeSet, HashMap};

use crate::testing::attributes;
use crate::testing::SemanticTestSource;

#[derive(Clone)]
pub(super) struct ClassRecord {
    pub source_index: usize,
    pub class_index: usize,
    pub name: String,
}

pub(super) enum Lineage {
    TestCase(Vec<usize>),
    NotTestCase,
    Invalid { code: &'static str, message: String },
}

pub(super) fn class_records(sources: &[SemanticTestSource<'_>]) -> Vec<ClassRecord> {
    sources
        .iter()
        .enumerate()
        .flat_map(|(source_index, source)| {
            source
                .assembly
                .classes
                .iter()
                .enumerate()
                .map(move |(class_index, class)| ClassRecord {
                    source_index,
                    class_index,
                    name: class
                        .name
                        .0
                        .iter()
                        .map(|segment| segment.0.as_str())
                        .collect::<Vec<_>>()
                        .join("."),
                })
        })
        .collect()
}

pub(super) fn by_name(records: &[ClassRecord]) -> HashMap<String, usize> {
    records
        .iter()
        .enumerate()
        .map(|(index, record)| (record.name.to_ascii_lowercase(), index))
        .collect()
}

pub(super) fn test_case_lineage(
    start: usize,
    records: &[ClassRecord],
    by_name: &HashMap<String, usize>,
    sources: &[SemanticTestSource<'_>],
) -> Lineage {
    let mut lineage = Vec::new();
    let mut seen = BTreeSet::new();
    let mut current = start;
    loop {
        if !seen.insert(current) {
            return Lineage::Invalid {
                code: "RunMat:TestDiscovery:InheritanceCycle",
                message: format!(
                    "test class inheritance cycle includes '{}'",
                    records[current].name
                ),
            };
        }
        lineage.push(current);
        let record = &records[current];
        let class = &sources[record.source_index].assembly.classes[record.class_index];
        let Some(parent) = class.declared_super_class.as_deref() else {
            return Lineage::NotTestCase;
        };
        if parent.eq_ignore_ascii_case("matlab.unittest.TestCase")
            || runmat_builtins::is_class_or_subclass(parent, "matlab.unittest.TestCase")
        {
            lineage.reverse();
            return Lineage::TestCase(lineage);
        }
        let Some(parent_index) = by_name.get(&parent.to_ascii_lowercase()) else {
            let declares_testing_metadata = class
                .methods
                .iter()
                .any(|method| attributes::has(&method.declared_attributes, "Test"));
            return if declares_testing_metadata {
                Lineage::Invalid {
                    code: "RunMat:TestDiscovery:UnresolvedSuperclass",
                    message: format!(
                        "cannot resolve superclass '{parent}' while discovering test class '{}'",
                        record.name
                    ),
                }
            } else {
                Lineage::NotTestCase
            };
        };
        current = *parent_index;
    }
}
