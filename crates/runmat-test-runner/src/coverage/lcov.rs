use std::collections::BTreeMap;

use runmat_test::coverage::{CoverageAggregate, CoverageMetric, CoverageSite};

use crate::reporter::RenderedReport;

pub(super) fn render(coverage: &CoverageAggregate) -> RenderedReport {
    let mut files = BTreeMap::<(&str, &str), Vec<&CoverageSite>>::new();
    for site in &coverage.sites {
        files
            .entry((&site.owner_identity, &site.relative_path))
            .or_default()
            .push(site);
    }
    let mut output = String::new();
    for ((_, path), sites) in files {
        output.push_str("TN:RunMat\nSF:");
        output.push_str(path);
        output.push('\n');
        let mut lines = BTreeMap::<u32, u64>::new();
        let mut functions = Vec::new();
        for site in sites {
            if !site.instrumented {
                continue;
            }
            let count = coverage.count(site);
            match site.metric {
                CoverageMetric::Function => functions.push((site, count)),
                CoverageMetric::Statement => {
                    let total = lines.entry(site.start_line).or_default();
                    *total = total.saturating_add(count);
                }
                _ => {}
            }
        }
        for (site, _) in &functions {
            output.push_str(&format!(
                "FN:{},{}\n",
                site.start_line,
                lcov_name(&site.semantic_path)
            ));
        }
        for (site, count) in &functions {
            output.push_str(&format!(
                "FNDA:{count},{}\n",
                lcov_name(&site.semantic_path)
            ));
        }
        output.push_str(&format!("FNF:{}\n", functions.len()));
        output.push_str(&format!(
            "FNH:{}\n",
            functions.iter().filter(|(_, count)| *count != 0).count()
        ));
        for (line, count) in &lines {
            output.push_str(&format!("DA:{line},{count}\n"));
        }
        output.push_str(&format!("LF:{}\n", lines.len()));
        output.push_str(&format!(
            "LH:{}\n",
            lines.values().filter(|count| **count != 0).count()
        ));
        output.push_str("end_of_record\n");
    }
    super::report("coverage.lcov", "text/plain", output)
}

fn lcov_name(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if matches!(character, '\n' | '\r' | ',') {
                '_'
            } else {
                character
            }
        })
        .collect()
}
