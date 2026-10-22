//! Evidence cells: every required obligation x envelope, with the tagged tests that claim it
//! and a verdict derived only from test results. Nothing here is hand-edited or hand-linked.
use crate::{
    requirement::{Envelope, Requirement},
    witness::{Tag, TestLocator},
};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TestResult {
    Passed,
    Failed(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Every tagged test passed and at least one tag declares complete coverage.
    Pass,
    /// Every tagged test passed, but no tag claims to discharge the whole criterion.
    Partial,
    /// At least one tagged test failed, was ignored, or could not be found.
    Fail,
    /// No test is tagged for this cell.
    Open,
    /// Tagged, but the tests were not run (report built with `--no-run`).
    Tagged,
}

impl Verdict {
    pub fn label(self) -> &'static str {
        match self {
            Verdict::Pass => "PASS",
            Verdict::Partial => "PARTIAL",
            Verdict::Fail => "FAIL",
            Verdict::Open => "OPEN",
            Verdict::Tagged => "TAGGED",
        }
    }
}

#[derive(Clone, Debug)]
pub struct CellTest {
    pub locator: TestLocator,
    pub boundary: String,
    pub complete: bool,
    pub result: Option<TestResult>,
}

#[derive(Clone, Debug)]
pub struct Cell {
    pub requirement: &'static str,
    pub obligation: &'static str,
    pub envelope: Envelope,
    pub verdict: Verdict,
    pub tests: Vec<CellTest>,
}

/// `results` is keyed by `TestLocator::describe()`; `None` means tests were not run.
pub fn cells(
    registry: &[&'static Requirement],
    tags: &[Tag],
    results: Option<&BTreeMap<String, TestResult>>,
) -> Vec<Cell> {
    let mut by_cell: BTreeMap<(String, String, String), Vec<&Tag>> = BTreeMap::new();
    for tag in tags {
        for envelope in &tag.envelopes {
            by_cell
                .entry((
                    tag.requirement.clone(),
                    tag.obligation.clone(),
                    format!("{envelope:?}"),
                ))
                .or_default()
                .push(tag);
        }
    }
    let mut out = Vec::new();
    for requirement in registry {
        for obligation in requirement.obligations {
            for envelope in obligation.required_envelopes {
                let claimed = by_cell
                    .get(&(
                        requirement.id.to_string(),
                        obligation.id.to_string(),
                        format!("{envelope:?}"),
                    ))
                    .cloned()
                    .unwrap_or_default();
                let tests: Vec<CellTest> = claimed
                    .iter()
                    .map(|tag| CellTest {
                        locator: tag.test.clone(),
                        boundary: tag.boundary.clone(),
                        complete: tag.complete,
                        result: results.map(|r| {
                            r.get(&tag.test.describe()).cloned().unwrap_or_else(|| {
                                TestResult::Failed("test not found in cargo test run".into())
                            })
                        }),
                    })
                    .collect();
                let verdict = if tests.is_empty() {
                    Verdict::Open
                } else if results.is_none() {
                    Verdict::Tagged
                } else if tests.iter().all(|t| t.result == Some(TestResult::Passed)) {
                    if tests.iter().any(|t| t.complete) {
                        Verdict::Pass
                    } else {
                        Verdict::Partial
                    }
                } else {
                    Verdict::Fail
                };
                out.push(Cell {
                    requirement: requirement.id,
                    obligation: obligation.id,
                    envelope: *envelope,
                    verdict,
                    tests,
                });
            }
        }
    }
    out
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Totals {
    pub pass: usize,
    pub partial: usize,
    pub fail: usize,
    pub open: usize,
    pub tagged: usize,
}

impl Totals {
    pub fn add(&mut self, verdict: Verdict) {
        match verdict {
            Verdict::Pass => self.pass += 1,
            Verdict::Partial => self.partial += 1,
            Verdict::Fail => self.fail += 1,
            Verdict::Open => self.open += 1,
            Verdict::Tagged => self.tagged += 1,
        }
    }
    pub fn total(&self) -> usize {
        self.pass + self.partial + self.fail + self.open + self.tagged
    }
}
