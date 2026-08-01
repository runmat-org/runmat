use crate::error::TestDomainError;
use crate::identity::RunId;

use super::{validate_plan, ProgramRevision, SuitePlan, TestPlan};

#[derive(Clone, Debug)]
pub struct TestPlanBuilder {
    revision: ProgramRevision,
    invocation_identity: String,
    suites: Vec<SuitePlan>,
}

impl TestPlanBuilder {
    pub fn new(revision: ProgramRevision, invocation_identity: impl Into<String>) -> Self {
        Self {
            revision,
            invocation_identity: invocation_identity.into(),
            suites: Vec::new(),
        }
    }

    pub fn add_suite(mut self, suite: SuitePlan) -> Self {
        self.suites.push(suite);
        self
    }

    pub fn build(mut self) -> Result<TestPlan, TestDomainError> {
        self.suites.sort_by(|a, b| a.id.cmp(&b.id));
        for suite in &mut self.suites {
            suite.fixture_groups.sort_by(|a, b| a.id.cmp(&b.id));
            for group in &mut suite.fixture_groups {
                group.fixtures.sort_by(|a, b| a.id.cmp(&b.id));
                group.tests.sort_by(|a, b| a.id.cmp(&b.id));
            }
        }
        let run_id = RunId::derive(
            &self.revision.canonical_identity(),
            &self.invocation_identity,
        );
        let plan = TestPlan::new(run_id, self.revision, self.suites);
        validate_plan(&plan)?;
        Ok(plan)
    }
}
