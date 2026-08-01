use std::collections::BTreeSet;

use crate::error::TestDomainError;
use crate::version::TEST_PLAN_SCHEMA_VERSION;

use super::TestPlan;

pub fn validate_plan(plan: &TestPlan) -> Result<(), TestDomainError> {
    if plan.schema_version != TEST_PLAN_SCHEMA_VERSION {
        return Err(TestDomainError::InvalidField {
            field: "schema_version",
            reason: format!(
                "expected {}, received {}",
                TEST_PLAN_SCHEMA_VERSION, plan.schema_version
            ),
        });
    }
    let mut suites = BTreeSet::new();
    let mut groups = BTreeSet::new();
    let mut fixtures = BTreeSet::new();
    let mut tests = BTreeSet::new();
    for suite in &plan.suites {
        insert(&mut suites, suite.id.as_str(), "suite")?;
        for group in &suite.fixture_groups {
            insert(&mut groups, group.id.as_str(), "fixture group")?;
            for fixture in &group.fixtures {
                insert(&mut fixtures, fixture.id.as_str(), "fixture")?;
                if fixture.group_id != group.id {
                    return Err(TestDomainError::InvalidField {
                        field: "fixture.group_id",
                        reason: format!(
                            "fixture {} belongs to {}, not {}",
                            fixture.id.as_str(),
                            fixture.group_id.as_str(),
                            group.id.as_str()
                        ),
                    });
                }
            }
            for test in &group.tests {
                insert(&mut tests, test.id.as_str(), "test")?;
                if test.fixture_group_id != group.id {
                    return Err(TestDomainError::MissingFixtureGroup {
                        test_id: test.id.as_str().to_owned(),
                        fixture_group_id: test.fixture_group_id.as_str().to_owned(),
                    });
                }
                if test.suite_id != suite.id {
                    return Err(TestDomainError::InvalidField {
                        field: "test.suite_id",
                        reason: format!(
                            "test {} belongs to {}, not {}",
                            test.id.as_str(),
                            test.suite_id.as_str(),
                            suite.id.as_str()
                        ),
                    });
                }
                if !test.procedure.source.span.is_valid() {
                    return Err(TestDomainError::InvalidField {
                        field: "test.procedure.source.span",
                        reason: format!("test {} has an inverted source span", test.id.as_str()),
                    });
                }
            }
        }
    }
    Ok(())
}

fn insert<'a>(
    seen: &mut BTreeSet<&'a str>,
    identity: &'a str,
    kind: &'static str,
) -> Result<(), TestDomainError> {
    if !seen.insert(identity) {
        return Err(TestDomainError::DuplicateIdentity {
            kind,
            identity: identity.to_owned(),
        });
    }
    Ok(())
}
