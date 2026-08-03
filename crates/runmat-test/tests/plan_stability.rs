mod common;

use common::{procedure, test_id};
use runmat_execution::{Digest, DomainContribution, ProgramEnvironment, ProgramRevision};
use runmat_test::descriptor::TestDescriptor;
use runmat_test::identity::{FixtureGroupId, SuiteId};
use runmat_test::plan::{shard_for, FixtureGroupPlan, SuitePlan, TestPlanBuilder};

#[test]
fn plan_is_stable_across_input_order_and_checkout_location() {
    let revision = ProgramRevision::new(
        Digest::sha256(b"graph"),
        Digest::sha256(b"sources"),
        ProgramEnvironment::new(
            4,
            9,
            Digest::sha256(b"runtime"),
            Digest::sha256(b"catalog"),
            "matlab",
        )
        .unwrap(),
    )
    .unwrap()
    .with_domain_contribution(
        DomainContribution::new("runmat.test.config", Digest::sha256(b"config")).unwrap(),
    )
    .unwrap();
    let suite_id = SuiteId::derive(&revision.canonical_identity(), "suite/example");
    let group_id = FixtureGroupId::derive(suite_id.as_str(), "class-state");
    let tests = ["zeta", "alpha"]
        .map(|name| TestDescriptor {
            id: test_id(name),
            suite_id: suite_id.clone(),
            fixture_group_id: group_id.clone(),
            display_name: name.into(),
            procedure: procedure(name),
            parameters: Vec::new(),
            tags: Vec::new(),
            requirements: Default::default(),
        })
        .to_vec();
    let build = |mut tests: Vec<TestDescriptor>| {
        TestPlanBuilder::new(revision.clone(), "cli-default")
            .add_suite(SuitePlan {
                id: suite_id.clone(),
                display_name: "example".into(),
                fixture_groups: vec![FixtureGroupPlan {
                    id: group_id.clone(),
                    fixtures: Vec::new(),
                    tests: {
                        tests.reverse();
                        tests
                    },
                }],
            })
            .build()
            .unwrap()
    };
    let first = build(tests.clone());
    let second = build(tests);

    assert_eq!(first, second);
    assert_eq!(
        serde_json::to_vec(&first).unwrap(),
        serde_json::to_vec(&second).unwrap()
    );
    assert!(first
        .tests()
        .all(|test| !test.id.as_str().contains("/Users/")));
    assert_eq!(
        shard_for(&first.tests().next().unwrap().id, 7),
        shard_for(&first.tests().next().unwrap().id, 7)
    );
}
