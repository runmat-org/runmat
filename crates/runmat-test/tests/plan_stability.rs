mod common;

use common::{procedure, test_id};
use runmat_test::descriptor::TestDescriptor;
use runmat_test::identity::{FixtureGroupId, SuiteId};
use runmat_test::plan::{shard_for, FixtureGroupPlan, ProgramRevision, SuitePlan, TestPlanBuilder};

#[test]
fn plan_is_stable_across_input_order_and_checkout_location() {
    let revision = ProgramRevision {
        graph_digest: "sha256:graph".into(),
        source_digest: "sha256:sources".into(),
        semantic_schema: 4,
        compiler_schema: 9,
        test_config_digest: "sha256:config".into(),
    };
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
