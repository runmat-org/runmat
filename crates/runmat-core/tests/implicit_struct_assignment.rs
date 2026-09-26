#![cfg(not(target_arch = "wasm32"))]

use runmat_core::{execute_text_request_for_testing, RunMatSession};
use runmat_gc::gc_test_context;

#[test]
fn workspace_member_assignment_initializes_only_unassigned_roots() {
    gc_test_context(|| {
        for jit in [false, true] {
            let mut session = RunMatSession::with_options(jit, false).unwrap();
            for source in [
                "config.database = struct('host', 'db.local', 'port', 5432); assert(config.database.port == 5432);",
                "nested.first.second = 17; assert(nested.first.second == 17);",
                "key = 'branch'; dynamic.(key).leaf = 19; assert(dynamic.branch.leaf == 19);",
                "config.database.port = 5433; assert(strcmp(config.database.host, 'db.local'));",
            ] {
                let result = execute_text_request_for_testing(&mut session, source).unwrap();
                assert!(result.error.is_none(), "{source}: {:?}", result.error);
            }
            let result =
                execute_text_request_for_testing(&mut session, "neverAssigned.field").unwrap();
            assert!(
                result.error.is_some(),
                "ordinary reads must not create structures"
            );
            let result =
                execute_text_request_for_testing(&mut session, "number = 7; number.field = 1;")
                    .unwrap();
            assert!(
                result.error.is_some(),
                "existing numeric values must not become structures"
            );
        }
    });
}
