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
            let result = execute_text_request_for_testing(&mut session, "neverAssigned.field")
                .expect_err("ordinary reads must not create structures");
            assert!(
                matches!(result, runmat_core::RunError::Semantic(error)
                    if error.identifier.as_deref() == Some("RunMat:UndefinedVariable")),
                "ordinary undefined reads retain their semantic diagnostic"
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

#[test]
fn expanded_fixed_targets_preserve_order_roots_and_one_rhs_output() {
    gc_test_context(|| {
        for jit in [false, true] {
            for source in [
                "A=[1,2;3,4]; I={1,2}; A(I{:})=9; assert(isequal(A,[1,9;3,4]));",
                "A=[1,2;3,4]; I={2}; A(:,I{:})=7; assert(isequal(A,[1,7;3,7]));",
                "A=[1,2,3,4]; I={2}; A(I{:})=[]; assert(isequal(A,[1,3,4]));",
                "s(1).k=1; s(2).k=2; A=[1,2;3,4]; A(s.k)=8; assert(isequal(A,[1,8;3,4]));",
                "I={1}; x.branch.values=[1,2]; x.branch.values(I{:})=6; assert(isequal(x.branch.values,[6,2]));",
                "I={1}; values(I{:})=6; assert(values==6);",
                "global targetIndices; targetIndices={1}; A=[1,2;3,4]; A(targetIndices{:},changeLaterSelector())=8; assert(isequal(A,[1,8;3,4])); function value=changeLaterSelector(); global targetIndices; targetIndices={2}; value=2; end",
                "global fixedReceiver; fixedReceiver=[1,2,3]; I={1}; fixedReceiver(I{:})=replaceReceiver(); assert(isequal(fixedReceiver,[8,2,3])); function value=replaceReceiver(); global fixedReceiver; assert(nargout==1); fixedReceiver=[99,99,99]; value=8; end",
                "global fixedReceiver; fixedReceiver=[1,2,3]; I={1}; fixedReceiver(I{:})=identityValue(replaceReceiver()); assert(isequal(fixedReceiver,[8,2,3])); function value=replaceReceiver(); global fixedReceiver; fixedReceiver=[99,99,99]; value=8; end; function value=identityValue(value); assert(nargout==1); end",
                "global fixedReceiver; fixedReceiver=[1,2,3]; I={1}; fixedReceiver(I{:})=await(replaceAsync()); assert(isequal(fixedReceiver,[8,2,3])); async function value=replaceAsync(); global fixedReceiver; fixedReceiver=[99,99,99]; value=8; end",
                "global fixedReceiver; fixedReceiver=[1,2,3]; I={1}; fixedReceiver(I{:})=identityValue(await(replaceAsync())); assert(isequal(fixedReceiver,[8,2,3])); async function value=replaceAsync(); global fixedReceiver; fixedReceiver=[99,99,99]; value=8; end; function value=identityValue(value); assert(nargout==1); end",
                "A=[1,2]; I={0}; try; A(I{:})=9; catch caught; end; assert(isequal(A,[1,2])); assert(exist('caught','var')==1);",
                // Outer end depends on argument position, which a variable
                // width list cannot prove. A nested expansion's own end has
                // an independent receiver and remains legal.
                "A=[1,2]; I={1,2}; A(I{end})=7; assert(isequal(A,[1,7]));",
            ] {
                let mut session = RunMatSession::with_options(jit, false).unwrap();
                session.set_compat_mode(runmat_core::CompatMode::RunMat);
                let result = execute_text_request_for_testing(&mut session, source).unwrap();
                assert!(result.error.is_none(), "jit={jit}: {source}: {:?}", result.error);
            }
        }
    });
}
