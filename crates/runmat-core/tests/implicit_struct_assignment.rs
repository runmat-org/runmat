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
                "indexed(1).k=1; indexed(2).k=2; indexed(2).extra=9; assert(indexed(1).k==1); assert(indexed(2).k==2); assert(isempty(indexed(1).extra)); assert(indexed(2).extra==9);",
                "grid(1,1).k=4; grid(2,3).k=8; assert(isequal(size(grid),[2,3])); assert(grid(1,1).k==4); assert(grid(2,3).k==8); assert(isempty(grid(1,2).k));",
                "key='dynamic'; deep(2).branch.(key)=11; assert(deep(2).branch.dynamic==11); assert(isempty(deep(1).branch));",
                "cells{1}.k=5; cells{2}.branch.leaf=6; cells{2,2}.k=7; assert(cells{1}.k==5); assert(cells{1,2}.branch.leaf==6); assert(cells{2,2}.k==7);",
                "typed.a(2)=7; typed.c{2}=8; typed.s.k=9; assert(isequal(typed.a,[0,7])); assert(typed.c{2}==8); assert(typed.s.k==9);",
                "nestedCells{2}{3}=4; nestedCells{1}(2)=5; assert(nestedCells{2}{3}==4); assert(isequal(nestedCells{1},[0,5]));",
                "inBounds={[]}; inBounds(1){1}(2)=7; assert(isequal(inBounds{1},[0,7]));",
                "appendRoot(end+1).k=7; assert(numel(appendRoot)==1); assert(appendRoot(1).k==7);",
                "appendChild(1).a(end+1).k=8; assert(numel(appendChild(1).a)==1); assert(appendChild(1).a(1).k==8);",
                "appendCell{2}(end+1).k=9; assert(numel(appendCell{2})==1); assert(appendCell{2}(1).k==9);",
                "schema.a=1; schema(2).a.b=struct('value',3); assert(schema(1).a==1); assert(schema(2).a.b.value==3);",
                "schemaArray(1).a=[]; schemaArray(2).a(2).k=6; assert(isempty(schemaArray(1).a)); assert(numel(schemaArray(2).a)==2); assert(schemaArray(2).a(2).k==6);",
                "global selectorCalls; selectorCalls=0; ordered(nextIndex()).k=12; assert(selectorCalls==1); assert(ordered(2).k==12); function value=nextIndex(); global selectorCalls; selectorCalls=selectorCalls+1; value=2; end",
                "global originalStruct; originalStruct=struct('k',1); originalStruct(2).k=replaceStruct(); assert(originalStruct(1).k==1); assert(originalStruct(2).k==13); function value=replaceStruct(); global originalStruct; originalStruct=struct('k',99); value=13; end",
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
            for source in [
                "zeroMember=0; zeroMember.unknownZeroSentinelProperty389=1;",
                "zero=0; zero(1).field=1;",
                "existingEmpty.a=[]; existingEmpty.a.k=1;",
                "existingZero.a=0; existingZero.a.k=1;",
                "existingRow.a=[]; existingRow(1).a.k=1;",
                "array=struct('a',1); array(1)=struct('b',2);",
                "badIndex=struct('a',1); badIndex(0).field=1;",
            ] {
                let result = execute_text_request_for_testing(&mut session, source).unwrap();
                assert!(
                    result.error.is_some(),
                    "invalid creation must fail: {source}"
                );
            }
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
                let result = execute_text_request_for_testing(&mut session, source)
                    .unwrap_or_else(|error| panic!("jit={jit}: {source}: {error:?}"));
                assert!(result.error.is_none(), "jit={jit}: {source}: {:?}", result.error);
            }
        }
    });
}
