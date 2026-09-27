#[path = "support/mod.rs"]
mod test_helpers;

#[test]
fn nd_cell_expansion_uses_shared_selector_semantics_in_calls_and_aggregates() {
    let source = "C = reshape(num2cell(1:8), [2 2 2]);
        assert(isequal(C{1,1,1}, 1));
        assert(isequal([C{1,:,:}], [1 3 5 7]));
        assert(isequal([C{2,:}], [2 4 6 8]));
        assert(isequal(C{2,2,2,1}, 8));
        assert(isequal(pair(C{1,1,1}, C{2,2,2}), [1 8]));
        function result = pair(a,b)
          result = [a b];
        end";
    let code = test_helpers::compile_source(source).expect("compile ND cell expansion");
    test_helpers::interpret(&code).expect("execute ND cell expansion");
}

#[test]
fn cell2mat_nd_example_retains_direct_brace_argument_assertions() {
    let source = "C = cell(2,1,2);
        C{1,1,1} = reshape(1:4,[1 2 2]);
        C{2,1,1} = reshape(5:12,[2 2 2]);
        C{1,1,2} = reshape(13:18,[1 2 3]);
        C{2,1,2} = reshape(19:30,[2 2 3]);
        A = cell2mat(C);
        assert(isequal(size(A),[3 2 5]));
        assert(isequal(A(1,:,1:2),C{1,1,1}));
        assert(isequal(A(2:3,:,1:2),C{2,1,1}));
        assert(isequal(A(1,:,3:5),C{1,1,2}));
        assert(isequal(A(2:3,:,3:5),C{2,1,2}));";
    let code = test_helpers::compile_source(source).expect("compile cell2mat example");
    test_helpers::interpret(&code).expect("execute cell2mat example");
}
