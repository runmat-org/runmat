use super::{execute_source, has_num};
use runmat_value::Value;

#[test]
fn anonymous_snapshot_and_nested_sibling_keep_distinct_binding_semantics() {
    let values = execute_source(
        r#"
        result = outer();
        function y = outer()
            shared = 1;
            f = @() shared + reader();
            shared = 2;
            y = f();
            function z = reader()
                z = shared;
            end
        end
        "#,
    );
    // The anonymous operand keeps 1 from creation; the nested call reads 2
    // from the live parent activation reached through the same callback.
    assert!(has_num(&values, 3.0), "unexpected values: {values:?}");
}

#[test]
fn nested_global_and_persistent_captures_delegate_to_canonical_storage() {
    let values = execute_source(
        r#"
        result = globalOwner() + persistentOwner();
        function y = globalOwner()
            global lexicalGlobalOwnerWitness;
            lexicalGlobalOwnerWitness = 1;
            child();
            y = lexicalGlobalOwnerWitness;
            function child()
                lexicalGlobalOwnerWitness = lexicalGlobalOwnerWitness + 1;
            end
        end
        function y = persistentOwner()
            persistent lexicalPersistentOwnerWitness;
            lexicalPersistentOwnerWitness = 1;
            child();
            y = lexicalPersistentOwnerWitness;
            function child()
                lexicalPersistentOwnerWitness = lexicalPersistentOwnerWitness + 1;
            end
        end
        "#,
    );
    assert!(
        has_num(&values, 4.0),
        "canonical storage was not shared: {values:?}"
    );
}

#[test]
fn escaped_sibling_handles_share_mutations_but_distinct_activations_do_not() {
    let values = execute_source(
        r#"
        result = escapedShared();
        function y = escapedShared()
            [inc, peek] = makePair(4);
            a = feval(inc, 1);
            b = feval(peek);
            c = feval(inc, 2);
            [inc2, peek2] = makePair(10);
            d = feval(inc2, 3);
            e = feval(peek);
            f = feval(peek2);
            y = a + 10*b + 100*c + 1000*d + 10000*e + 100000*f;
            function [writer, reader] = makePair(seed)
                shared = seed;
                writer = @bump;
                reader = @read;
                function value = bump(delta)
                    shared = shared + delta;
                    value = shared;
                end
                function value = read()
                    value = shared;
                end
            end
        end
        "#,
    );
    // 5, 5, 7 remain in the first activation; 13, 13 belong to the second.
    assert_eq!(&values[..], &[Value::Num(1_383_755.0)]);
}

#[test]
fn recursive_nested_handles_keep_each_activation_isolated() {
    let values = execute_source(
        r#"
        result = recurse(2);
        function y = recurse(n)
            shared = n;
            bump_handle = @bump;
            if n > 0
                inner = recurse(n - 1);
            else
                inner = 0;
            end
            observed = feval(bump_handle, 1);
            y = 10*inner + observed;
            function value = bump(delta)
                shared = shared + delta;
                value = shared;
            end
        end
        "#,
    );
    // Innermost to outermost: 1, 12, 123. Shared cells cannot alias across recursion.
    assert_eq!(&values[..], &[Value::Num(123.0)]);
}
