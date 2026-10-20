use super::*;
use crate::execution::RootedValueLease;
use runmat_value::{CellArray, HandleRef};

fn target(handle: runmat_gc::GcHandle) -> Value {
    Value::HandleObject(HandleRef {
        class_name: "TableBinaryOwnershipFixture".into(),
        target: handle,
        valid: true,
    })
}

fn nested_struct(handle: runmat_gc::GcHandle) -> Value {
    let mut fields = runmat_value::UnboundedStructValueBuilder::new();
    fields.insert("Target", target(handle));
    Value::Struct(fields.finish())
}

fn nested_cell(handle: runmat_gc::GcHandle) -> Value {
    Value::Cell(CellArray::new(vec![target(handle)], 1, 1).expect("nested cell"))
}

fn managed_right(first: runmat_gc::GcHandle, second: runmat_gc::GcHandle) -> Value {
    // Both the variable order and named row order differ from the left table.
    // Selection allocates new outer Cells while their nested managed members
    // remain reachable only through the returned plan after the inputs retire.
    let b = Value::Cell(
        CellArray::new(vec![nested_cell(second), nested_cell(first)], 2, 1).expect("B column"),
    );
    let a = Value::Cell(
        CellArray::new(vec![nested_struct(second), nested_struct(first)], 2, 1).expect("A column"),
    );
    let mut right =
        super::super::super::table_from_columns(vec!["B".into(), "A".into()], vec![b, a])
            .expect("right table");
    set_row_names(&mut right, &["second", "first"]);
    right
}

#[derive(Clone, Copy)]
enum Publication {
    Published,
    Rejected,
    Abandoned,
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn selected_nested_rhs_retains_targets_through_publication_failure_and_abandonment() {
    runmat_gc::gc_test_context(|| {
        for outcome in [
            Publication::Published,
            Publication::Rejected,
            Publication::Abandoned,
        ] {
            let first = runmat_gc::gc_allocate_rooted(Value::String("first".into()))
                .expect("allocate first target");
            let second = runmat_gc::gc_allocate_rooted(Value::String("second".into()))
                .expect("allocate second target");
            let first_handle = first.handle();
            let second_handle = second.handle();
            let mut left = table(&["A", "B"], &[&[1.0, 2.0], &[3.0, 4.0]]);
            set_row_names(&mut left, &["first", "second"]);
            let right = managed_right(first_handle, second_handle);
            let inputs = RootedValueLease::for_values([&left, &right].into_iter())
                .expect("retain original inputs during planning");
            let BinaryInputPlan::Structured(plan) =
                super::super::plan_binary(left, right).expect("align managed table")
            else {
                panic!("expected structured plan")
            };
            first.unroot().expect("retire first allocation root");
            second.unroot().expect("retire second allocation root");
            drop(inputs);

            let Value::Cell(selected_a) = &plan.right_values.values()[0] else {
                panic!("expected selected A cell")
            };
            let Value::Struct(first_row) = &selected_a.data()[0] else {
                panic!("expected nested A struct")
            };
            assert_eq!(first_row.fields["Target"], target(first_handle));
            let Value::Cell(selected_b) = &plan.right_values.values()[1] else {
                panic!("expected selected B cell")
            };
            let Value::Cell(first_nested) = &selected_b.data()[0] else {
                panic!("expected nested B cell")
            };
            assert_eq!(first_nested.data()[0], target(first_handle));
            runmat_gc::gc_collect_major().expect("collect after original inputs retire");
            assert!(runmat_gc::gc_clone_value(&first_handle).is_ok());
            assert!(runmat_gc::gc_clone_value(&second_handle).is_ok());

            match outcome {
                Publication::Published => {
                    let names = plan.variables.into_iter().map(|(name, _)| name).collect();
                    let output =
                        super::super::finish_binary_owned(&plan.source, names, plan.right_values)
                            .expect("publish replacement table");
                    runmat_gc::gc_collect_major().expect("collect under final Object owner");
                    assert!(runmat_gc::gc_clone_value(&first_handle).is_ok());
                    assert!(runmat_gc::gc_clone_value(&second_handle).is_ok());
                    drop(output);
                }
                Publication::Rejected => {
                    let error = super::super::finish_binary_owned(
                        &plan.source,
                        vec!["A".into()],
                        plan.right_values,
                    )
                    .expect_err("mismatched names must reject replacement");
                    assert!(error.message().contains("owned column count"));
                }
                Publication::Abandoned => drop(plan),
            }
            runmat_gc::gc_collect_major().expect("collect after final owner retirement");
            assert!(runmat_gc::gc_clone_value(&first_handle).is_err());
            assert!(runmat_gc::gc_clone_value(&second_handle).is_err());
        }
    });
}
