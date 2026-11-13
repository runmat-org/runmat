use super::*;
use crate::execution::RootedValueMember;
use runmat_execution::value::{DurableStructArrayField, InlineValue, ValuePayload};
use runmat_value::{HandleRef, ObjectArrayAdmission, ObjectInstance};

fn handle_value(class: &str, target: runmat_gc::GcHandle) -> Value {
    Value::HandleObject(HandleRef {
        class_name: runmat_types::ClassIdentity::new(class).unwrap().into(),
        target,
        valid: true,
    })
}

#[test]
fn object_array_handle_storage_uses_real_managed_targets_and_rejects_mixing() {
    runmat_gc::gc_test_context(|| {
        let first = runmat_gc::gc_allocate_rooted(Value::String("first".into())).unwrap();
        let second = runmat_gc::gc_allocate_rooted(Value::String("second".into())).unwrap();
        let class = runmat_types::ClassIdentity::new("ManagedArray").unwrap();
        let handles = vec![
            HandleRef {
                class_name: class.clone().into(),
                target: first.handle(),
                valid: true,
            },
            HandleRef {
                class_name: class.clone().into(),
                target: second.handle(),
                valid: true,
            },
        ];
        let array = ObjectArrayAdmission::handles(class.clone(), handles, vec![1, 2])
            .unwrap()
            .finish_synchronously()
            .unwrap()
            .into_array();
        assert_eq!(array.len(), 2);

        let error = ObjectArrayAdmission::values(
            class,
            vec![
                Value::Object(ObjectInstance::new("ManagedArray")),
                handle_value("ManagedArray", first.handle()),
            ],
            vec![1, 2],
        )
        .unwrap()
        .finish_synchronously()
        .expect_err("mixed storage must be rejected");
        assert_eq!(
            error,
            "object array cannot mix value-object and handle-object storage"
        );
    });
}

#[test]
fn struct_admission_retains_managed_children_until_pending_owner_drops() {
    runmat_gc::gc_test_context(|| {
        let initial = runmat_gc::gc_allocate_rooted(Value::String("child".into())).unwrap();
        let handle = initial.handle();
        let child = RootedValueMember::root(handle_value("ManagedChild", handle)).unwrap();
        initial.unroot().unwrap();

        let source = DurableStructArray {
            shape: vec![1, 1],
            fields: vec![DurableStructArrayField {
                name: "member".into(),
                values: vec![ValuePayload::Inline(Box::new(InlineValue::I32(1)))],
            }],
        };
        let control = ControlOwner::new(None).unwrap();
        let mut partials = control.vector(1, "root").unwrap();
        partials.push(child, "root").unwrap();
        let mut task = StructArrayTask::new(
            &source,
            TakenValues::from_partials(partials, None).unwrap(),
            control.path("root").unwrap(),
        )
        .unwrap();
        while task.admission.is_none() {
            let (completed, value) = task.advance(1).unwrap();
            assert_eq!(completed, 1);
            assert!(value.is_none());
        }
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_ok());

        let (completed, value) = task.advance(1).unwrap();
        assert_eq!(completed, 1);
        assert!(value.is_none());
        drop(task);
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_err());
    });
}
