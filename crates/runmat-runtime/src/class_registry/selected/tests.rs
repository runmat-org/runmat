use std::collections::HashMap;
use std::rc::Rc;
use std::task::Poll;

use runmat_types::{ClassIdentity, MemberAccess, MemberName};
use runmat_value::Value;

use super::super::{
    prepare_selected_class_projection, CheckedClassRegistration, RuntimeClass, RuntimeProperty,
};

fn product(seed: f64) -> Rc<super::SelectedClassProjection> {
    let base = ClassIdentity::new("RetainedBase").expect("class identity");
    let derived = ClassIdentity::new("RetainedDerived").expect("class identity");
    let property = RuntimeProperty {
        name: MemberName::from("seed"),
        is_static: false,
        is_constant: false,
        is_dependent: false,
        get_access: MemberAccess::Public,
        set_access: MemberAccess::Public,
        default_value: Some(Value::Num(seed)),
    };
    let class = |name, parent, properties| CheckedClassRegistration {
        class: RuntimeClass {
            name,
            parent,
            properties,
            methods: HashMap::new(),
            events: HashMap::new(),
        },
        method_bindings: HashMap::new(),
        is_sealed: false,
        is_abstract: false,
        enumerations: Vec::new(),
    };
    prepare_selected_class_projection(vec![
        class(
            base.clone(),
            None,
            HashMap::from([(property.name.clone(), property)]),
        ),
        class(derived, Some(base), HashMap::new()),
    ])
    .expect("closed selected product")
}

async fn pause_once() {
    let mut first = true;
    futures::future::poll_fn(move |cx| {
        if std::mem::take(&mut first) {
            cx.waker().wake_by_ref();
            Poll::Pending
        } else {
            Poll::Ready(())
        }
    })
    .await
}

#[test]
fn suspended_sibling_constructors_keep_distinct_selected_defaults() {
    let runtime = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let old = runtime
        .clone()
        .with_selected_class_projection(product(11.0));
    let new = runtime
        .clone()
        .with_selected_class_projection(product(31.0));
    let identity = ClassIdentity::new("RetainedDerived").expect("class identity");
    let make = |context: crate::context::RuntimeContext| {
        let identity = identity.clone();
        context.scope(async move {
            pause_once().await;
            let value = crate::create_class_object_by_identity(identity)
                .await
                .expect("selected object");
            let Value::Object(object) = value else {
                panic!("selected value class must be an object");
            };
            object.property("seed").cloned()
        })
    };
    let (old_seed, new_seed) =
        futures::executor::block_on(async { futures::future::join(make(old), make(new)).await });
    assert_eq!(old_seed, Some(Value::Num(11.0)));
    assert_eq!(new_seed, Some(Value::Num(31.0)));
    assert!(super::super::get_class(&identity).is_none());
}

#[test]
fn selected_receiver_retains_canonical_dynamicprops_ancestry() {
    let base = ClassIdentity::new("SelectedDynamicBase").expect("class identity");
    let derived = ClassIdentity::new("SelectedDynamicDerived").expect("class identity");
    let registration = |name, parent| CheckedClassRegistration {
        class: RuntimeClass {
            name,
            parent,
            properties: HashMap::new(),
            methods: HashMap::new(),
            events: HashMap::new(),
        },
        method_bindings: HashMap::new(),
        is_sealed: false,
        is_abstract: false,
        enumerations: Vec::new(),
    };
    let product = prepare_selected_class_projection(vec![
        registration(
            base.clone(),
            Some(runmat_types::standard::DYNAMIC_PROPERTIES.owned()),
        ),
        registration(derived.clone(), Some(base.clone())),
    ])
    .expect("checked selected ancestry");
    let facts = product
        .receiver_declarations(&derived)
        .expect("complete selected receiver declarations");
    assert_eq!(
        facts.ancestry(),
        &[
            derived,
            base,
            runmat_types::standard::DYNAMIC_PROPERTIES.owned(),
        ]
    );
    assert!(facts.is_handle_class());
    assert_eq!(
        facts
            .class_declarations()
            .get(&runmat_types::standard::DYNAMIC_PROPERTIES.owned())
            .and_then(|class| class.parent.as_ref()),
        Some(&runmat_types::standard::HANDLE.owned())
    );
}
