use super::*;
use crate::context::RuntimeWorkspaceService;
use crate::workspace::{BaseWorkspaceOwner, BaseWorkspaceProjectionService};
use runmat_value::Value;
use std::cell::RefCell;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn committed_last_writer_origin_survives_service_drop() {
    let owner = BaseWorkspaceOwner::new();
    let observed = Rc::new(RefCell::new(Vec::new()));
    let observer = WorkspacePublicationObserver::new({
        let observed = Rc::clone(&observed);
        move |changes| {
            observed.borrow_mut().extend_from_slice(changes);
            Ok(())
        }
    });
    let service = BaseWorkspaceProjectionService::new(owner.begin_projection().unwrap())
        .with_publication_observer(observer);
    service
        .assign_executor_slot(2, "source", Value::Num(1.0))
        .unwrap();
    service.assign("source", Value::Num(3.0)).unwrap();
    service.assign("written", Value::Num(-1.0)).unwrap();
    service
        .assign_executor_slot(4, "written", Value::Num(4.0))
        .unwrap();
    service.publish().unwrap();
    drop(service);
    assert_eq!(
        *observed.borrow(),
        vec![
            WorkspacePublicationOrigin::DynamicEffect {
                name: "source".into()
            },
            WorkspacePublicationOrigin::ExecutorSlot {
                name: "written".into(),
                slot: 4
            },
        ]
    );
    assert_eq!(owner.lookup("source").unwrap(), Some(Value::Num(3.0)));
    assert_eq!(owner.lookup("written").unwrap(), Some(Value::Num(4.0)));
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn failed_publication_never_reports_staged_source_origin() {
    let owner = BaseWorkspaceOwner::new();
    let observed = Rc::new(RefCell::new(Vec::new()));
    let observer = WorkspacePublicationObserver::new({
        let observed = Rc::clone(&observed);
        move |changes| {
            observed.borrow_mut().extend_from_slice(changes);
            Ok(())
        }
    });
    let service = BaseWorkspaceProjectionService::new(owner.begin_projection().unwrap())
        .with_publication_observer(observer);
    service
        .assign_executor_slot(1, "unpublished", Value::Num(7.0))
        .unwrap();
    owner.close().unwrap();
    assert!(service.publish().is_err());
    assert!(observed.borrow().is_empty());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn clear_then_source_store_reports_the_post_clear_origin() {
    let owner = BaseWorkspaceOwner::new();
    let observed = Rc::new(RefCell::new(Vec::new()));
    let observer = WorkspacePublicationObserver::new({
        let observed = Rc::clone(&observed);
        move |changes| {
            observed.borrow_mut().extend_from_slice(changes);
            Ok(())
        }
    });
    let service = BaseWorkspaceProjectionService::new(owner.begin_projection().unwrap())
        .with_publication_observer(observer);
    service
        .assign_executor_slot(1, "retired", Value::Num(1.0))
        .unwrap();
    service.clear().unwrap();
    service
        .assign_executor_slot(2, "restored", Value::Num(2.0))
        .unwrap();
    service.publish().unwrap();
    assert_eq!(
        *observed.borrow(),
        vec![
            WorkspacePublicationOrigin::Cleared,
            WorkspacePublicationOrigin::ExecutorSlot {
                name: "restored".into(),
                slot: 2,
            },
        ]
    );
    assert_eq!(owner.lookup("retired").unwrap(), None);
    assert_eq!(owner.lookup("restored").unwrap(), Some(Value::Num(2.0)));
}
