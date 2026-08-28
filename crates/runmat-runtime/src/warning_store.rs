use std::cell::RefCell;

#[derive(Clone, Debug)]
pub struct RuntimeWarning {
    pub identifier: String,
    pub message: String,
}

thread_local! {
    static WARNINGS: RefCell<Vec<RuntimeWarning>> = const { RefCell::new(Vec::new()) };
}

pub fn push(identifier: &str, message: &str) {
    if let Some(context) = crate::context::legacy::active() {
        context.state().warnings.borrow_mut().push(RuntimeWarning {
            identifier: identifier.to_string(),
            message: message.to_string(),
        });
        return;
    }
    WARNINGS.with(|warnings| {
        warnings.borrow_mut().push(RuntimeWarning {
            identifier: identifier.to_string(),
            message: message.to_string(),
        })
    });
}

pub fn take_all() -> Vec<RuntimeWarning> {
    if let Some(context) = crate::context::legacy::active() {
        return context.state().warnings.borrow_mut().drain(..).collect();
    }
    WARNINGS.with(|warnings| warnings.borrow_mut().drain(..).collect())
}

pub fn extend(warnings: impl IntoIterator<Item = RuntimeWarning>) {
    if let Some(context) = crate::context::legacy::active() {
        context.state().warnings.borrow_mut().extend(warnings);
        return;
    }
    WARNINGS.with(|slot| slot.borrow_mut().extend(warnings));
}

pub fn reset() {
    if let Some(context) = crate::context::legacy::active() {
        context.state().warnings.borrow_mut().clear();
        return;
    }
    WARNINGS.with(|warnings| warnings.borrow_mut().clear());
}
