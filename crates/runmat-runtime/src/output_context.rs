use runmat_thread_local::runmat_thread_local;
use std::cell::RefCell;
use std::rc::Rc;

use crate::context::output_projection::{InvocationOutputProjection, PresentationOutputToken};

runmat_thread_local! {
    static PRESENTATION_OUTPUT_PROJECTION: RefCell<InvocationOutputProjection> =
        const { RefCell::new(InvocationOutputProjection::new()) };
}

pub struct OutputCountGuard {
    token: PresentationOutputToken,
    owner: Option<Rc<RefCell<InvocationOutputProjection>>>,
}

impl Drop for OutputCountGuard {
    fn drop(&mut self) {
        if let Some(owner) = &self.owner {
            owner.borrow_mut().remove_presentation(self.token);
        } else {
            PRESENTATION_OUTPUT_PROJECTION.with(|projection| {
                projection.borrow_mut().remove_presentation(self.token);
            });
        }
    }
}

pub fn push_output_count(count: usize) -> OutputCountGuard {
    if let Some(owner) = active_projection() {
        let token = owner.borrow_mut().push_presentation(count);
        return OutputCountGuard {
            token,
            owner: Some(owner),
        };
    }
    let token = PRESENTATION_OUTPUT_PROJECTION
        .with(|projection| projection.borrow_mut().push_presentation(count));
    OutputCountGuard { token, owner: None }
}

pub fn requested_output_count() -> Option<usize> {
    if let Some(owner) = active_projection() {
        return owner.borrow().current_presentation();
    }
    PRESENTATION_OUTPUT_PROJECTION.with(|projection| projection.borrow().current_presentation())
}

fn active_projection() -> Option<Rc<RefCell<InvocationOutputProjection>>> {
    crate::context::legacy::active().map(|context| Rc::clone(context.output_projection()))
}
