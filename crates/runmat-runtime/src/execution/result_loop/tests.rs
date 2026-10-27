use super::*;
use std::{
    cell::{Cell, RefCell},
    marker::PhantomPinned,
    rc::Rc,
};

struct Step {
    index: usize,
    address: Cell<Option<usize>>,
    ready: Rc<Cell<bool>>,
    retired: Rc<RefCell<Vec<usize>>>,
    error: bool,
    _pin: PhantomPinned,
}
impl Future for Step {
    type Output = Result<ControlFlow<usize, usize>, &'static str>;
    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.as_ref().get_ref();
        let address = this as *const Self as usize;
        assert_eq!(
            this.address.replace(Some(address)).unwrap_or(address),
            address
        );
        if !this.ready.get() {
            return Poll::Pending;
        }
        Poll::Ready(if this.error {
            Err("original step error")
        } else if this.index == 2 {
            Ok(ControlFlow::Break(42))
        } else {
            Ok(ControlFlow::Continue(this.index + 1))
        })
    }
}
impl Drop for Step {
    fn drop(&mut self) {
        if let Some(address) = self.address.get() {
            assert_eq!(address, self as *const Self as usize);
        }
        self.retired.borrow_mut().push(self.index);
    }
}

#[test]
fn steps_preserve_pin_retirement_pending_errors_and_abandonment() {
    for (error, abandon) in [(false, false), (true, false), (false, true)] {
        let ready = Rc::new(Cell::new(false));
        let retired = Rc::new(RefCell::new(Vec::new()));
        let (selected_ready, selected_retired) = (ready.clone(), retired.clone());
        let mut work = Box::pin(loop_result(0, move |index| {
            assert_eq!(*selected_retired.borrow(), (0..index).collect::<Vec<_>>());
            Step {
                index,
                address: Cell::new(None),
                ready: selected_ready.clone(),
                retired: selected_retired.clone(),
                error,
                _pin: PhantomPinned,
            }
        }));
        let waker = futures::task::noop_waker();
        let mut context = Context::from_waker(&waker);
        assert_eq!(work.as_mut().poll(&mut context), Poll::Pending);
        if !abandon {
            ready.set(true);
            assert_eq!(
                work.as_mut().poll(&mut context),
                Poll::Ready(if error {
                    Err("original step error")
                } else {
                    Ok(42)
                })
            );
        }
        drop(work);
        assert_eq!(
            *retired.borrow(),
            if error || abandon {
                vec![0]
            } else {
                vec![0, 1, 2]
            }
        );
    }
}
