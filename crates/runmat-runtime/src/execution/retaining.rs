//! Retire a pinned operation before releasing its associated lifetime owner.
//! This is inline state within the caller's existing continuation allocation.

use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

pin_project_lite::pin_project! {
    struct Retaining<F, T> {
        // Declaration order also preserves retirement on abandonment.
        #[pin]
        future: Option<F>,
        owner: Option<T>,
    }
}

// Construct in the caller instead of passing another full future-sized frame.
#[inline(always)]
pub(crate) fn with_retained_owner<F: Future, T>(
    future: F,
    owner: T,
) -> impl Future<Output = F::Output> {
    Retaining {
        future: Some(future),
        owner: Some(owner),
    }
}

impl<F: Future, T> Future for Retaining<F, T> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let mut this = self.project();
        match this
            .future
            .as_mut()
            .as_pin_mut()
            .expect("retained operation polled after completion")
            .poll(context)
        {
            Poll::Pending => Poll::Pending,
            Poll::Ready(output) => {
                this.future.set(None);
                this.owner.take();
                Poll::Ready(output)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::Cell, marker::PhantomPinned, rc::Rc};

    struct Operation {
        ready: Rc<Cell<bool>>,
        dropped: Rc<Cell<bool>>,
        address: Cell<Option<usize>>,
        error: bool,
        _pin: PhantomPinned,
    }

    impl Future for Operation {
        type Output = Result<usize, &'static str>;
        fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
            let address = self.as_ref().get_ref() as *const Self as usize;
            assert_eq!(
                self.address.replace(Some(address)).unwrap_or(address),
                address
            );
            if !self.ready.get() {
                Poll::Pending
            } else {
                Poll::Ready(if self.error {
                    Err("original error")
                } else {
                    Ok(42)
                })
            }
        }
    }
    impl Drop for Operation {
        fn drop(&mut self) {
            if let Some(address) = self.address.get() {
                assert_eq!(address, self as *const Self as usize);
            }
            assert!(!self.dropped.replace(true));
        }
    }
    struct Owner {
        operation: Rc<Cell<bool>>,
        dropped: Rc<Cell<bool>>,
    }
    impl Drop for Owner {
        fn drop(&mut self) {
            assert!(self.operation.get(), "operation retires before owner");
            assert!(!self.dropped.replace(true));
        }
    }

    #[test]
    fn retained_owner_preserves_pin_error_completion_and_abandonment_order() {
        for mode in 0..4 {
            let ready = Rc::new(Cell::new(false));
            let operation = Rc::new(Cell::new(false));
            let owner = Rc::new(Cell::new(false));
            let mut future = Box::pin(with_retained_owner(
                Operation {
                    ready: ready.clone(),
                    dropped: operation.clone(),
                    address: Cell::new(None),
                    error: mode == 3,
                    _pin: PhantomPinned,
                },
                Owner {
                    operation: operation.clone(),
                    dropped: owner.clone(),
                },
            ));
            if mode != 0 {
                let mut context = Context::from_waker(futures::task::noop_waker_ref());
                assert_eq!(future.as_mut().poll(&mut context), Poll::Pending);
                assert!(!operation.get());
                assert!(!owner.get());
                if mode >= 2 {
                    ready.set(true);
                    assert_eq!(
                        future.as_mut().poll(&mut context),
                        Poll::Ready(if mode == 3 {
                            Err("original error")
                        } else {
                            Ok(42)
                        })
                    );
                    assert!(operation.get());
                    assert!(
                        owner.get(),
                        "completed future can remain held without retaining its owner"
                    );
                }
            }
            drop(future);
            assert!(operation.get());
            assert!(owner.get());
        }
    }
}
