use super::*;
use crate::fusion::FusionPlan;
use futures::task::noop_waker;
use std::cell::Cell;
use std::rc::Rc;

fn assert_active(plan: &FusionPlanRef, group: Option<usize>) {
    with_active_context(|active| {
        let active = active.as_ref().expect("selected frame plan");
        assert!(Rc::ptr_eq(&active.plan, plan));
        assert_eq!(active.active_group, group);
    });
}

struct PendingFrame {
    plan: FusionPlanRef,
    polled: bool,
    drops: Rc<Cell<usize>>,
}

impl Future for PendingFrame {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<()> {
        assert_active(&self.plan, self.polled.then_some(11));
        if self.polled {
            return Poll::Ready(());
        }
        with_active_context(|active| active.as_mut().unwrap().active_group = Some(11));
        // A nested frame without a plan cannot inherit this frame's group.
        futures::executor::block_on(scope_fusion_plan(None, async {
            with_active_context(|active| assert!(active.is_none()));
        }));
        assert_active(&self.plan, Some(11));
        self.polled = true;
        Poll::Pending
    }
}

impl Drop for PendingFrame {
    fn drop(&mut self) {
        assert_active(&self.plan, self.polled.then_some(11));
        self.drops.set(self.drops.get() + 1);
    }
}

#[test]
fn interleaved_frames_restore_caller_and_retain_groups_through_ready_and_cancel() {
    let caller = Rc::new(FusionPlan { groups: Vec::new() });
    let mut caller_context = Some(ActiveContext {
        plan: Rc::clone(&caller),
        active_group: Some(7),
    });
    let _caller = Activation::enter(&mut caller_context);
    let left = Rc::new(FusionPlan { groups: Vec::new() });
    let right = Rc::new(FusionPlan { groups: Vec::new() });
    let left_drops = Rc::new(Cell::new(0));
    let right_drops = Rc::new(Cell::new(0));
    let mut left_future = Box::pin(scope_fusion_plan(
        Some(Rc::clone(&left)),
        PendingFrame {
            plan: left,
            polled: false,
            drops: Rc::clone(&left_drops),
        },
    ));
    let mut right_future = Box::pin(scope_fusion_plan(
        Some(Rc::clone(&right)),
        PendingFrame {
            plan: right,
            polled: false,
            drops: Rc::clone(&right_drops),
        },
    ));
    let waker = noop_waker();
    let mut cx = Context::from_waker(&waker);
    assert!(left_future.as_mut().poll(&mut cx).is_pending());
    assert_active(&caller, Some(7));
    assert!(right_future.as_mut().poll(&mut cx).is_pending());
    assert_active(&caller, Some(7));
    drop(left_future);
    assert_eq!(left_drops.get(), 1);
    assert_active(&caller, Some(7));
    assert!(right_future.as_mut().poll(&mut cx).is_ready());
    assert_eq!(right_drops.get(), 1);
    assert_active(&caller, Some(7));
    drop(right_future);
    assert_eq!(right_drops.get(), 1);
    assert_active(&caller, Some(7));
}
