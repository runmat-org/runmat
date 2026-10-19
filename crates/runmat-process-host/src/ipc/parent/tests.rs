use super::*;

#[test]
fn parent_loss_does_not_depend_on_executor_polling_and_disarm_preserves_success() {
    let armed = AtomicBool::new(true);
    assert!(parent_loss(&b""[..], &armed));
    assert!(parent_loss(&b"unexpected request"[..], &armed));
    armed.store(false, Ordering::Release);
    assert!(!parent_loss(&b""[..], &armed));
}

#[test]
fn transient_interruptions_do_not_end_an_attached_worker() {
    struct InterruptedOnce(bool);
    impl Read for InterruptedOnce {
        fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
            if !self.0 {
                self.0 = true;
                Err(std::io::ErrorKind::Interrupted.into())
            } else {
                Ok(0)
            }
        }
    }
    assert!(parent_loss(InterruptedOnce(false), &AtomicBool::new(true)));
}
