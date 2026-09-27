use super::*;

#[derive(Default)]
pub(super) struct FakeDriver {
    pub(super) fail_push: AtomicBool,
    pub(super) fail_pop: AtomicBool,
    pub(super) fail_free: AtomicBool,
    pub(super) panic_pop: AtomicBool,
    pub(super) panic_free: AtomicBool,
    pub(super) panic_release: AtomicBool,
    pub(super) events: Mutex<Vec<&'static str>>,
    pub(super) on_free: Mutex<Option<Box<dyn FnOnce() + Send>>>,
    freed: Mutex<Vec<u64>>,
}

impl FakeDriver {
    pub(super) fn frees(&self) -> Vec<u64> {
        self.freed.lock().unwrap().clone()
    }
    pub(super) fn release_count(&self) -> usize {
        self.events
            .lock()
            .unwrap()
            .iter()
            .filter(|event| **event == "release")
            .count()
    }
}

impl ContextDriver for FakeDriver {
    fn push(&self, _: usize) -> Result<()> {
        self.events.lock().unwrap().push("push");
        if self.fail_push.swap(false, Ordering::SeqCst) {
            return Err(anyhow!("push failed"));
        }
        Ok(())
    }
    fn pop(&self) -> Result<()> {
        self.events.lock().unwrap().push("pop");
        if self.panic_pop.swap(false, Ordering::SeqCst) {
            panic!("pop unwound");
        }
        if self.fail_pop.swap(false, Ordering::SeqCst) {
            return Err(anyhow!("pop failed"));
        }
        Ok(())
    }
    fn free(&self, pointer: u64) -> Result<()> {
        self.events.lock().unwrap().push("free");
        self.freed.lock().unwrap().push(pointer);
        if self.panic_free.swap(false, Ordering::SeqCst) {
            panic!("free unwound");
        }
        let callback = self.on_free.lock().unwrap().take();
        if let Some(callback) = callback {
            callback();
        }
        if self.fail_free.swap(false, Ordering::SeqCst) {
            return Err(anyhow!("free uncertain"));
        }
        Ok(())
    }
    fn release(&self, _: i32) -> Result<()> {
        self.events.lock().unwrap().push("release");
        if self.panic_release.swap(false, Ordering::SeqCst) {
            panic!("release unwound");
        }
        Ok(())
    }
}
