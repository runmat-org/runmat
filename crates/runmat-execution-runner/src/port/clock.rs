pub trait Clock {
    fn now_millis(&self) -> u64;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ManualClock {
    now_millis: u64,
}

impl ManualClock {
    pub fn new(now_millis: u64) -> Self {
        Self { now_millis }
    }

    pub fn set(&mut self, now_millis: u64) {
        self.now_millis = now_millis;
    }
}

impl Clock for ManualClock {
    fn now_millis(&self) -> u64 {
        self.now_millis
    }
}
