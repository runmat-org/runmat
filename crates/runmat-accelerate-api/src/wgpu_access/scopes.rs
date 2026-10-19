//! Pop is synchronous; only detached completion futures may leave the gate.
use std::future::Future;

pub(super) trait ScopeDriver {
    type Pending: Future;
    fn push(&self, filter: wgpu::ErrorFilter);
    fn pop(&self) -> Self::Pending;
}

pub(super) struct ActiveScopes<'a, D: ScopeDriver> {
    driver: &'a D,
    remaining: usize,
}
impl<'a, D: ScopeDriver> ActiveScopes<'a, D> {
    pub fn push(driver: &'a D) -> Self {
        let mut scopes = Self {
            driver,
            remaining: 0,
        };
        scopes.push_one(wgpu::ErrorFilter::Validation);
        scopes.push_one(wgpu::ErrorFilter::OutOfMemory);
        scopes.push_one(wgpu::ErrorFilter::Internal);
        scopes
    }

    fn push_one(&mut self, filter: wgpu::ErrorFilter) {
        self.driver.push(filter);
        self.remaining += 1;
    }

    pub fn detach<T>(mut self, value: T) -> DetachedScopes<T, D::Pending> {
        let mut pending = Vec::with_capacity(self.remaining);
        while self.remaining != 0 {
            pending.push(self.driver.pop());
            self.remaining -= 1;
        }
        DetachedScopes { value, pending }
    }
}
impl<D: ScopeDriver> Drop for ActiveScopes<'_, D> {
    fn drop(&mut self) {
        // Also runs while the caller's closure unwinds. The matching pops occur
        // before the enclosing exclusive gate guard is released. Never wait.
        while self.remaining != 0 {
            drop(self.driver.pop());
            self.remaining -= 1;
        }
    }
}

pub(super) struct DetachedScopes<T, P> {
    value: T,
    pending: Vec<P>,
}
impl<T, P> DetachedScopes<T, P> {
    pub async fn resolve<E>(self) -> Result<T, Vec<E>>
    where
        P: Future<Output = Option<E>>,
    {
        let mut errors = Vec::new();
        for pending in self.pending {
            if let Some(error) = pending.await {
                errors.push(error);
            }
        }
        if errors.is_empty() {
            Ok(self.value)
        } else {
            Err(errors)
        }
    }
}
