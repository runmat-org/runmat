//! One immutable direct-call projection; no runtime issuance is cached here.
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use runmat_vm::SelectedExecutableCode;

use super::ExecutableUnit;

#[derive(Debug, Default)]
pub(super) struct DirectCodeCache(RefCell<Option<CachedCode>>);

#[derive(Debug)]
struct CachedCode {
    interop: runmat_types::InteropManifest,
    code: Rc<SelectedExecutableCode>,
}

impl ExecutableUnit {
    pub(crate) fn cached_direct_code(
        &self,
        interop: &runmat_types::InteropManifest,
    ) -> Option<Rc<SelectedExecutableCode>> {
        self.direct_code_cache
            .0
            .borrow()
            .as_ref()
            .filter(|cached| &cached.interop == interop)
            .map(|cached| Rc::clone(&cached.code))
    }

    pub(crate) fn admit_direct_code(
        &self,
        interop: runmat_types::InteropManifest,
        encode: impl FnOnce() -> Result<Arc<[u8]>, String>,
    ) -> Result<Rc<SelectedExecutableCode>, String> {
        if let Some(code) = self.cached_direct_code(&interop) {
            return Ok(code);
        }
        let code = SelectedExecutableCode::admit(encode()?)?;
        *self.direct_code_cache.0.borrow_mut() = Some(CachedCode {
            interop,
            code: Rc::clone(&code),
        });
        Ok(code)
    }

    /// A consuming identity/projection change must not poison an unchanged
    /// clone's cache or reuse that clone's differently selected product.
    pub(crate) fn reset_direct_code_cache(&mut self) {
        self.direct_code_cache = Rc::new(DirectCodeCache::default());
    }
}
