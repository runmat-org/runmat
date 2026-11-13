//! Exact child-side project load admission and live user-owner inventory.

use super::*;

impl NativeFfiAdapter {
    /// The child reports its actual owner inventory after every completed
    /// operation. An unloaded user library ceases to block replacement unless
    /// a live pointer still keeps its native code reachable.
    pub(crate) fn has_user_native_state(&self) -> bool {
        let state = self.state.borrow();
        if state.libraries.values().any(|entry| !entry.project_owned) {
            return true;
        }
        state.pointers.values().any(|pointer| {
            let library = match pointer {
                PointerEntry::CallerOwned { library, .. } => library.as_ref(),
                PointerEntry::Opaque { library, .. } => Some(library),
            };
            !library.is_some_and(|library| {
                state
                    .libraries
                    .values()
                    .any(|entry| entry.project_owned && Rc::ptr_eq(&entry.library, library))
            })
        })
    }

    /// Child-only admission of one authenticated project load. The parent
    /// retains the private root lease; this process resolves no ambient path.
    pub(crate) fn install_project_load(
        &self,
        request: NativeProjectLoadRequest,
    ) -> Result<(), RuntimeError> {
        request.validate().map_err(invalid_call)?;
        let root = std::fs::canonicalize(&request.private_root).map_err(|error| {
            invalid_call(format!("native project root is unavailable: {error}"))
        })?;
        if root != request.private_root {
            return Err(invalid_call("native project root changed after admission"));
        }
        let library = root.join(&request.library);
        let manifest = root.join(&request.manifest);
        for path in [&library, &manifest] {
            if !std::fs::canonicalize(path)
                .map_err(|error| {
                    invalid_call(format!("native project path is unavailable: {error}"))
                })?
                .starts_with(&root)
            {
                return Err(invalid_call("native project path escapes its private root"));
            }
        }
        let prepared = self.prepare_project_artifact_with_policy(
            &library,
            &manifest,
            Some((request.library_digest, request.manifest_digest)),
            NativeFfiIsolationPolicy::in_process(),
            Some((&request.alias, &request.artifact_identity)),
        )?;
        let entry = prepared
            .library
            .ok_or_else(|| invalid_call("native project load has no library"))?;
        self.state
            .borrow_mut()
            .libraries
            .insert(request.alias, entry);
        Ok(())
    }
}
