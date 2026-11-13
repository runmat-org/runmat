use crate::ExecutableSource;

/// The captured project gives each source a paired owner, relative path, and
/// access path. These fields must never be selected from different rows.
#[derive(Clone, Debug)]
struct CapturedSourceCandidate {
    owner_identity: String,
    relative_path: String,
    access_path: String,
}

pub(in super::super) fn executable_source_identity(
    root: &ExecutableSource,
    handoff: Option<&runmat_package::FrozenProjectHandoff>,
    root_full_path: Option<&str>,
    display_name: &str,
    full_path: Option<&str>,
) -> Result<(String, String), String> {
    let candidates = handoff
        .map(|handoff| {
            handoff
                .project
                .all_sources()
                .map(|(descriptor, access)| CapturedSourceCandidate {
                    owner_identity: descriptor.id.package_instance.to_string(),
                    relative_path: descriptor.id.relative_path.to_string(),
                    access_path: access.to_string_lossy().replace('\\', "/"),
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    executable_source_identity_from_candidates(
        root,
        root_full_path,
        display_name,
        full_path,
        &candidates,
    )
}

fn executable_source_identity_from_candidates(
    root: &ExecutableSource,
    root_full_path: Option<&str>,
    display_name: &str,
    full_path: Option<&str>,
    candidates: &[CapturedSourceCandidate],
) -> Result<(String, String), String> {
    let display = display_name.replace('\\', "/");
    let root_relative = root.relative_path.replace('\\', "/");
    let full = full_path.map(|path| path.replace('\\', "/"));
    let root_full = root_full_path.map(|path| path.replace('\\', "/"));
    if let Some(captured) = root.captured_primary() {
        if full.as_deref() == Some(captured.access_path.as_str()) {
            if !captured.matches_access(captured.access_path.as_str(), &root.text) {
                return Err("captured primary access or bytes changed".into());
            }
            return Ok((
                captured.owner_identity.clone(),
                captured.relative_path.clone(),
            ));
        }
        if full.is_none() && display == root_relative {
            return Err("captured primary has no matching issued access path".into());
        }
    }
    if !candidates.is_empty() {
        if let Some(full) = &full {
            let mut exact = candidates
                .iter()
                .filter(|candidate| candidate.access_path.as_str() == full.as_str());
            if let Some(identity) = exact.next() {
                if exact.next().is_some() {
                    return Err("source access path has more than one project owner".into());
                }
                return Ok((
                    identity.owner_identity.clone(),
                    identity.relative_path.clone(),
                ));
            }
            // An externally captured full path cannot inherit the root
            // package owner merely because its display name differs. A
            // retained external definition needs its own captured owner.
            return Err(format!(
                "source path '{full}' does not match captured project access authority for '{display}'"
            ));
        }
        let mut relative = candidates
            .iter()
            .filter(|candidate| candidate.relative_path.replace('\\', "/") == display);
        if let Some(identity) = relative.next() {
            if relative.next().is_some() {
                return Err("relative source path has more than one project owner".into());
            }
            return Ok((
                identity.owner_identity.clone(),
                identity.relative_path.clone(),
            ));
        }
    }
    if display == root_relative
        && match (&full, &root_full) {
            (None, _) => true,
            (Some(full), Some(root_full)) => full == root_full,
            (Some(full), None) => full == &root_relative,
        }
    {
        return Ok((root.owner_identity.clone(), root.relative_path.clone()));
    }
    if display == root_relative {
        return Err(
            "source shares the primary relative path without matching access authority".into(),
        );
    }
    Ok((root.owner_identity.clone(), display))
}

pub(in super::super) fn executable_primary_access_path(
    root: &ExecutableSource,
    handoff: Option<&runmat_package::FrozenProjectHandoff>,
) -> Result<Option<String>, String> {
    let Some(handoff) = handoff else {
        return Ok(root.explicit_lookup_path().map(str::to_owned));
    };
    if let Some(captured) = root.captured_primary() {
        captured.validate(handoff, root)?;
        return Ok(Some(captured.access_path.clone()));
    }
    let mut matches = handoff.project.all_sources().filter(|(descriptor, _)| {
        descriptor.id.package_instance.to_string() == root.owner_identity
            && descriptor.id.relative_path.to_string().replace('\\', "/")
                == root.relative_path.replace('\\', "/")
    });
    let first = matches
        .next()
        .ok_or_else(|| "primary source does not belong to the captured project".to_string())?;
    if matches.next().is_some() {
        return Err("primary source has ambiguous captured access authority".into());
    }
    Ok(Some(first.1.to_string_lossy().replace('\\', "/")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(owner: &str, relative: &str, access: &str) -> CapturedSourceCandidate {
        CapturedSourceCandidate {
            owner_identity: owner.into(),
            relative_path: relative.into(),
            access_path: access.into(),
        }
    }

    #[test]
    fn only_explicit_lookup_path_grants_non_project_file_access() {
        let inline = ExecutableSource::new("unit", "/display/main.m", "x = 1;");
        assert_eq!(executable_primary_access_path(&inline, None).unwrap(), None);
        let captured = ExecutableSource::frozen("unit", "main.m", "/issued/main.m", "x = 1;");
        assert_eq!(
            executable_primary_access_path(&captured, None).unwrap(),
            Some("/issued/main.m".to_owned()),
        );
    }

    #[test]
    fn exact_access_selects_dependency_not_same_named_primary() {
        let root = ExecutableSource::new("root", "foo.m", "x = 1;");
        let candidates = [
            candidate("root", "foo.m", "/root/foo.m"),
            candidate("dependency", "foo.m", "/deps/foo.m"),
        ];
        assert_eq!(
            executable_source_identity_from_candidates(
                &root,
                Some("/root/foo.m"),
                "foo.m",
                Some("/deps/foo.m"),
                &candidates,
            )
            .unwrap(),
            ("dependency".into(), "foo.m".into())
        );
        assert!(executable_source_identity_from_candidates(
            &root,
            Some("/root/foo.m"),
            "foo.m",
            Some("/unowned/foo.m"),
            &candidates,
        )
        .is_err());
    }

    #[test]
    fn same_relative_path_in_two_packages_requires_captured_access() {
        let root = ExecutableSource::new("root", "main.m", "x = 1;");
        let candidates = [
            candidate("package-a", "foo.m", "/a/foo.m"),
            candidate("package-b", "foo.m", "/b/foo.m"),
        ];
        assert!(executable_source_identity_from_candidates(
            &root,
            Some("/root/main.m"),
            "foo.m",
            None,
            &candidates,
        )
        .is_err());
        assert_eq!(
            executable_source_identity_from_candidates(
                &root,
                Some("/root/main.m"),
                "foo.m",
                Some("/b/foo.m"),
                &candidates,
            )
            .unwrap(),
            ("package-b".into(), "foo.m".into())
        );
    }

    #[test]
    fn unowned_full_path_cannot_be_attributed_to_primary_package() {
        let root = ExecutableSource::new("root", "main.m", "x = 1;");
        let candidates = [candidate("root", "main.m", "/root/main.m")];
        assert!(executable_source_identity_from_candidates(
            &root,
            Some("/root/main.m"),
            "unowned.m",
            Some("/other/unowned.m"),
            &candidates,
        )
        .is_err());
    }

    #[test]
    fn no_project_handoff_keeps_unit_local_generated_source_namespace() {
        let root = ExecutableSource::new("local-unit", "main.m", "x = 1;");
        assert_eq!(
            executable_source_identity_from_candidates(&root, None, "<eval>", None, &[],).unwrap(),
            ("local-unit".into(), "<eval>".into())
        );
    }
}
