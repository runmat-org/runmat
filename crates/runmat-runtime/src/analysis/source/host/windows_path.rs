use std::path::{Component, Path, Prefix};

/// Compare normalized source components across Win32 and extended-length namespaces.
/// Keep the original paths for I/O, including their long-path prefixes.
pub(super) fn strip_root<'a>(path: &'a Path, root: &Path) -> Option<&'a Path> {
    let mut remaining = path.components();
    for expected in root.components() {
        let actual = remaining.next()?;
        let equal = match (expected, actual) {
            (Component::Prefix(expected), Component::Prefix(actual)) => {
                match (expected.kind(), actual.kind()) {
                    (
                        Prefix::Disk(left) | Prefix::VerbatimDisk(left),
                        Prefix::Disk(right) | Prefix::VerbatimDisk(right),
                    ) => left.eq_ignore_ascii_case(&right),
                    (
                        Prefix::UNC(left_server, left_share)
                        | Prefix::VerbatimUNC(left_server, left_share),
                        Prefix::UNC(right_server, right_share)
                        | Prefix::VerbatimUNC(right_server, right_share),
                    ) => left_server == right_server && left_share == right_share,
                    _ => expected == actual,
                }
            }
            _ => expected == actual,
        };
        if !equal {
            return None;
        }
    }
    Some(remaining.as_path())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_containment_preserves_volume_share_and_directory_boundaries() {
        for (roots, paths) in [
            (
                [r"C:\project", r"\\?\C:\project"],
                [
                    r"C:\project\models\main.fea",
                    r"\\?\C:\project\models\main.fea",
                ],
            ),
            (
                [r"\\server\share\project", r"\\?\UNC\server\share\project"],
                [
                    r"\\server\share\project\models\main.fea",
                    r"\\?\UNC\server\share\project\models\main.fea",
                ],
            ),
        ] {
            for root in roots {
                for path in paths {
                    assert_eq!(
                        strip_root(Path::new(path), Path::new(root)),
                        Some(Path::new(r"models\main.fea"))
                    );
                }
            }
        }
        for path in [
            r"D:\project\main.fea",
            r"\\?\D:\project\main.fea",
            r"C:\project-other\main.fea",
            r"\\.\C:\project\main.fea",
        ] {
            assert!(
                strip_root(Path::new(path), Path::new(r"\\?\C:\project")).is_none(),
                "{path}"
            );
        }
        assert!(strip_root(
            Path::new(r"\\server\other\project\main.fea"),
            Path::new(r"\\?\UNC\server\share\project")
        )
        .is_none());
    }
}
