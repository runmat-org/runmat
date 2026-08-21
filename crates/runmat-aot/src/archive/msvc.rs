use std::collections::HashMap;
use std::io::Cursor;

use ar_archive_writer::{
    write_archive_to_stream, ArchiveKind as WriterArchiveKind, NewArchiveMember,
    DEFAULT_OBJECT_READER,
};
use object::read::archive::{ArchiveFile, ArchiveKind as ReaderArchiveKind};
use runmat_execution::Digest;

use crate::{AotError, AotResult};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedMsvcRuntimeArchive {
    pub archive: Vec<u8>,
    pub native_link_tokens: Vec<String>,
    pub duplicate_members_removed: usize,
    pub bundled_link_tokens_removed: usize,
}

pub fn prepare_msvc_runtime_archive(
    archive: &[u8],
    native_link_tokens: Vec<String>,
) -> AotResult<PreparedMsvcRuntimeArchive> {
    let parsed = ArchiveFile::parse(archive).map_err(|error| {
        AotError::contract(
            "aot.archive.msvc",
            format!("parse MSVC runtime archive: {error}"),
        )
    })?;
    if parsed.kind() != ReaderArchiveKind::Coff {
        return Err(AotError::contract(
            "aot.archive.msvc",
            "MSVC runtime archive is not a COFF archive",
        ));
    }

    let mut members = Vec::<(String, Vec<u8>)>::new();
    let mut retained = HashMap::<(Vec<u8>, Digest), Vec<usize>>::new();
    let mut duplicate_members_removed = 0_usize;
    for member in parsed.members() {
        let member = member.map_err(|error| {
            AotError::contract(
                "aot.archive.msvc",
                format!("read MSVC runtime archive member: {error}"),
            )
        })?;
        let name_bytes = member.name().to_vec();
        let name = String::from_utf8(name_bytes.clone()).map_err(|_| {
            AotError::contract(
                "aot.archive.msvc",
                "MSVC runtime archive member name is not valid UTF-8",
            )
        })?;
        let data = member.data(archive).map_err(|error| {
            AotError::contract(
                "aot.archive.msvc",
                format!("read MSVC runtime archive member data: {error}"),
            )
        })?;
        let identity = (name_bytes, Digest::sha256(data));
        let is_exact_duplicate = retained.get(&identity).is_some_and(|indices| {
            indices
                .iter()
                .any(|index| members[*index].1.as_slice() == data)
        });
        if is_exact_duplicate {
            duplicate_members_removed += 1;
            continue;
        }
        let index = members.len();
        members.push((name, data.to_vec()));
        retained.entry(identity).or_default().push(index);
    }

    let archive = if duplicate_members_removed == 0 {
        archive.to_vec()
    } else {
        let new_members = members
            .iter()
            .map(|(name, data)| {
                NewArchiveMember::new(data.as_slice(), &DEFAULT_OBJECT_READER, name.clone())
            })
            .collect::<Vec<_>>();
        let mut output = Cursor::new(Vec::new());
        write_archive_to_stream(
            &mut output,
            &new_members,
            WriterArchiveKind::Coff,
            false,
            None,
        )
        .map_err(|error| {
            AotError::contract(
                "aot.archive.msvc",
                format!("write normalized MSVC runtime archive: {error}"),
            )
        })?;
        output.into_inner()
    };

    let original_token_count = native_link_tokens.len();
    let native_link_tokens = native_link_tokens
        .into_iter()
        .filter(|token| !is_bundled_windows_targets_import_library(token))
        .collect::<Vec<_>>();
    let bundled_link_tokens_removed = original_token_count - native_link_tokens.len();

    Ok(PreparedMsvcRuntimeArchive {
        archive,
        native_link_tokens,
        duplicate_members_removed,
        bundled_link_tokens_removed,
    })
}

fn is_bundled_windows_targets_import_library(token: &str) -> bool {
    let Some(version) = token
        .strip_prefix("windows.")
        .and_then(|token| token.strip_suffix(".lib"))
    else {
        return false;
    };
    let parts = version.split('.').collect::<Vec<_>>();
    parts.len() == 3
        && parts
            .iter()
            .all(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coff_archive(members: &[(&str, &[u8])]) -> Vec<u8> {
        let members = members
            .iter()
            .map(|(name, data)| {
                NewArchiveMember::new(*data, &DEFAULT_OBJECT_READER, (*name).to_string())
            })
            .collect::<Vec<_>>();
        let mut output = Cursor::new(Vec::new());
        write_archive_to_stream(&mut output, &members, WriterArchiveKind::Coff, false, None)
            .unwrap();
        output.into_inner()
    }

    #[test]
    fn removes_only_exact_duplicate_members_and_bundled_windows_token() {
        let archive = coff_archive(&[
            ("same.obj", b"identical"),
            ("same.obj", b"identical"),
            ("same.obj", b"different"),
            ("other.obj", b"identical"),
        ]);
        let prepared = prepare_msvc_runtime_archive(
            &archive,
            vec![
                "kernel32.lib".into(),
                "windows.0.52.0.lib".into(),
                "windows.custom.lib".into(),
            ],
        )
        .unwrap();
        assert_eq!(prepared.duplicate_members_removed, 1);
        assert_eq!(prepared.bundled_link_tokens_removed, 1);
        assert_eq!(
            prepared.native_link_tokens,
            vec!["kernel32.lib", "windows.custom.lib"]
        );
        let parsed = ArchiveFile::parse(prepared.archive.as_slice()).unwrap();
        assert_eq!(parsed.members().count(), 3);
    }
}
