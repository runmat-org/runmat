//! Acquisition of the stock compile-time runtime. Only compile and explicit
//! installation enter this module; ordinary execution has no download path.

use anyhow::{Context, Result};
use runmat_aot::archive::{decode_runtime_pack, RuntimeArchive, RuntimePackManifest};
use runmat_package_cache_native::{concurrency::ProcessLock, filesystem::atomic_write};
use std::io::{self, IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

pub async fn resolve(install: bool, offline: bool) -> Result<RuntimeArchive> {
    tokio::task::spawn_blocking(move || resolve_blocking(install, offline, None))
        .await
        .context("compile-time runtime installer task failed")?
}

pub async fn install(from: Option<PathBuf>, offline: bool) -> Result<()> {
    tokio::task::spawn_blocking(move || {
        resolve_blocking(true, offline, from.as_deref())?;
        eprintln!("The compile-time AOT runtime is installed.");
        Ok(())
    })
    .await
    .context("compile-time runtime installer task failed")?
}

fn resolve_blocking(install: bool, offline: bool, from: Option<&Path>) -> Result<RuntimeArchive> {
    if let Some(runtime) = runmat_aot::archive::embedded_runtime_archive()? {
        anyhow::ensure!(
            from.is_none(),
            "this build already embeds its compile-time runtime"
        );
        return Ok(runtime);
    }
    let expected = runmat_aot::archive::embedded_runtime_pack_manifest()?.context(
        "this build has no compile-time runtime distribution manifest; use an official RunMat distribution or build one with scripts/build-runmat-with-aot-runtime.sh",
    )?;
    let root = std::env::var_os("RUNMAT_AOT_CACHE_DIR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| dirs::cache_dir().map(|root| root.join("runmat/aot-runtime")))
        .context("cannot locate the user cache directory")?;
    let cache = PackCache { root };
    // An explicitly supplied pack must be verified even on a cache hit.
    let local = from
        .map(|path| {
            let bytes = read_bounded(
                std::fs::File::open(path).with_context(|| format!("open `{}`", path.display()))?,
                expected.pack_bytes,
            )?;
            decode_runtime_pack(&expected, &bytes)?;
            Ok::<_, anyhow::Error>(bytes)
        })
        .transpose()?;
    if let Some(runtime) = cache.load(&expected)? {
        return Ok(runtime);
    }
    if offline && from.is_none() {
        anyhow::bail!("the matching AOT runtime is not cached; install it while connected with `runmat aot install`, or provision a downloaded pack with `runmat aot install --from PATH`");
    }
    if !install {
        if !io::stdin().is_terminal() || !io::stderr().is_terminal() {
            anyhow::bail!("the compile-time AOT runtime is not installed; run `runmat aot install` or compile with `--install-aot-runtime`");
        }
        if !confirm_install(
            &mut io::stdin().lock(),
            &mut io::stderr().lock(),
            expected.pack_bytes,
        )? {
            anyhow::bail!("compile-time runtime installation declined; install it later with `runmat aot install`");
        }
    }
    // Consent happens before taking the process lock. Recheck after locking:
    // another compiler may have completed the same installation meanwhile.
    cache.install(&expected, || {
        if let Some(bytes) = local {
            Ok(bytes)
        } else {
            download(&expected)
        }
    })
}

fn confirm_install(
    input: &mut impl io::BufRead,
    output: &mut impl Write,
    pack_bytes: u64,
) -> Result<bool> {
    write!(output, "The compile-time AOT runtime is not installed ({pack_bytes} bytes). Download and install it now? [y/N] ")?;
    output.flush()?;
    let mut answer = String::new();
    // Bound input even when a terminal is fed an unexpectedly long line.
    io::BufRead::read_line(&mut input.take(64), &mut answer)?;
    if answer.len() == 64 && !answer.ends_with('\n') {
        return Ok(false);
    }
    Ok(matches!(
        answer.trim().to_ascii_lowercase().as_str(),
        "y" | "yes"
    ))
}

fn download(expected: &RuntimePackManifest) -> Result<Vec<u8>> {
    let url = format!(
        "https://github.com/runmat-org/runmat/releases/download/v{}/{}",
        expected.runtime.runmat_version,
        expected.file_name()?,
    );
    let response = ureq::AgentBuilder::new()
        .https_only(true)
        .timeout_connect(Duration::from_secs(30))
        .timeout_read(Duration::from_secs(120))
        .timeout(Duration::from_secs(1800))
        .build()
        .get(&url)
        .call()
        .with_context(|| format!("download compile-time runtime from {url}"))?;
    read_bounded(response.into_reader(), expected.pack_bytes)
}

fn read_bounded(reader: impl Read, expected_bytes: u64) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    reader
        .take(
            expected_bytes
                .checked_add(1)
                .context("runtime pack size overflow")?,
        )
        .read_to_end(&mut bytes)?;
    anyhow::ensure!(
        u64::try_from(bytes.len())? == expected_bytes,
        "runtime pack length differs from the trusted manifest"
    );
    Ok(bytes)
}

struct PackCache {
    root: PathBuf,
}

impl PackCache {
    fn path(&self, expected: &RuntimePackManifest) -> PathBuf {
        let digest = expected.pack_digest.to_string();
        self.root
            .join(format!("{}.zip", digest.trim_start_matches("sha256:")))
    }

    fn load(&self, expected: &RuntimePackManifest) -> Result<Option<RuntimeArchive>> {
        expected.validate()?;
        let path = self.path(expected);
        let file = match std::fs::File::open(&path) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("open cached runtime `{}`", path.display()))
            }
        };
        let bytes = read_bounded(file, expected.pack_bytes)?;
        Ok(Some(
            decode_runtime_pack(expected, &bytes)
                .context("cached AOT runtime failed verification")?,
        ))
    }

    fn install(
        &self,
        expected: &RuntimePackManifest,
        acquire: impl FnOnce() -> Result<Vec<u8>>,
    ) -> Result<RuntimeArchive> {
        expected.validate()?;
        let _lock = ProcessLock::acquire(&self.root.join("install.lock"))?;
        if let Some(runtime) = self.load(expected)? {
            return Ok(runtime);
        }
        let bytes = acquire()?;
        let runtime = decode_runtime_pack(expected, &bytes)?;
        atomic_write(&self.path(expected), &bytes)?;
        // Cache cleanup is best effort and cannot undo a successful install.
        if let Err(error) = self.prune(&self.path(expected)) {
            eprintln!("Could not remove obsolete AOT runtime packs: {error}");
        }
        Ok(runtime)
    }

    fn prune(&self, current: &Path) -> Result<()> {
        let mut previous = Vec::new();
        for entry in std::fs::read_dir(&self.root)? {
            let entry = entry?;
            let name = entry.file_name();
            let Some(digest) = name.to_str().and_then(|name| name.strip_suffix(".zip")) else {
                continue;
            };
            if digest.len() != 64
                || !digest
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            {
                continue;
            }
            let path = entry.path();
            if path != current && entry.file_type()?.is_file() {
                previous.push((entry.metadata()?.modified()?, path));
            }
        }
        previous.sort_by(|a, b| b.cmp(a));
        for (_, path) in previous.into_iter().skip(1) {
            std::fs::remove_file(path)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consent_requires_an_explicit_yes_and_consumes_only_one_line() {
        for answer in ["", "\n", "n\n", "no\n", "later\n", "yes please\n"] {
            assert!(!confirm_install(&mut io::Cursor::new(answer), &mut Vec::new(), 42).unwrap());
        }
        let mut input = io::Cursor::new("YES\nnext line\n");
        assert!(confirm_install(&mut input, &mut Vec::new(), 42).unwrap());
        assert_eq!(input.position(), 4);
        let long = format!("yes{}no\n", " ".repeat(64));
        assert!(!confirm_install(&mut io::Cursor::new(long), &mut Vec::new(), 42).unwrap());
    }

    #[test]
    fn pack_reads_refuse_short_and_oversize_payloads() {
        assert_eq!(read_bounded(io::Cursor::new(b"abc"), 3).unwrap(), b"abc");
        assert!(read_bounded(io::Cursor::new(b"ab"), 3).is_err());
        assert!(read_bounded(io::Cursor::new(b"abcd"), 3).is_err());
    }

    #[test]
    fn retention_keeps_current_and_one_previous_pack_and_leaves_other_files() {
        let directory = tempfile::tempdir().unwrap();
        let cache = PackCache {
            root: directory.path().to_path_buf(),
        };
        let current = cache.root.join(format!("{}.zip", "a".repeat(64)));
        let previous =
            ["b", "c", "d"].map(|name| cache.root.join(format!("{}.zip", name.repeat(64))));
        for path in std::iter::once(&current).chain(previous.iter()) {
            std::fs::write(path, b"cache fixture").unwrap();
        }
        let unrelated = cache.root.join("keep.zip");
        std::fs::write(&unrelated, b"unrelated").unwrap();
        cache.prune(&current).unwrap();
        assert!(current.exists());
        assert_eq!(previous.iter().filter(|path| path.exists()).count(), 1);
        assert!(unrelated.exists());
    }
}
