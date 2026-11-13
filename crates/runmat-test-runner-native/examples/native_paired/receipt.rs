//! Read-only validation through the same receipt owner used by execution.

use std::{fs::File, io::Read, path::Path};

use anyhow::{ensure, Result};
use runmat_execution::Digest;
use runmat_test_runner_native::target_test::NativePairedQualificationReceipt;

pub(super) fn validate(path: &Path) -> Result<()> {
    const MAXIMUM_BYTES: u64 = 64 * 1024 * 1024;
    let mut bytes = Vec::new();
    File::open(path)?
        .take(MAXIMUM_BYTES + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= MAXIMUM_BYTES,
        "qualification receipt exceeds its wire bound"
    );
    NativePairedQualificationReceipt::from_canonical_bytes(&bytes)?;
    // This digest identifies the exact bytes validated, not execution rights.
    println!("{}", Digest::sha256(&bytes));
    Ok(())
}
