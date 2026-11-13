//! Keep completed-link authority alive for independent operator review.

use std::{path::Path, time::Duration};

use anyhow::{ensure, Context, Result};

pub async fn wait(application: &Path, tests: &Path) -> Result<()> {
    ensure!(
        !application.exists() && !tests.exists(),
        "review mode requires new operator intent paths, not prior image grants"
    );
    println!("Native execution receipt is ready for independent installation review. Publish intents atomically at {} and {}", application.display(), tests.display());
    let deadline = tokio::time::Instant::now() + Duration::from_secs(600);
    loop {
        if application
            .try_exists()
            .context("inspect application review intent")?
            && tests.try_exists().context("inspect test review intent")?
        {
            return Ok(());
        }
        ensure!(
            tokio::time::Instant::now() < deadline,
            "independent installation review timed out; no admission was issued"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}
