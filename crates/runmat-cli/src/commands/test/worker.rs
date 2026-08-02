use anyhow::Result;

pub async fn run_stdio() -> Result<()> {
    runmat_test_runner_native::run_core_worker_stdio().await?;
    Ok(())
}
