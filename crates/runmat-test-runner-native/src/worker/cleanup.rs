use tokio::process::Child;

pub(super) async fn reap(child: &mut Child) -> std::io::Result<()> {
    let _ = child.wait().await?;
    Ok(())
}
