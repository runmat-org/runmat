use tokio::process::Child;

pub(super) async fn hard_terminate(
    child: &mut Child,
    process_id: Option<u32>,
) -> std::io::Result<()> {
    super::process_tree::kill(child, process_id).await
}
