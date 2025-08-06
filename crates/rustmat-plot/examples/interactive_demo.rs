//! Interactive plotting demo
//!
//! Demonstrates the new interactive GUI plotting capabilities of RustMat Plot.
//!
//! Run with: cargo run --example interactive_demo --features gui

#[cfg(feature = "gui")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Enable logging for debugging
    env_logger::init();

    println!("🚀 RustMat Plot - Interactive Demo");
    println!("Starting interactive plot window...");

    // Launch interactive window with default configuration
    rustmat_plot::show_interactive().await?;

    Ok(())
}

#[cfg(not(feature = "gui"))]
fn main() {
    eprintln!("This example requires the 'gui' feature to be enabled.");
    eprintln!("Run with: cargo run --example interactive_demo --features gui");
    std::process::exit(1);
}
