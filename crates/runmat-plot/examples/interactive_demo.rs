//! Interactive plotting demo
//!
//! Demonstrates the new interactive GUI plotting capabilities of RunMat Plot.
//!
//! Run with: cargo run --example interactive_demo --features gui

#[cfg(feature = "gui")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Enable logging for debugging
    env_logger::init();

    println!("🚀 RunMat Plot - Interactive Demo");
    println!("Starting interactive plot window...");

    // Launch interactive window with default configuration
    let figure = runmat_plot::plots::Figure::new()
        .with_title("Interactive Demo")
        .with_labels("X", "Y")
        .with_grid(true);

    let result = runmat_plot::show_interactive_platform_optimal(figure)?;
    println!("Interactive plot result: {result}");

    Ok(())
}

#[cfg(not(feature = "gui"))]
fn main() {
    eprintln!("This example requires the 'gui' feature to be enabled.");
    eprintln!("Run with: cargo run --example interactive_demo --features gui");
    std::process::exit(1);
}
