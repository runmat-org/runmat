//! Native WGPU 24 Plot smoke gate.
//! Run on a desktop main thread with a real adapter and display:
//! `cargo run -p runmat-plot --example native_gui_smoke --features gui`
//! A successful presentation is a submission/lifecycle witness, not GPU
//! completion or display timing; the separate PNG readback checks real pixels.
//! Pass `--cli /absolute/path/to/runmat` to qualify the actual CLI/Core/native
//! host path instead; it must have been built with its `gui` feature.

#[cfg(not(target_arch = "wasm32"))]
#[path = "native_gui_smoke/cli.rs"]
mod cli;

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args_os().skip(1);
    if let Some(option) = arguments.next() {
        if option != "--cli" {
            return Err(std::io::Error::other("expected --cli /absolute/path/to/runmat").into());
        }
        let executable = arguments
            .next()
            .ok_or_else(|| std::io::Error::other("missing CLI executable"))?;
        if arguments.next().is_some() {
            return Err(std::io::Error::other("unexpected smoke argument").into());
        }
        return cli::run(std::path::Path::new(&executable));
    }
    use runmat_plot::export::native_surface::render_figure_png_bytes_interactive;
    use runmat_plot::gui::{register_main_thread, PlotWindow, WindowConfig};
    use runmat_plot::plots::{Figure, LinePlot};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    register_main_thread();
    let mut figure = Figure::new();
    let x: Vec<f64> = (0..64).map(|index| f64::from(index) / 8.0).collect();
    let y: Vec<f64> = x.iter().map(|value| value.sin()).collect();
    figure.add_line_plot(LinePlot::new(x, y).map_err(std::io::Error::other)?);

    let (width, height) = (640, 480);
    let png = pollster::block_on(render_figure_png_bytes_interactive(
        figure.clone(),
        width,
        height,
    ))
    .map_err(std::io::Error::other)?;
    let pixels = image::load_from_memory_with_format(&png, image::ImageFormat::Png)?.to_rgba8();
    if pixels.dimensions() != (width, height) {
        return Err(std::io::Error::other("PNG export has the wrong dimensions").into());
    }
    let background = *pixels.get_pixel(0, 0);
    let varied_pixels = pixels
        .pixels()
        .filter(|pixel| **pixel != background)
        .count();
    if varied_pixels == 0 {
        return Err(std::io::Error::other("PNG export produced uniform pixels").into());
    }

    let config = WindowConfig {
        title: "RunMat native WGPU 24 smoke".to_owned(),
        width,
        height,
        ..WindowConfig::default()
    };
    let mut window = pollster::block_on(PlotWindow::new(config))?;
    window.set_figure(figure)?;
    window.window.request_redraw();

    let close = Arc::new(AtomicBool::new(false));
    window.install_close_signal(close.clone());
    let presentations = window.presentation_count();
    let observed = presentations.clone();
    let watcher_close = close.clone();
    let watchdog = std::thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !watcher_close.load(Ordering::Acquire) {
            if observed.get() > 0 || Instant::now() >= deadline {
                watcher_close.store(true, Ordering::Release);
                break;
            }
            std::thread::sleep(Duration::from_millis(16));
        }
    });

    let run_result = pollster::block_on(window.run());
    close.store(true, Ordering::Release);
    watchdog
        .join()
        .map_err(|_| std::io::Error::other("GUI close watchdog panicked"))?;
    run_result?;
    if presentations.get() == 0 {
        return Err(std::io::Error::other("GUI exited without presenting a frame").into());
    }
    println!(
        "native GUI presented {} frame(s); PNG export returned {} varied pixels",
        presentations.get(),
        varied_pixels
    );
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn main() {
    eprintln!("native_gui_smoke requires a desktop main thread");
}
