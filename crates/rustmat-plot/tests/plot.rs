use rustmat_plot::{
    load_config, plot_3d_scatter, plot_bar, plot_histogram, plot_line, plot_scatter, plot_surface,
    PlotConfig, CONFIG_ENV,
};
use serial_test::serial;
use std::env;
use std::fs;
use std::fs::File;
use std::io::Write;
use tempfile::tempdir;

#[test]
#[serial]
fn default_config_used_when_env_missing() {
    env::remove_var(CONFIG_ENV);
    assert_eq!(PlotConfig::default().width, 800);
}

#[test]
#[serial]
fn load_config_from_file() {
    let dir = tempdir().unwrap();
    let cfg_path = dir.path().join("cfg.yml");
    let mut f = File::create(&cfg_path).unwrap();
    writeln!(
        f,
        "width: 200\nheight: 100\nline_color: '#ff0000'\nline_width: 3\nscatter_color: '#00ff00'\nbar_color: '#00ff00'\nhist_color: '#00ff00'\nmarker_size: 2\nbackground: '#00ff00'"
    )
    .unwrap();
    let cfg = load_config(&cfg_path).unwrap();
    assert_eq!(cfg.width, 200);
    assert_eq!(cfg.background, "#00ff00");
}

#[test]
#[serial]
fn plot_svg_file() {
    env::remove_var(CONFIG_ENV);
    let dir = tempdir().unwrap();
    let path = dir.path().join("out.svg");
    let xs = [0.0, 1.0];
    let ys = [0.0, 1.0];
    plot_line(&xs, &ys, path.to_str().unwrap()).unwrap();
    let data = fs::read_to_string(&path).unwrap();
    assert!(data.trim_start().starts_with("<svg"));
}

#[test]
#[serial]
fn error_on_bad_color() {
    let dir = tempdir().unwrap();
    let cfg_path = dir.path().join("cfg.yml");
    fs::write(&cfg_path, "line_color: 'bad'").unwrap();
    env::set_var(CONFIG_ENV, &cfg_path);
    let xs = [0.0];
    let ys = [0.0];
    let result = plot_line(&xs, &ys, dir.path().join("a.svg").to_str().unwrap());
    assert!(result.is_err());
    env::remove_var(CONFIG_ENV);
}

#[test]
#[serial]
fn scatter_and_bar_and_hist() {
    env::remove_var(CONFIG_ENV);
    let dir = tempdir().unwrap();
    let scatter = dir.path().join("scatter.svg");
    let bar = dir.path().join("bar.svg");
    let hist = dir.path().join("hist.svg");
    let xs = [0.0, 1.0, 2.0];
    let ys = [1.0, 0.5, 1.5];
    plot_scatter(&xs, &ys, scatter.to_str().unwrap()).unwrap();
    plot_bar(&["a", "b", "c"], &ys, bar.to_str().unwrap()).unwrap();
    plot_histogram(&ys, 2, hist.to_str().unwrap()).unwrap();
    assert!(fs::metadata(scatter).unwrap().len() > 0);
    assert!(fs::metadata(bar).unwrap().len() > 0);
    assert!(fs::metadata(hist).unwrap().len() > 0);
}

#[test]
#[serial]
fn line_length_mismatch_errors() {
    let xs = [0.0];
    let ys = [0.0, 1.0];
    let err = plot_line(&xs, &ys, "out.svg");
    assert!(err.is_err());
}

#[test]
#[serial]
fn histogram_zero_bins_errors() {
    let values = [0.0];
    let err = plot_histogram(&values, 0, "out.svg");
    assert!(err.is_err());
}

#[test]
#[serial]
fn unsupported_extension_errors() {
    let xs = [0.0];
    let ys = [0.0];
    let err = plot_line(&xs, &ys, "foo.txt");
    assert!(err.is_err());
}

#[test]
#[serial]
fn bar_length_mismatch_errors() {
    let labels = ["a", "b"];
    let values = [1.0];
    let err = plot_bar(&labels, &values, "bar.svg");
    assert!(err.is_err());
}

#[test]
#[serial]
fn scatter_unsupported_extension_errors() {
    let xs = [0.0];
    let ys = [0.0];
    let err = plot_scatter(&xs, &ys, "out.pdf");
    assert!(err.is_err());
}

#[test]
#[serial]
fn scatter3d_and_surface_render() {
    env::remove_var(CONFIG_ENV);
    let dir = tempdir().unwrap();
    let scatter = dir.path().join("s3d.svg");
    let surface = dir.path().join("surf.svg");
    let xs = [0.0, 1.0, 0.0, 1.0];
    let ys = [0.0, 0.0, 1.0, 1.0];
    let zs = [0.0, 1.0, 1.0, 0.0];
    plot_3d_scatter(&xs, &ys, &zs, scatter.to_str().unwrap()).unwrap();
    plot_surface(&xs, &ys, &zs, surface.to_str().unwrap()).unwrap();
    assert!(fs::metadata(scatter).unwrap().len() > 0);
    assert!(fs::metadata(surface).unwrap().len() > 0);
}

#[test]
#[serial]
fn scatter3d_length_mismatch_errors() {
    let xs = [0.0];
    let ys = [0.0, 1.0];
    let zs = [0.0];
    assert!(plot_3d_scatter(&xs, &ys, &zs, "out.svg").is_err());
}

#[test]
#[serial]
fn surface_non_square_errors() {
    let xs = [0.0, 1.0, 0.0];
    let ys = [0.0, 0.0, 1.0];
    let zs = [0.0, 1.0, 1.0];
    assert!(plot_surface(&xs, &ys, &zs, "out.svg").is_err());
}
