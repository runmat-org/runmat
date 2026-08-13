pub(crate) const WAVE_SOURCE: &str = r#"x = linspace(-8, 8, 120);
y = linspace(-8, 8, 120);
[X, Y] = meshgrid(x, y);
R = sqrt(X.^2 + Y.^2);

amplitude = 1.0;
wavelength = 4.0;
speed = 1.5;
k = 2*pi / wavelength;
omega = 2*pi*speed / wavelength;

for t = 0:1/30:8
    Z = amplitude * sin(k * R - omega * t) .* exp(-0.08 * R);
    surf(X, Y, Z);
end
"#;

pub(crate) const WAVE_EXPECTATIONS: [(&str, &[usize]); 12] = [
    ("x =", &[1, 120]),
    ("y =", &[1, 120]),
    ("X, Y", &[120, 120]),
    ("Y]", &[120, 120]),
    ("R =", &[120, 120]),
    ("Z =", &[120, 120]),
    ("amplitude =", &[1, 1]),
    ("wavelength =", &[1, 1]),
    ("speed =", &[1, 1]),
    ("k =", &[1, 1]),
    ("omega =", &[1, 1]),
    ("t =", &[1, 1]),
];
