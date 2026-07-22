#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_source;

#[test]
fn binary_decimal_conversion_builtins_execute_through_vm() {
    let input = "\
        d1 = bi2de([1 0 1 0]); \
        if d1 ~= 5; error('bi2de right-msb row failed'); end; \
        d2 = bi2de([1 0 1 0], 'left-msb'); \
        if d2 ~= 10; error('bi2de left-msb row failed'); end; \
        B = [1 0 1; 0 1 1]; \
        D = bi2de(B); \
        if D(1) ~= 5 || D(2) ~= 6; error('bi2de row-wise matrix failed'); end; \
        T = bi2de([2 1 0], 3, 'left-msb'); \
        if T ~= 21; error('bi2de base-3 left-msb failed'); end; \
        bits = de2bi([0; 1; 2; 5]); \
        sz = size(bits); \
        if sz(1) ~= 4 || sz(2) ~= 3; error('de2bi minimum-width shape failed'); end; \
        if bits(4,1) ~= 1 || bits(4,2) ~= 0 || bits(4,3) ~= 1; \
            error('de2bi right-msb digits failed'); \
        end; \
        tri = de2bi(21, 4, 3, 'left-msb'); \
        if tri(1) ~= 0 || tri(2) ~= 2 || tri(3) ~= 1 || tri(4) ~= 0; \
            error('de2bi base-3 left-msb digits failed'); \
        end; \
        tri_auto = de2bi([5; 21], [], 3, 'left-msb'); \
        tri_sz = size(tri_auto); \
        if tri_sz(1) ~= 2 || tri_sz(2) ~= 3; error('de2bi empty-width base shape failed'); end; \
        if tri_auto(1,1) ~= 0 || tri_auto(1,2) ~= 1 || tri_auto(1,3) ~= 2; \
            error('de2bi empty-width first row failed'); \
        end; \
        if tri_auto(2,1) ~= 2 || tri_auto(2,2) ~= 1 || tri_auto(2,3) ~= 0; \
            error('de2bi empty-width second row failed'); \
        end; \
        roundtrip = bi2de(de2bi([0; 1; 2; 7], 3), 2); \
        if roundtrip(1) ~= 0 || roundtrip(2) ~= 1 || roundtrip(3) ~= 2 || roundtrip(4) ~= 7; \
            error('bi2de/de2bi roundtrip failed'); \
        end;";
    execute_source(input).expect("execute comms binary conversion script");
}
