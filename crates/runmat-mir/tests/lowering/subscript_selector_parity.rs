use super::*;

#[test]
fn noncallable_dotted_selectors_share_ordinary_index_facts() {
    for (ordinary, dotted) in [
        (
            "function y = f(); a = [1, 2]; y = a(2); end",
            "function y = f(); s = struct{a = [1, 2]}; y = s.a(2); end",
        ),
        (
            "function y = f(i); a = [1, 2]; y = a(i); end",
            "function y = f(i); s = struct{a = [1, 2]}; y = s.a(i); end",
        ),
        (
            "function y = f(); a = [1, 2]; y = a(end); end",
            "function y = f(); s = struct{a = [1, 2]}; y = s.a(end); end",
        ),
        (
            "function y = f(); a = [1, 2]; y = a([1, 2]); end",
            "function y = f(); s = struct{a = [1, 2]}; y = s.a([1, 2]); end",
        ),
    ] {
        let (ordinary_body, ordinary_store) = analyze_single_body(ordinary);
        let (dotted_body, dotted_store) = analyze_single_body(dotted);
        let ordinary_fact = output_fact(&ordinary_body, &ordinary_store);
        let dotted_fact = output_fact(&dotted_body, &dotted_store);
        assert_eq!(
            dotted_fact.kind, ordinary_fact.kind,
            "dotted selector kind diverged for {dotted}"
        );
        assert_eq!(
            dotted_fact.shape, ordinary_fact.shape,
            "dotted selector shape diverged for {dotted}"
        );
    }
}
