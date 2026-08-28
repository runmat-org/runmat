#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_source;

#[test]
fn compiled_label_apis_preserve_adjacent_wide_uint64_values() {
    execute_source(
        "base=bitshift(uint64(1),53); lo=base+uint64(1); hi=base+uint64(2); A=uint64([lo hi lo]); E=onehotencode(A,1,'uint64','ClassNames',uint64([lo hi])); if ~isa(E,'uint64') || E(1,1)~=uint64(1) || E(2,2)~=uint64(1); error('onehotencode exactness'); end; D=onehotdecode(double(E),uint64([lo hi]),1,'uint64'); if ~isa(D,'uint64') || D(1)~=lo || D(2)~=hi || D(3)~=lo; error('onehotdecode exactness'); end; O=ordinal(uint64([hi lo hi]),{'lower','upper'},uint64([lo hi])); if ~isordinal(O); error('ordinal metadata'); end; s=num2str(hi,'%.0f'); if ~strcmp(s,'9007199254740994'); error('num2str exactness'); end;",
    )
    .expect("compiled exact label semantics");
}

#[test]
fn compiled_text_controls_accept_every_integer_class() {
    for constructor in [
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    ] {
        let source = format!(
            "a=regexprep('a a a','a','X',{constructor}(2)); if ~strcmp(a,'a X a'); error('regexprep control'); end; b=replaceBetween('abcdef',{constructor}(2),{constructor}(4),'X'); if ~strcmp(b,'aXef'); error('replaceBetween control'); end; d=tokenizedDocument('a medium enormous'); d=removeShortWords(d,{constructor}(1)); d=removeLongWords(d,{constructor}(9)); d=removeWords(d,{constructor}(1));"
        );
        execute_source(&source).expect("compiled integer text controls");
    }
}

#[test]
fn compiled_ordinal_edges_do_not_collapse_wide_uint64_boundaries() {
    execute_source(
        "base=bitshift(uint64(1),53); a=base+uint64(1); b=a+uint64(1); c=b+uint64(1); O=ordinal(uint64([a b c]),{'first','second'},[],uint64([a b c])); if ~isordinal(O); error('ordinal edge metadata'); end;",
    )
    .expect("compiled exact ordinal edges");
}
