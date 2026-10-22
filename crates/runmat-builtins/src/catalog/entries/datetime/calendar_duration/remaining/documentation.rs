use super::*;
const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Datetime implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/datetime/operators.rs"),
};
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[SOURCE],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest,
        label: "Canonical residual datetime calls", location: "crates/runmat-runtime/src/builtins/datetime/operators/remaining/tests.rs" }],
    notes: &["Calendar scratch is not yet completely finite-owned: these calendar methods and datetick fail closed under Managed allocation before existing scratch or provider work. No complete allocation capability is asserted."],
};
macro_rules! doc {
    ($id:ident,$name:literal,$summary:literal,$description:literal,$program:literal,$oracle:literal,$compat:ident) => {
        pub(super) const $id: BuiltinDocumentation = BuiltinDocumentation {
            authority: BuiltinDocumentationAuthority::Catalog,title:Some($name),slug:Some($name),
            summary:$summary,description:$description,
            keywords:&["datetime","calendarDuration",$name],
            related:&["datetime","calendarDuration","calmonths","caldays"],
            sections:&[BuiltinDocumentationSection { heading:"Runtime behavior", paragraphs:&[
                "Outputs are host-resident. Method calls accept exactly two inputs and zero or one requested output; isdatetime accepts one input; datetick accepts any number. Excess outputs are rejected before scratch, provider calls, or output publication.",
                "Calendar methods use existing binary64 month/day components. Arrays require identical shapes unless an operand has one element. No singleton-dimension broadcasting is added. Resident operands are not downloaded by calendar methods; isdatetime checks class without reading payloads. Datetick retains compatibility downloads but does not change any axes."
            ]}],
            examples:&[BuiltinExample { id:"behavior",title:$summary,program:$program,display_output:None,
                compatibility:BuiltinExampleCompatibility::$compat,harness:BuiltinExampleHarness::Portable,
                fixture:BuiltinExampleFixture::None,requirements:BuiltinExampleRequirements::NONE,
                verification:BuiltinExampleVerification::Assertions { source:$oracle } }],
            example_exemption:None,faqs:&[],links:&[SOURCE],media:&[],evidence:EVIDENCE,
            introduced:None,status:Some(BuiltinDocumentationStatus::Partial),
        };
    };
}
doc!(ISDATETIME,"isdatetime","Test whether a value is a datetime object.",
    "Returns one logical scalar for the entire input by exact runtime class identity. Numeric, text, unrelated objects, and resident numeric inputs return false. A datetime object containing an empty or nonscalar serial array still returns true. This predicate performs no allocation workspace admission or provider access.",
    "d = datetime(2024,1,1); yes = isdatetime(d); no = isdatetime(1);",
    "assert(islogical(yes)); assert(yes); assert(~no);",Matlab);
doc!(DATETICK,"datetick","Accept legacy date-axis calls without formatting axes.",
    "RunMat compatibility no-op, not a date-axis formatter. All arguments are accepted without option validation. Resident argument payloads are gathered through their existing owners, then discarded. Success returns double zero, suppressed from automatic display; no axes, tick labels, files, or workspace are changed. This deliberately does not claim MATLAB datetick plotting semantics.",
    "status = datetick('x','yyyy','keeplimits');",
    "assert(isa(status,'double')); assert(status == 0);",RunMat);
doc!(PLUS,"calendarDuration.plus","Add calendar components or shift a datetime.",
    "Adds stored month/day components of two calendarDuration values and returns calendarDuration. With a datetime RHS, returns datetime using the RHS Format and existing month-end clamping. Numeric day counts are not accepted by this receiver method.",
    "c = calmonths(1) + calmonths(2); n = calmonths(c); d = calmonths(1) + datetime(2024,1,31);",
    "assert(n == 3); assert(isdatetime(d)); assert(day(d) == 29);",Matlab);
doc!(MINUS,"calendarDuration.minus","Subtract stored calendar components.",
    "Subtracts month and day components separately and returns calendarDuration. Both operands must be calendarDuration. Datetime and numeric RHS operands are rejected.",
    "c = calmonths(3) - calmonths(1); n = calmonths(c);",
    "assert(n == 2); assert(iscalendarduration(c));",Matlab);
doc!(EQ,"calendarDuration.eq","Compare calendar components for equality.",
    "RunMat comparison returns host double zero/one rather than logical. Corresponding month and day components must either be exactly equal or differ by at most 1e-12. This compares stored components, not equivalent elapsed times anchored to a date. NaN components compare unequal; equal-sign infinities compare equal. Use explicit logical conversion when needed.",
    "tf = calmonths(2) == calmonths(2);",
    "assert(isa(tf,'double')); assert(tf == 1);",RunMat);
doc!(NE,"calendarDuration.ne","Compare calendar components for inequality.",
    "Returns the double zero/one complement of calendarDuration equality, retaining its component tolerance and NaN/infinity behavior. It is not a logical output and does not compare date-anchored elapsed duration.",
    "tf = calmonths(2) ~= calmonths(3);",
    "assert(isa(tf,'double')); assert(tf == 1);",RunMat);
