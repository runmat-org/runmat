// Request-bound Threefry4x64_20. Each 64-bit lane is (low, high) u32 so
// this kernel does not require an unavailable WGSL u64 shader feature.
struct Tensor { data: array<@DTYPE@>, };
struct Params {
    offset: u32,
    chunk: u32,
    first0: u32,
    first1: u32,
    first2: u32,
    first3: u32,
    seed0: u32,
    seed1: u32,
    integer_lower: u32,
    integer_mask: u32,
    _pad0: u32,
    _pad1: u32,
    @BATCH_FIELDS@
};
struct Pair { a: vec2<u32>, b: vec2<u32>, };
// A value record preserves the four lanes across shader-language backends.
// HLSL has no array-valued function return type.
struct CounterWords { lanes: array<vec2<u32>, 4>, };

@group(0) @binding(0) var<storage, read_write> Out: Tensor;
@group(0) @binding(1) var<uniform> params: Params;

fn add64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    return vec2<u32>(lo, a.y + b.y + select(0u, 1u, lo < a.x));
}

fn rotl64(word: vec2<u32>, rotation: u32) -> vec2<u32> {
    if rotation == 32u { return word.yx; }
    if rotation < 32u {
        return vec2<u32>(
            (word.x << rotation) | (word.y >> (32u - rotation)),
            (word.y << rotation) | (word.x >> (32u - rotation))
        );
    }
    let shift = rotation - 32u;
    return vec2<u32>(
        (word.y << shift) | (word.x >> (32u - shift)),
        (word.x << shift) | (word.y >> (32u - shift))
    );
}

fn mix64(a: vec2<u32>, b: vec2<u32>, rotation: u32) -> Pair {
    let sum = add64(a, b);
    return Pair(sum, rotl64(b, rotation) ^ sum);
}

fn counter_for_item_attempt(index: u32, attempt: vec2<u32>) -> CounterWords {
    let first0 = params.first0 + index;
    let carry0 = select(0u, 1u, first0 < params.first0);
    let first1 = params.first1 + carry0;
    let carry1 = select(0u, 1u, first1 < params.first1);
    let first2 = params.first2 + carry1;
    let carry2 = select(0u, 1u, first2 < params.first2);
    return CounterWords(array<vec2<u32>, 4>(
        vec2<u32>(first0, first1),
        vec2<u32>(first2, params.first3 + carry2),
        attempt,
        vec2<u32>(0u, 0u)
    ));
}

fn counter_for_item(index: u32) -> CounterWords {
    return counter_for_item_attempt(index, vec2<u32>(0u, 0u));
}

fn threefry(counter: CounterWords) -> CounterWords {
    let parity = vec2<u32>(0xA9FC1A22u, 0x1BD11BDAu);
    let keys = array<vec2<u32>, 5>(
        vec2<u32>(params.seed0, params.seed1),
        vec2<u32>(0u, 0u),
        vec2<u32>(0u, 0u),
        vec2<u32>(0u, 0u),
        parity ^ vec2<u32>(params.seed0, params.seed1)
    );
    let rotations = array<vec2<u32>, 8>(
        vec2<u32>(14u, 16u), vec2<u32>(52u, 57u),
        vec2<u32>(23u, 40u), vec2<u32>(5u, 37u),
        vec2<u32>(25u, 33u), vec2<u32>(46u, 12u),
        vec2<u32>(58u, 22u), vec2<u32>(32u, 32u)
    );
    var words = array<vec2<u32>, 4>(
        add64(counter.lanes[0], keys[0]), add64(counter.lanes[1], keys[1]),
        add64(counter.lanes[2], keys[2]), add64(counter.lanes[3], keys[3])
    );
    for (var round: u32 = 0u; round < 20u; round = round + 1u) {
        let rotation = rotations[round % 8u];
        if round % 2u == 0u {
            let left = mix64(words[0], words[1], rotation.x);
            let right = mix64(words[2], words[3], rotation.y);
            words[0] = left.a; words[1] = left.b;
            words[2] = right.a; words[3] = right.b;
        } else {
            let left = mix64(words[0], words[3], rotation.x);
            let right = mix64(words[2], words[1], rotation.y);
            words[0] = left.a; words[3] = left.b;
            words[2] = right.a; words[1] = right.b;
        }
        if round % 4u == 3u {
            let injection = round / 4u + 1u;
            for (var lane: u32 = 0u; lane < 4u; lane = lane + 1u) {
                words[lane] = add64(words[lane], keys[(injection + lane) % 5u]);
            }
            words[3] = add64(words[3], vec2<u32>(injection, 0u));
        }
    }
    return CounterWords(words);
}

@UNIFORM_FUNCTION@

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.chunk { return; }
    let global_idx = params.offset + gid.x;
    let words = threefry(counter_for_item(global_idx)).lanes;
    Out.data[global_idx] = @SAMPLE@;
}
