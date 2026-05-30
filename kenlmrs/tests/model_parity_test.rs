/// Integration tests verifying probability parity with C++ KenLM.
///
/// Expected values are taken directly from `lm/model_test.cc` in the C++ KenLM repository.
/// All probabilities are in log10. Tolerance: ±0.001 (matches C++ `SLOPPY_CHECK_CLOSE`).
use kenlmrs::model::ProbingModel;
use kenlmrs::types::{Config, State};
use kenlmrs::vocabulary::Vocabulary;

const TEST_ARPA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/test.arpa");
const TOL: f32 = 0.001;

fn assert_prob_close(expected: f32, actual: f32, context: &str) {
    let diff = (expected - actual).abs();
    assert!(
        diff < TOL,
        "{context}: expected {expected:.6}, got {actual:.6}, diff {diff:.6} > tol {TOL}"
    );
}

fn load_model() -> ProbingModel {
    ProbingModel::new(TEST_ARPA, &Config::default())
        .unwrap_or_else(|e| panic!("Failed to load {TEST_ARPA}: {e}"))
}

// ── Starters: score first word given <s> ──────────────────────────────────────

#[test]
fn starters_looking_after_bos_should_be_bigram() {
    let model = load_model();
    let vocab = model.vocab();
    let state = model.begin_sentence_state();
    let mut out = State::default();

    let idx = vocab.index("looking");
    let ret = model.full_score(&state, idx, &mut out);

    assert_eq!(ret.ngram_length, 2, "looking after <s> should be a bigram");
    assert_prob_close(-0.4846522, ret.prob, "P(looking|<s>)");
    assert!(ret.independent_left, "looking after <s> has no longer left context");
}

#[test]
fn starters_comma_after_bos_uses_unigram_plus_backoff() {
    let model = load_model();
    let vocab = model.vocab();
    let state = model.begin_sentence_state();
    let mut out = State::default();

    let idx = vocab.index(",");
    let ret = model.full_score(&state, idx, &mut out);

    // Unigram prob + <s> backoff: -1.383514 + -0.4149733
    assert_eq!(ret.ngram_length, 1);
    assert_prob_close(-1.383514 + -0.4149733, ret.prob, "P(,|<s>)");
}

#[test]
fn starters_unknown_word_after_bos_uses_unk_plus_backoff() {
    let model = load_model();
    let vocab = model.vocab();
    let state = model.begin_sentence_state();
    let mut out = State::default();

    let idx = vocab.index("this_is_not_found");
    let ret = model.full_score(&state, idx, &mut out);

    // <unk> prob + <s> backoff: -1.995635 + -0.4149733
    assert_eq!(ret.ngram_length, 1);
    assert_prob_close(-1.995635 + -0.4149733, ret.prob, "P(<unk>|<s>)");
}

// ── Continuation: full sentence from <s> ─────────────────────────────────────

#[test]
fn continuation_sentence_matches_cpp_model_test() {
    let model = load_model();
    let vocab = model.vocab();

    struct Expected<'a> {
        word: &'a str,
        ngram: u8,
        prob: f32,
    }

    let cases = [
        Expected { word: "looking",  ngram: 2, prob: -0.484652 },
        Expected { word: "on",       ngram: 3, prob: -0.348837 },
        Expected { word: "a",        ngram: 4, prob: -0.0155266 },
        Expected { word: "little",   ngram: 5, prob: -0.00306122 },
    ];

    let mut state = model.begin_sentence_state();
    for c in &cases {
        let mut out = State::default();
        let ret = model.full_score(&state, vocab.index(c.word), &mut out);
        assert_eq!(ret.ngram_length, c.ngram, "ngram length for '{}'", c.word);
        assert_prob_close(c.prob, ret.prob, &format!("P({}|context)", c.word));
        state = out;
    }
}

// ── Blanks: null-context (no sentence marker) scoring ────────────────────────

#[test]
fn blanks_also_is_unigram_not_independent_left() {
    let model = load_model();
    let vocab = model.vocab();
    let state = ProbingModel::empty_context_state();
    let mut out = State::default();

    let ret = model.full_score(&state, vocab.index("also"), &mut out);

    assert_eq!(ret.ngram_length, 1);
    assert_prob_close(-1.687872, ret.prob, "P(also|null)");
    // "also" appears as context for higher-order n-grams → NOT independent
    assert!(!ret.independent_left, "also has higher-order children");
}

#[test]
fn blanks_also_would_consider_higher_looking_is_5gram() {
    let model = load_model();
    let vocab = model.vocab();
    let mut state = ProbingModel::empty_context_state();

    let sequence = [
        ("also",      1u8, -1.687872_f32),
        ("would",     2,   -2.0),
        ("consider",  3,   -3.0),
        ("higher",    4,   -4.0),
        ("looking",   5,   -5.0),
    ];

    for (word, ngram, expected_prob) in &sequence {
        let mut out = State::default();
        let ret = model.full_score(&state, vocab.index(word), &mut out);
        assert_eq!(ret.ngram_length, *ngram, "ngram length for '{word}'");
        assert_prob_close(*expected_prob, ret.prob, &format!("P({word}|context)"));
        state = out;
    }

    // After the 5-gram, state should be trimmed to length 1
    assert_eq!(state.length, 1, "state length after 5-gram sequence");
}

// ── MinimalState: backoff sentinel values (-0.0 vs +0.0) ─────────────────────

#[test]
fn minimal_state_baz_has_no_extension_so_state_length_is_zero() {
    let model = load_model();
    let vocab = model.vocab();
    let state = ProbingModel::empty_context_state();
    let mut out = State::default();

    let ret = model.full_score(&state, vocab.index("baz"), &mut out);

    assert_prob_close(-6.535897, ret.prob, "P(baz|null)");
    // baz has backoff = -0.0 (no extension), so out.length must be 0
    assert_eq!(out.length, 0, "baz backoff is -0.0, state must not extend");
}

#[test]
fn minimal_state_foo_has_extension_so_state_length_is_one() {
    let model = load_model();
    let vocab = model.vocab();
    let state = ProbingModel::empty_context_state();
    let mut out = State::default();

    let ret = model.full_score(&state, vocab.index("foo"), &mut out);

    assert_prob_close(-3.141592, ret.prob, "P(foo|null)");
    // foo has no backoff listed (backoff = kNoExtensionBackoff initially),
    // but appears as context for "foo bar" bigram, so state.length = 1
    assert_eq!(out.length, 1, "foo appears as bigram context, state.length = 1");
}

#[test]
fn minimal_state_foo_bar_uses_bigram_with_backoff_adjustment() {
    let model = load_model();
    let vocab = model.vocab();
    let mut state = ProbingModel::empty_context_state();

    // score "foo" first
    let mut out = State::default();
    model.full_score(&state, vocab.index("foo"), &mut out);
    state = out;

    // score "bar" given "foo" context — bigram hit + foo's backoff 3.0
    let mut out2 = State::default();
    let ret = model.full_score(&state, vocab.index("bar"), &mut out2);

    assert_eq!(ret.ngram_length, 2, "foo bar is a bigram");
    // C++ model_test: -6.0 total (foo bar bigram = -6, and state has foo backoff 3.0)
    assert_prob_close(-6.0, ret.prob, "P(bar|foo)");
    assert_eq!(out2.length, 1);
}

// ── Unknowns ──────────────────────────────────────────────────────────────────

#[test]
fn unknowns_unk_unk_bigram_scores_minus_15() {
    let model = load_model();
    let vocab = model.vocab();
    let mut state = ProbingModel::empty_context_state();

    // First <unk>
    let unk = vocab.index("not_found");
    let mut out = State::default();
    model.full_score(&state, unk, &mut out);
    state = out;

    // Second <unk> given first <unk>
    let mut out2 = State::default();
    let ret = model.full_score(&state, vocab.index("not_found2"), &mut out2);

    assert_eq!(ret.ngram_length, 2, "<unk><unk> is a bigram");
    assert_prob_close(-15.0, ret.prob, "P(<unk>|<unk>)");
}

// ── Vocabulary ────────────────────────────────────────────────────────────────

#[test]
fn vocabulary_special_words_have_canonical_indices() {
    let model = load_model();
    let vocab = model.vocab();

    assert_eq!(vocab.not_found(), 0, "<unk> must be index 0");
    assert_eq!(vocab.begin_sentence(), 1, "<s> must be index 1");
    assert_eq!(vocab.end_sentence(), 2, "</s> must be index 2");
}

#[test]
fn vocabulary_known_words_are_found() {
    let model = load_model();
    let vocab = model.vocab();

    assert_ne!(vocab.index("looking"), vocab.not_found());
    assert_ne!(vocab.index("on"), vocab.not_found());
    assert_ne!(vocab.index("a"), vocab.not_found());
}

#[test]
fn vocabulary_unknown_words_map_to_not_found() {
    let model = load_model();
    let vocab = model.vocab();

    assert_eq!(vocab.index("xyzzy_not_in_model"), vocab.not_found());
}
