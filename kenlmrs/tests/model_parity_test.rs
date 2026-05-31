/// Integration tests verifying probability parity with C++ KenLM.
///
/// Expected values are taken directly from `lm/model_test.cc` in the C++ KenLM repository.
/// All probabilities are in log10. Tolerance: ±0.001 (matches C++ `SLOPPY_CHECK_CLOSE`).
use kenlmrs::model::{
    ArrayTrieModel, ProbingModel, QuantArrayTrieModel, QuantTrieModel, TrieModel,
};
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
    assert!(
        ret.independent_left,
        "looking after <s> has no longer left context"
    );
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
        Expected {
            word: "looking",
            ngram: 2,
            prob: -0.484652,
        },
        Expected {
            word: "on",
            ngram: 3,
            prob: -0.348837,
        },
        Expected {
            word: "a",
            ngram: 4,
            prob: -0.0155266,
        },
        Expected {
            word: "little",
            ngram: 5,
            prob: -0.00306122,
        },
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
        ("also", 1u8, -1.687872_f32),
        ("would", 2, -2.0),
        ("consider", 3, -3.0),
        ("higher", 4, -4.0),
        ("looking", 5, -5.0),
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
    assert_eq!(
        out.length, 1,
        "foo appears as bigram context, state.length = 1"
    );
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

// ── Binary format round-trip tests ────────────────────────────────────────────

#[test]
fn probing_binary_round_trip_scores_match() {
    use tempfile::NamedTempFile;
    let original = load_model();

    // Save to a temp binary file
    let tmp = NamedTempFile::new().expect("tempfile");
    let path = tmp.path().to_str().unwrap().to_string();
    original.save(&path).expect("save binary");

    // Reload from binary
    let loaded = ProbingModel::load_binary(&path, &Config::default()).expect("load_binary");

    // Score the same sentences and compare
    let words = ["looking", "on", "a"];
    let o_vocab = original.vocab();
    let l_vocab = loaded.vocab();

    let mut o_state = original.begin_sentence_state();
    let mut l_state = loaded.begin_sentence_state();

    for word in &words {
        let o_idx = o_vocab.index(word);
        let l_idx = l_vocab.index(word);
        let mut o_out = State::default();
        let mut l_out = State::default();
        let o_ret = original.full_score(&o_state, o_idx, &mut o_out);
        let l_ret = loaded.full_score(&l_state, l_idx, &mut l_out);
        assert_prob_close(
            o_ret.prob,
            l_ret.prob,
            &format!(
                "binary round-trip P({word}|ctx): original={:.6} loaded={:.6}",
                o_ret.prob, l_ret.prob
            ),
        );
        o_state = o_out;
        l_state = l_out;
    }
}

#[test]
fn load_virtual_detects_binary_file() {
    use kenlmrs::model::load_virtual;
    use kenlmrs::types::ModelType;
    use tempfile::NamedTempFile;

    let original = load_model();
    let tmp = NamedTempFile::new().expect("tempfile");
    let path = tmp.path().to_str().unwrap().to_string();
    original.save(&path).expect("save binary");

    // load_virtual should detect binary and load it
    let loaded = load_virtual(&path, &Config::default(), ModelType::Probing).expect("load_virtual");
    let o_idx = original.vocab().index("looking");
    let l_idx = loaded.vocab().index("looking");
    let mut o_out = State::default();
    let mut l_out = State::default();
    let o_ret = original.full_score(&original.begin_sentence_state(), o_idx, &mut o_out);
    let l_ret = loaded.full_score(&loaded.begin_sentence_state(), l_idx, &mut l_out);
    assert_prob_close(o_ret.prob, l_ret.prob, "load_virtual binary detection");
}

// ── Builder pipeline tests ────────────────────────────────────────────────────

#[test]
fn build_arpa_and_score() {
    use kenlmrs::builder::pipeline::build_arpa;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // Write a tiny corpus
    let mut corpus = NamedTempFile::new().unwrap();
    writeln!(corpus, "the cat sat on the mat").unwrap();
    writeln!(corpus, "the cat sat on the hat").unwrap();
    writeln!(corpus, "the dog sat on the log").unwrap();
    corpus.flush().unwrap();

    let arpa_file = NamedTempFile::new().unwrap();
    let arpa_path = arpa_file.path().to_str().unwrap().to_string();

    build_arpa(corpus.path().to_str().unwrap(), &arpa_path, 2).expect("build_arpa should succeed");

    // Load the generated ARPA with ProbingModel
    let model = ProbingModel::new(&arpa_path, &Config::default())
        .expect("ProbingModel should load the generated ARPA");

    // Score at least one word — model should return a valid (non-NaN) probability
    let vocab = model.vocab();
    let idx = vocab.index("cat");
    let mut out = State::default();
    let ret = model.full_score(&ProbingModel::empty_context_state(), idx, &mut out);
    assert!(
        ret.prob.is_finite(),
        "built model should produce finite scores"
    );
    assert!(ret.prob <= 0.0, "log-prob should be non-positive");
}

// ── TrieModel parity tests ─────────────────────────────────────────────────────

fn load_trie_model() -> TrieModel {
    TrieModel::new(TEST_ARPA, &Config::default())
        .unwrap_or_else(|e| panic!("Failed to load TrieModel from {TEST_ARPA}: {e}"))
}

#[test]
fn trie_model_loads_without_panic() {
    let _model = load_trie_model();
}

#[test]
fn trie_model_unigram_prob_matches_probing() {
    let probing = load_model();
    let trie = load_trie_model();
    let p_vocab = probing.vocab();
    let t_vocab = trie.vocab();

    let words = ["looking", "on", "a", "not_in_vocab_xyz"];
    for word in &words {
        let p_idx = p_vocab.index(word);
        let t_idx = t_vocab.index(word);
        if p_idx == p_vocab.not_found() {
            continue;
        }

        let mut p_out = State::default();
        let mut t_out = State::default();
        let p_state = ProbingModel::empty_context_state();
        let t_state = TrieModel::empty_context_state();

        let p_ret = probing.full_score(&p_state, p_idx, &mut p_out);
        let t_ret = trie.full_score(&t_state, t_idx, &mut t_out);

        assert_prob_close(
            p_ret.prob,
            t_ret.prob,
            &format!("unigram P({word}): probing vs trie"),
        );
    }
}

#[test]
fn trie_bigram_looking_on_prob() {
    // Direct check: score "on" in empty context (unigram), then in "looking" context (bigram)
    let trie = load_trie_model();
    let t_vocab = trie.vocab();
    let on_idx = t_vocab.index("on");
    let looking_idx = t_vocab.index("looking");

    // Unigram prob of "on"
    let mut out = State::default();
    let t_state = TrieModel::empty_context_state();
    let uni_on = trie.full_score(&t_state, on_idx, &mut out);

    // Bigram "looking on": score "on" after "looking" (empty sentence context)
    // Build a state with just "looking" as context
    let mut looking_state = TrieModel::empty_context_state();
    let mut tmp = State::default();
    trie.full_score(&looking_state, looking_idx, &mut tmp);
    looking_state = tmp;

    let mut out2 = State::default();
    let bigram_on = trie.full_score(&looking_state, on_idx, &mut out2);

    // The probing model for reference
    let probing = load_model();
    let p_vocab = probing.vocab();
    let p_looking = p_vocab.index("looking");
    let p_on = p_vocab.index("on");
    let mut p_ls = ProbingModel::empty_context_state();
    let mut p_tmp = State::default();
    probing.full_score(&p_ls, p_looking, &mut p_tmp);
    p_ls = p_tmp;
    let mut p_out2 = State::default();
    let p_bigram_on = probing.full_score(&p_ls, p_on, &mut p_out2);

    assert_prob_close(
        p_bigram_on.prob,
        bigram_on.prob,
        &format!(
            "P(on|looking): probing={:.6} trie={:.6}",
            p_bigram_on.prob, bigram_on.prob
        ),
    );
}

#[test]
fn trie_trigram_s_looking_on() {
    // P(on | <s> looking) should match the trigram probability
    let trie = load_trie_model();
    let t_vocab = trie.vocab();
    let looking_idx = t_vocab.index("looking");
    let on_idx = t_vocab.index("on");

    // Build state: score "looking" after <s>, giving bigram state
    let bos_state = trie.begin_sentence_state();
    let mut after_looking = State::default();
    let looking_ret = trie.full_score(&bos_state, looking_idx, &mut after_looking);

    // Now score "on" — should find trigram <s> looking on = -0.3488368
    let mut out = State::default();
    let on_ret = trie.full_score(&after_looking, on_idx, &mut out);

    assert_prob_close(
        -0.3488368,
        on_ret.prob,
        &format!(
            "P(on|<s> looking): trie={:.6}, expected -0.3488368 (trigram)",
            on_ret.prob
        ),
    );
    assert_eq!(
        on_ret.ngram_length, 3,
        "should be trigram match, got {}",
        on_ret.ngram_length
    );
}

#[test]
fn trie_model_first_word_score_matches_probing() {
    let probing = load_model();
    let trie = load_trie_model();
    let p_vocab = probing.vocab();
    let t_vocab = trie.vocab();

    let word = "looking";
    let p_idx = p_vocab.index(word);
    let t_idx = t_vocab.index(word);
    let p_state = probing.begin_sentence_state();
    let t_state = trie.begin_sentence_state();
    let mut p_out = State::default();
    let mut t_out = State::default();

    let p_ret = probing.full_score(&p_state, p_idx, &mut p_out);
    let t_ret = trie.full_score(&t_state, t_idx, &mut t_out);

    assert_prob_close(
        p_ret.prob,
        t_ret.prob,
        &format!(
            "P({word}|<s>): probing {:.6} vs trie {:.6}",
            p_ret.prob, t_ret.prob
        ),
    );
}

#[test]
fn trie_model_sentence_score_matches_probing() {
    let probing = load_model();
    let trie = load_trie_model();
    let p_vocab = probing.vocab();
    let t_vocab = trie.vocab();

    // Score "looking on a" after <s> with both models; totals must agree
    let words = ["looking", "on", "a"];
    let mut p_state = probing.begin_sentence_state();
    let mut t_state = trie.begin_sentence_state();
    let mut p_total = 0.0f32;
    let mut t_total = 0.0f32;

    for word in &words {
        let p_idx = p_vocab.index(word);
        let t_idx = t_vocab.index(word);
        let mut p_out = State::default();
        let mut t_out = State::default();
        let p_s = probing.full_score(&p_state, p_idx, &mut p_out);
        let t_s = trie.full_score(&t_state, t_idx, &mut t_out);
        assert_prob_close(
            p_s.prob,
            t_s.prob,
            &format!(
                "P({word}|ctx): probing {:.6} vs trie {:.6}",
                p_s.prob, t_s.prob
            ),
        );
        p_total += p_s.prob;
        t_total += t_s.prob;
        p_state = p_out;
        t_state = t_out;
    }
    assert_prob_close(p_total, t_total, "sentence total: probing vs trie");
}

// ── ArrayTrieModel parity tests ────────────────────────────────────────────────

fn load_array_trie_model() -> ArrayTrieModel {
    ArrayTrieModel::new(TEST_ARPA, &Config::default())
        .unwrap_or_else(|e| panic!("Failed to load ArrayTrieModel from {TEST_ARPA}: {e}"))
}

#[test]
fn array_trie_model_loads_without_panic() {
    let _model = load_array_trie_model();
}

#[test]
fn array_trie_model_unigram_prob_matches_probing() {
    let probing = load_model();
    let trie = load_array_trie_model();
    let p_vocab = probing.vocab();
    let t_vocab = trie.vocab();

    let words = ["looking", "on", "a", "not_in_vocab_xyz"];
    for word in &words {
        let p_idx = p_vocab.index(word);
        let t_idx = t_vocab.index(word);
        if p_idx == p_vocab.not_found() {
            continue;
        }

        let p_ret = probing.full_score(
            &ProbingModel::empty_context_state(),
            p_idx,
            &mut State::default(),
        );
        let t_ret = trie.full_score(
            &ArrayTrieModel::empty_context_state(),
            t_idx,
            &mut State::default(),
        );

        assert_prob_close(
            p_ret.prob,
            t_ret.prob,
            &format!("array_trie unigram P({word}): probing vs array_trie"),
        );
    }
}

#[test]
fn array_trie_model_sentence_score_matches_probing() {
    let probing = load_model();
    let trie = load_array_trie_model();
    let p_vocab = probing.vocab();
    let t_vocab = trie.vocab();

    let words = ["looking", "on", "a"];
    let mut p_state = probing.begin_sentence_state();
    let mut t_state = trie.begin_sentence_state();
    let mut p_total = 0.0f32;
    let mut t_total = 0.0f32;

    for word in &words {
        let p_idx = p_vocab.index(word);
        let t_idx = t_vocab.index(word);
        let mut p_out = State::default();
        let mut t_out = State::default();
        let p_s = probing.full_score(&p_state, p_idx, &mut p_out);
        let t_s = trie.full_score(&t_state, t_idx, &mut t_out);
        assert_prob_close(
            p_s.prob,
            t_s.prob,
            &format!(
                "array_trie P({word}|ctx): probing {:.6} vs array_trie {:.6}",
                p_s.prob, t_s.prob
            ),
        );
        p_total += p_s.prob;
        t_total += t_s.prob;
        p_state = p_out;
        t_state = t_out;
    }
    assert_prob_close(
        p_total,
        t_total,
        "array_trie sentence total: probing vs array_trie",
    );
}

// ── QuantTrieModel parity tests ────────────────────────────────────────────────

fn load_quant_trie_model() -> QuantTrieModel {
    QuantTrieModel::new(TEST_ARPA, &Config::default())
        .unwrap_or_else(|e| panic!("Failed to load QuantTrieModel from {TEST_ARPA}: {e}"))
}

#[test]
fn quant_trie_model_loads_without_panic() {
    let _model = load_quant_trie_model();
}

#[test]
fn quant_trie_model_unigram_prob_matches_probing() {
    let probing = load_model();
    let trie = load_quant_trie_model();
    let p_vocab = probing.vocab();
    let t_vocab = trie.vocab();

    let words = ["looking", "on", "a"];
    for word in &words {
        let p_idx = p_vocab.index(word);
        let t_idx = t_vocab.index(word);
        if p_idx == p_vocab.not_found() {
            continue;
        }

        let p_ret = probing.full_score(
            &ProbingModel::empty_context_state(),
            p_idx,
            &mut State::default(),
        );
        let t_ret = trie.full_score(
            &QuantTrieModel::empty_context_state(),
            t_idx,
            &mut State::default(),
        );

        assert_prob_close(
            p_ret.prob,
            t_ret.prob,
            &format!("quant_trie unigram P({word}): probing vs quant_trie"),
        );
    }
}

#[test]
fn quant_trie_model_sentence_score_matches_probing() {
    let probing = load_model();
    let trie = load_quant_trie_model();
    let p_vocab = probing.vocab();
    let t_vocab = trie.vocab();

    let words = ["looking", "on", "a"];
    let mut p_state = probing.begin_sentence_state();
    let mut t_state = trie.begin_sentence_state();
    let mut p_total = 0.0f32;
    let mut t_total = 0.0f32;

    for word in &words {
        let p_idx = p_vocab.index(word);
        let t_idx = t_vocab.index(word);
        let mut p_out = State::default();
        let mut t_out = State::default();
        let p_s = probing.full_score(&p_state, p_idx, &mut p_out);
        let t_s = trie.full_score(&t_state, t_idx, &mut t_out);
        assert_prob_close(
            p_s.prob,
            t_s.prob,
            &format!(
                "quant_trie P({word}|ctx): probing {:.6} vs quant_trie {:.6}",
                p_s.prob, t_s.prob
            ),
        );
        p_total += p_s.prob;
        t_total += t_s.prob;
        p_state = p_out;
        t_state = t_out;
    }
    assert_prob_close(
        p_total,
        t_total,
        "quant_trie sentence total: probing vs quant_trie",
    );
}

// ── QuantArrayTrieModel parity tests ──────────────────────────────────────────

fn load_quant_array_trie_model() -> QuantArrayTrieModel {
    QuantArrayTrieModel::new(TEST_ARPA, &Config::default())
        .unwrap_or_else(|e| panic!("Failed to load QuantArrayTrieModel from {TEST_ARPA}: {e}"))
}

#[test]
fn quant_array_trie_model_loads_without_panic() {
    let _model = load_quant_array_trie_model();
}

#[test]
fn quant_array_trie_model_unigram_prob_matches_probing() {
    let probing = load_model();
    let trie = load_quant_array_trie_model();
    let p_vocab = probing.vocab();
    let t_vocab = trie.vocab();

    let words = ["looking", "on", "a"];
    for word in &words {
        let p_idx = p_vocab.index(word);
        let t_idx = t_vocab.index(word);
        if p_idx == p_vocab.not_found() {
            continue;
        }

        let p_ret = probing.full_score(
            &ProbingModel::empty_context_state(),
            p_idx,
            &mut State::default(),
        );
        let t_ret = trie.full_score(
            &QuantArrayTrieModel::empty_context_state(),
            t_idx,
            &mut State::default(),
        );

        assert_prob_close(
            p_ret.prob,
            t_ret.prob,
            &format!("quant_array_trie unigram P({word}): probing vs quant_array_trie"),
        );
    }
}

#[test]
fn quant_array_trie_model_sentence_score_matches_probing() {
    let probing = load_model();
    let trie = load_quant_array_trie_model();
    let p_vocab = probing.vocab();
    let t_vocab = trie.vocab();

    let words = ["looking", "on", "a"];
    let mut p_state = probing.begin_sentence_state();
    let mut t_state = trie.begin_sentence_state();
    let mut p_total = 0.0f32;
    let mut t_total = 0.0f32;

    for word in &words {
        let p_idx = p_vocab.index(word);
        let t_idx = t_vocab.index(word);
        let mut p_out = State::default();
        let mut t_out = State::default();
        let p_s = probing.full_score(&p_state, p_idx, &mut p_out);
        let t_s = trie.full_score(&t_state, t_idx, &mut t_out);
        assert_prob_close(
            p_s.prob,
            t_s.prob,
            &format!(
                "quant_array_trie P({word}|ctx): probing {:.6} vs quant_array_trie {:.6}",
                p_s.prob, t_s.prob
            ),
        );
        p_total += p_s.prob;
        t_total += t_s.prob;
        p_state = p_out;
        t_state = t_out;
    }
    assert_prob_close(p_total, t_total, "quant_array_trie sentence total");
}

// ── TrieModel binary round-trip tests ─────────────────────────────────────────

fn score_sentence(model: &TrieModel, words: &[&str]) -> f32 {
    let vocab = model.vocab();
    let mut state = model.begin_sentence_state();
    let mut total = 0.0f32;
    for &word in words {
        let idx = vocab.index(word);
        let mut out = State::default();
        total += model.full_score(&state, idx, &mut out).prob;
        state = out;
    }
    total
}

#[test]
fn trie_model_binary_round_trip() {
    let original = load_trie_model();
    let tmp = tempfile::NamedTempFile::new().unwrap();
    original.save(tmp.path().to_str().unwrap()).unwrap();
    let loaded = TrieModel::load_binary(tmp.path().to_str().unwrap(), &Config::default()).unwrap();

    let words = ["looking", "on", "a"];
    let orig_score = score_sentence(&original, &words);
    let load_score = score_sentence(&loaded, &words);
    assert_prob_close(
        orig_score,
        load_score,
        "trie binary round-trip sentence score",
    );
}

#[test]
fn array_trie_model_binary_round_trip() {
    let original = load_array_trie_model();
    let tmp = tempfile::NamedTempFile::new().unwrap();
    original.save(tmp.path().to_str().unwrap()).unwrap();
    let loaded =
        ArrayTrieModel::load_binary(tmp.path().to_str().unwrap(), &Config::default()).unwrap();

    let orig_vocab = original.vocab();
    let load_vocab = loaded.vocab();
    let words = ["looking", "on", "a"];
    let mut orig_s = original.begin_sentence_state();
    let mut load_s = loaded.begin_sentence_state();
    let mut orig_total = 0.0f32;
    let mut load_total = 0.0f32;
    for &w in &words {
        let mut o1 = State::default();
        let mut o2 = State::default();
        orig_total += original
            .full_score(&orig_s, orig_vocab.index(w), &mut o1)
            .prob;
        load_total += loaded
            .full_score(&load_s, load_vocab.index(w), &mut o2)
            .prob;
        orig_s = o1;
        load_s = o2;
    }
    assert_prob_close(orig_total, load_total, "array_trie binary round-trip");
}

#[test]
fn quant_trie_model_binary_round_trip() {
    let original = load_quant_trie_model();
    let tmp = tempfile::NamedTempFile::new().unwrap();
    original.save(tmp.path().to_str().unwrap()).unwrap();
    let loaded =
        QuantTrieModel::load_binary(tmp.path().to_str().unwrap(), &Config::default()).unwrap();

    let orig_vocab = original.vocab();
    let load_vocab = loaded.vocab();
    let words = ["looking", "on", "a"];
    let mut orig_s = original.begin_sentence_state();
    let mut load_s = loaded.begin_sentence_state();
    let mut orig_total = 0.0f32;
    let mut load_total = 0.0f32;
    for &w in &words {
        let mut o1 = State::default();
        let mut o2 = State::default();
        orig_total += original
            .full_score(&orig_s, orig_vocab.index(w), &mut o1)
            .prob;
        load_total += loaded
            .full_score(&load_s, load_vocab.index(w), &mut o2)
            .prob;
        orig_s = o1;
        load_s = o2;
    }
    assert_prob_close(orig_total, load_total, "quant_trie binary round-trip");
}

#[test]
fn quant_array_trie_model_binary_round_trip() {
    let original = load_quant_array_trie_model();
    let tmp = tempfile::NamedTempFile::new().unwrap();
    original.save(tmp.path().to_str().unwrap()).unwrap();
    let loaded =
        QuantArrayTrieModel::load_binary(tmp.path().to_str().unwrap(), &Config::default()).unwrap();

    let orig_vocab = original.vocab();
    let load_vocab = loaded.vocab();
    let words = ["looking", "on", "a"];
    let mut orig_s = original.begin_sentence_state();
    let mut load_s = loaded.begin_sentence_state();
    let mut orig_total = 0.0f32;
    let mut load_total = 0.0f32;
    for &w in &words {
        let mut o1 = State::default();
        let mut o2 = State::default();
        orig_total += original
            .full_score(&orig_s, orig_vocab.index(w), &mut o1)
            .prob;
        load_total += loaded
            .full_score(&load_s, load_vocab.index(w), &mut o2)
            .prob;
        orig_s = o1;
        load_s = o2;
    }
    assert_prob_close(orig_total, load_total, "quant_array_trie binary round-trip");
}

// ── Streaming Pipeline integration test ───────────────────────────────────────

#[test]
fn streaming_pipeline_produces_valid_arpa() {
    use kenlmrs::builder::pipeline::{Pipeline, PipelineConfig};
    use std::io::Write;

    // Write a small synthetic corpus
    let corpus_file = tempfile::NamedTempFile::new().unwrap();
    {
        let mut f = std::fs::File::create(corpus_file.path()).unwrap();
        writeln!(f, "the cat sat on the mat").unwrap();
        writeln!(f, "the cat sat on the mat").unwrap();
        writeln!(f, "the dog ran on the road").unwrap();
    }

    let arpa_file = tempfile::NamedTempFile::new().unwrap();
    let mut config = PipelineConfig::new();
    config.order = 2;
    Pipeline(
        &config,
        corpus_file.path().to_str().unwrap(),
        arpa_file.path().to_str().unwrap(),
    )
    .expect("Pipeline should build ARPA without error");

    // Verify the output is loadable and scores "the" as a unigram
    let model = ProbingModel::new(arpa_file.path().to_str().unwrap(), &Config::default())
        .expect("Pipeline output should be loadable by ProbingModel");
    let vocab = model.vocab();
    let the_idx = vocab.index("the");
    assert_ne!(the_idx, vocab.not_found(), "'the' should be in vocab");
    let mut out = State::default();
    let ret = model.full_score(&ProbingModel::empty_context_state(), the_idx, &mut out);
    assert!(
        ret.prob < 0.0,
        "log-prob of 'the' should be negative, got {}",
        ret.prob
    );
}
