/// Criterion benchmarks for kenlm-rs scoring throughput.
///
/// Measures how fast the Rust probing model can score words given context,
/// mirroring the inner loop of the C++ `query` binary for fair comparison.
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use kenlmrs::model::ProbingModel;
use kenlmrs::types::{Config, State};
use kenlmrs::vocabulary::Vocabulary;

const TEST_ARPA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/test.arpa");

/// Fixed sentences that cover unigrams, bigrams, trigrams, 4-grams, and 5-grams.
const SENTENCES: &[&str] = &[
    "looking on a little more loin",
    "also would consider higher looking",
    "foo bar",
    "baz",
    "looking on a little more loin",
    "what i would also consider",
    "the screening a little more",
    "higher small looking on a",
];

fn score_sentence(model: &ProbingModel, words: &[u32]) -> f32 {
    let vocab = model.vocab();
    let eos = vocab.end_sentence();
    let mut state = model.begin_sentence_state();
    let mut total = 0.0_f32;
    for &word in words {
        let mut out = State::default();
        total += model.full_score(&state, word, &mut out).prob;
        state = out;
    }
    let mut out = State::default();
    total += model.full_score(&state, eos, &mut out).prob;
    total
}

fn bench_full_score(c: &mut Criterion) {
    let model = ProbingModel::new(TEST_ARPA, &Config::default())
        .expect("failed to load test.arpa");
    let vocab = model.vocab();

    // Pre-resolve all word indices so lookup overhead is excluded from the hot loop.
    let sentences: Vec<Vec<u32>> = SENTENCES
        .iter()
        .map(|s| s.split_whitespace().map(|w| vocab.index(w)).collect())
        .collect();

    let total_words: u64 = sentences.iter().map(|s| s.len() as u64 + 1).sum(); // +1 for </s>

    let mut group = c.benchmark_group("scoring");
    group.throughput(Throughput::Elements(total_words));

    group.bench_function("full_score_all_sentences", |b| {
        b.iter(|| {
            let mut sum = 0.0_f32;
            for words in &sentences {
                sum += score_sentence(black_box(&model), black_box(words));
            }
            black_box(sum)
        })
    });

    group.finish();
}

fn bench_model_load(c: &mut Criterion) {
    c.bench_function("model_load", |b| {
        b.iter(|| {
            ProbingModel::new(black_box(TEST_ARPA), black_box(&Config::default()))
                .expect("failed to load")
        })
    });
}

criterion_group!(benches, bench_full_score, bench_model_load);
criterion_main!(benches);
