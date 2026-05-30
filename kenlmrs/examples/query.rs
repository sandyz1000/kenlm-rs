/// KenLM-RS query binary — scores sentences from stdin using a language model.
///
/// Matches the output format of the C++ KenLM `query` binary:
///   p=<log10_prob> [<ngram_len>] <word>
///   Total: <total> OOV: <oov_count> Tokens: <token_count>
use clap::Parser;
use kenlmrs::model::ProbingModel;
use kenlmrs::types::{Config, State};
use kenlmrs::vocabulary::Vocabulary;
use std::io::{self, BufRead, BufWriter, Write};

#[derive(Parser)]
#[command(name = "query", about = "Score sentences from stdin using a KenLM ARPA model")]
struct Args {
    /// Path to the ARPA language model file
    model: String,

    /// Include sentence-boundary markers (<s> at start, </s> at end)
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    sentence_context: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    eprintln!("Loading model from: {}", args.model);
    let model = ProbingModel::new(&args.model, &Config::default())?;
    eprintln!("Model loaded (order {})", model.order());

    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let sentence = line?;
        if sentence.trim().is_empty() {
            continue;
        }
        score_sentence(&model, &sentence, args.sentence_context, &mut out)?;
    }

    Ok(())
}

fn score_sentence(
    model: &ProbingModel,
    sentence: &str,
    sentence_context: bool,
    out: &mut impl Write,
) -> Result<(), io::Error> {
    let vocab = model.vocab();
    let mut state = if sentence_context {
        model.begin_sentence_state()
    } else {
        ProbingModel::empty_context_state()
    };

    let mut total = 0.0_f32;
    let mut oov = 0_usize;
    let mut token_count = 0_usize;

    for word in sentence.split_whitespace() {
        let idx = vocab.index(word);
        if idx == vocab.not_found() {
            oov += 1;
        }

        let mut next = State::default();
        let ret = model.full_score(&state, idx, &mut next);
        writeln!(out, "p={:.6} [{}] {}", ret.prob, ret.ngram_length, word)?;

        total += ret.prob;
        token_count += 1;
        state = next;
    }

    if sentence_context {
        let eos = vocab.end_sentence();
        let mut next = State::default();
        let ret = model.full_score(&state, eos, &mut next);
        writeln!(out, "p={:.6} [{}] </s>", ret.prob, ret.ngram_length)?;
        total += ret.prob;
        token_count += 1;
    }

    writeln!(out, "Total: {:.6} OOV: {} Tokens: {}", total, oov, token_count)
}
