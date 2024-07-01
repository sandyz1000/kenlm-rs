use std::io::{self, Write};

#[derive(Debug)]
pub struct QueryPrinter {
    out: io::Stdout,
    print_word: bool,
    print_line: bool,
    print_summary: bool,
    flush: bool,
}

impl QueryPrinter {
    fn new(fd: i32, print_word: bool, print_line: bool, print_summary: bool, flush: bool) -> Self {
        QueryPrinter {
            out: io::stdout(),
            print_word,
            print_line,
            print_summary,
            flush,
        }
    }

    fn word(&mut self, surface: &str, vocab: usize, ret: &FullScoreReturn) {
        if !self.print_word {
            return;
        }
        write!(
            self.out,
            "{}={} {} {}\t",
            surface, vocab, ret.ngram_length, ret.prob
        )
        .unwrap();
        if self.flush {
            self.out.flush().unwrap();
        }
    }

    fn line(&mut self, oov: u64, total: f32) {
        if !self.print_line {
            return;
        }
        writeln!(self.out, "Total: {} OOV: {}", total, oov).unwrap();
        if self.flush {
            self.out.flush().unwrap();
        }
    }

    fn summary(&mut self, ppl_including_oov: f64, ppl_excluding_oov: f64, corpus_oov: u64, corpus_tokens: u64) {
        if !self.print_summary {
            return;
        }
        writeln!(
            self.out,
            "Perplexity including OOVs:\t{}\nPerplexity excluding OOVs:\t{}\nOOVs:\t{}\nTokens:\t{}",
            ppl_including_oov, ppl_excluding_oov, corpus_oov, corpus_tokens
        )
        .unwrap();
        self.out.flush().unwrap();
    }
}

#[derive(Debug)]
pub struct FullScoreReturn {
    ngram_length: u32,
    prob: f32,
}


#[derive(Debug)]
struct Model;

impl Model {
    fn begin_sentence_state(&self) -> State {
        State
    }

    fn null_context_state(&self) -> State {
        State
    }

    fn full_score(&self, _state: State, _vocab: WordIndex, _out: &mut State) -> FullScoreReturn {
        FullScoreReturn {
            ngram_length: 0,
            prob: 0.0,
        }
    }

    fn get_vocabulary(&self) -> Vocabulary {
        Vocabulary
    }
}

#[derive(Debug)]
pub struct State;

#[derive(Debug)]
pub struct Vocabulary;

impl Vocabulary {
    fn index(&self, _word: &str) -> WordIndex {
        0
    }

    fn not_found(&self) -> WordIndex {
        0
    }

    fn end_sentence(&self) -> WordIndex {
        0
    }
}

type WordIndex = usize;

fn query<M: ModelTrait, P: PrinterTrait>(model: &M, sentence_context: bool, printer: &mut P) {
    let mut state = if sentence_context {
        model.begin_sentence_state()
    } else {
        model.null_context_state()
    };
    let mut out = State;
    let mut ret;
    let mut word = String::new();

    let mut corpus_total = 0.0;
    let mut corpus_total_oov_only = 0.0;
    let mut corpus_oov = 0;
    let mut corpus_tokens = 0;

    let mut in_reader = io::stdin();

    loop {
        state = if sentence_context {
            model.begin_sentence_state()
        } else {
            model.null_context_state()
        };
        let mut total = 0.0;
        let mut oov = 0;

        while in_reader.read_line(&mut word).unwrap() > 0 {
            let vocab = model.get_vocabulary().index(&word);
            ret = model.full_score(state, vocab, &mut out);
            if vocab == model.get_vocabulary().not_found() {
                oov += 1;
                corpus_total_oov_only += ret.prob;
            }
            total += ret.prob;
            printer.word(&word, vocab, &ret);
            corpus_tokens += 1;
            state = out;
        }

        if sentence_context {
            ret = model.full_score(state, model.get_vocabulary().end_sentence(), &mut out);
            total += ret.prob;
            corpus_tokens += 1;
            printer.word("</s>", model.get_vocabulary().end_sentence(), &ret);
        }
        printer.line(oov, total);
        corpus_total += total;
        corpus_oov += oov;
    }

    printer.summary(
        10.0_f64.powf(-(corpus_total / corpus_tokens as f64)),
        10.0_f64.powf(-((corpus_total - corpus_total_oov_only) / (corpus_tokens - corpus_oov) as f64)),
        corpus_oov,
        corpus_tokens,
    );
}

pub trait ModelTrait {
    fn begin_sentence_state(&self) -> State;
    fn null_context_state(&self) -> State;
    fn full_score(&self, state: State, vocab: WordIndex, out: &mut State) -> FullScoreReturn;
    fn get_vocabulary(&self) -> Vocabulary;
}

pub trait PrinterTrait {
    fn word(&mut self, surface: &str, vocab: usize, ret: &FullScoreReturn);
    fn line(&mut self, oov: u64, total: f32);
    fn summary(&mut self, ppl_including_oov: f64, ppl_excluding_oov: f64, corpus_oov: u64, corpus_tokens: u64);
}
