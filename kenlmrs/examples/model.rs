/// Model example - demonstrates language model loading and querying
/// Similar to the Python kenlm example
use kenlmrs::types::{Config, State};
use kenlmrs::vocabulary::{ProbingVocabulary, Vocabulary};

fn main() {
    println!("KenLM-RS Model Example");
    println!("======================\n");

    // This example demonstrates the API similar to Python kenlm
    // model = kenlm.LanguageModel('test.arpa')

    let config = Config::default();
    println!("Configuration:");
    println!("  Load method: {:?}", config.load_method);
    println!("  Probing multiplier: {}", config.probing_multiplier);
    println!("  Unknown log prob: {}", config.unknown_missing_logprob);
    println!();

    // Initialize a vocabulary
    let mut vocab = ProbingVocabulary::new();

    // Build vocabulary similar to a 5-gram model
    println!("Building vocabulary...");
    let words = vec![
        "<s>", "</s>", "<unk>", "language", "modeling", "is", "fun", ".", "the", "a", "an", "and",
        "of", "to", "in", "that", "it", "was", "for",
    ];

    for word in &words {
        vocab.add_word(word);
    }
    println!("Vocabulary size: {}", words.len());
    println!();

    // Example 1: Sentence scoring
    println!("Example 1: Sentence Scoring");
    println!("{}", "=".repeat(40));
    let sentence = "language modeling is fun .";
    println!("Sentence: \"{}\"", sentence);

    let tokens: Vec<&str> = sentence.split_whitespace().collect();
    let score = score_sentence(&tokens, &vocab);
    println!("Total score: {:.4}", score);
    println!(
        "Perplexity: {:.2}",
        (10.0f32).powf(-score / (tokens.len() as f32))
    );
    println!();

    // Example 2: Full scores with n-gram matching
    println!("Example 2: Full Scores with N-gram Matching");
    println!("{}", "=".repeat(40));
    full_scores_example(&tokens, &vocab);
    println!();

    // Example 3: Out-of-vocabulary detection
    println!("Example 3: OOV Detection");
    println!("{}", "=".repeat(40));
    oov_detection_example(&vocab);
    println!();

    // Example 4: Stateful querying
    println!("Example 4: Stateful Query");
    println!("{}", "=".repeat(40));
    stateful_query_example(&vocab);
    println!();

    println!("Model example completed!");
}

/// Score a sentence (simplified)
fn score_sentence(tokens: &[&str], vocab: &ProbingVocabulary) -> f32 {
    let mut total = 0.0f32;

    // Add <s> at the beginning
    for (i, token) in tokens.iter().enumerate() {
        let idx = vocab.index(token);

        // Simulate scoring based on context length
        let ngram_length = std::cmp::min(i + 2, 5); // Up to 5-gram
        let base_score = if idx == vocab.not_found() {
            -100.0 // Unknown word
        } else {
            -1.5 - (ngram_length as f32) * 0.2 // Better score with longer context
        };

        total += base_score;
    }

    // Add score for </s>
    total += -0.5;

    total
}

/// Full scores showing n-gram matches
fn full_scores_example(tokens: &[&str], vocab: &ProbingVocabulary) {
    println!("Word-by-word scoring:");

    // Context starts with <s>
    let mut context = vec!["<s>"];

    for token in tokens {
        let idx = vocab.index(token);
        let is_oov = idx == vocab.not_found();

        // Determine n-gram length (how much context was used)
        let ngram_length = std::cmp::min(context.len() + 1, 5);

        // Calculate score
        let score = if is_oov {
            -100.0
        } else {
            -1.5 - (ngram_length as f32) * 0.2
        };

        // Build the n-gram string
        let start_idx = if context.len() + 1 > ngram_length {
            context.len() + 1 - ngram_length
        } else {
            0
        };
        let mut ngram = context[start_idx..].to_vec();
        ngram.push(token);

        println!(
            "  {:.4} [{}] {}{}",
            score,
            ngram_length,
            ngram.join(" "),
            if is_oov { " [OOV]" } else { "" }
        );

        // Update context
        context.push(token);
        if context.len() > 4 {
            // Keep last 4 words for 5-gram context
            context.remove(0);
        }
    }

    // Final </s> token
    let ngram_length = std::cmp::min(context.len() + 1, 5);
    let start_idx = if context.len() + 1 > ngram_length {
        context.len() + 1 - ngram_length
    } else {
        0
    };
    let mut ngram = context[start_idx..].to_vec();
    ngram.push("</s>");
    println!("  {:.4} [{}] {}", -0.5, ngram_length, ngram.join(" "));
}

/// Detect out-of-vocabulary words
fn oov_detection_example(vocab: &ProbingVocabulary) {
    let test_words = vec![
        "language",
        "modeling",
        "is",
        "fun",
        "unknown",
        "supercalifragilisticexpialidocious",
    ];

    println!("Checking vocabulary membership:");
    for word in test_words {
        let idx = vocab.index(word);
        let in_vocab = idx != vocab.not_found();
        println!(
            "  \"{}\" -> {} {}",
            word,
            idx,
            if in_vocab { "✓" } else { "[OOV]" }
        );
    }
}

/// Stateful query example
fn stateful_query_example(vocab: &ProbingVocabulary) {
    println!("Stateful scoring (like model.BaseScore in Python):");
    println!();

    // Initialize states
    let mut state1 = State::new();
    let mut state2 = State::new();

    // Begin sentence context
    println!("Starting with <s> context...");
    state1.length = 1;
    state1.words[0] = vocab.index("<s>");

    // Score "language" given "<s>"
    let word1 = "language";
    let word1_idx = vocab.index(word1);
    let score1 = base_score(word1_idx, &state1, vocab);
    println!("  BaseScore(<s>, \"{}\") = {:.4}", word1, score1);

    // Update state
    state2.length = 2;
    state2.words[0] = state1.words[0];
    state2.words[1] = word1_idx;

    // Score "modeling" given "<s> language"
    let word2 = "modeling";
    let word2_idx = vocab.index(word2);
    let score2 = base_score(word2_idx, &state2, vocab);
    println!("  BaseScore(<s> language, \"{}\") = {:.4}", word2, score2);

    // Update state for next word
    state1.length = 3;
    state1.words[0] = state2.words[0];
    state1.words[1] = state2.words[1];
    state1.words[2] = word2_idx;

    // Score "is" given "<s> language modeling"
    let word3 = "is";
    let word3_idx = vocab.index(word3);
    let score3 = base_score(word3_idx, &state1, vocab);
    println!(
        "  BaseScore(<s> language modeling, \"{}\") = {:.4}",
        word3, score3
    );

    let total = score1 + score2 + score3;
    println!();
    println!("Accumulated score: {:.4}", total);
    println!("(This matches scoring \"language modeling is\" with bos=True, eos=False)");
}

/// Simulate base scoring with state
fn base_score(word_idx: u32, state: &State, vocab: &ProbingVocabulary) -> f32 {
    let is_oov = word_idx == vocab.not_found();
    let ngram_length = state.length + 1;

    if is_oov {
        -100.0
    } else {
        -1.5 - (ngram_length as f32) * 0.2
    }
}
