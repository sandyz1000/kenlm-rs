/// Query example - similar to KenLM's query_main.cc
/// This demonstrates how to query a language model for sentence probabilities
use kenlmrs::types::{ Config, LoadMethod };
use kenlmrs::vocabulary::{ Vocabulary, ProbingVocabulary };
use std::io::{ self, BufRead };

fn main() {
    println!("KenLM-RS Query Example");
    println!("======================\n");

    // Configuration similar to C++ KenLM
    let mut config = Config::default();
    config.load_method = LoadMethod::ReadMethod; // Default load method
    config.probing_multiplier = 1.5;

    println!("Configuration:");
    println!("  Load method: {:?}", config.load_method);
    println!("  Probing multiplier: {}", config.probing_multiplier);
    println!();

    // Initialize vocabulary
    let mut vocab = ProbingVocabulary::new();

    // Add some sample words to vocabulary
    let words = vec![
        "<s>",
        "</s>",
        "<unk>",
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "language",
        "modeling",
        "fun",
        "hello",
        "world",
        "test"
    ];

    println!("Building vocabulary with {} words", words.len());
    for word in &words {
        vocab.add_word(word);
    }
    println!();

    // Demo: Query mode
    println!("Enter sentences to score (Ctrl+D to exit):");
    println!("Each sentence will be wrapped with <s> and </s>");
    println!();

    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        match line {
            Ok(sentence) if !sentence.trim().is_empty() => {
                score_sentence(&sentence, &vocab, &config);
            }
            Ok(_) => {
                continue;
            }
            Err(e) => {
                eprintln!("Error reading input: {}", e);
                break;
            }
        }
    }

    println!("\nQuery session complete.");
}

/// Score a sentence and print detailed information
fn score_sentence(sentence: &str, vocab: &ProbingVocabulary, _config: &Config) {
    println!("\nScoring: \"{}\"", sentence);
    println!("{}", "=".repeat(60));

    // Tokenize the sentence
    let tokens: Vec<&str> = sentence.split_whitespace().collect();

    // Wrap with sentence markers
    let mut words = vec!["<s>"];
    words.extend(&tokens);
    words.push("</s>");

    println!("Tokens ({} total):", words.len());
    for (i, word) in words.iter().enumerate() {
        let idx = vocab.index(word);
        let oov_marker = if idx == vocab.not_found() { " [OOV]" } else { "" };
        println!("  {}: {} -> {}{}", i, word, idx, oov_marker);
    }
    println!();

    // Simulate scoring (placeholder since we don't have a full model yet)
    println!("Word-level scores:");
    let mut total_score = 0.0f32;
    let mut oov_count = 0u32;

    for (i, word) in words.iter().enumerate().skip(1) {
        let idx = vocab.index(word);
        let is_oov = idx == vocab.not_found();

        if is_oov {
            oov_count += 1;
        }

        // Simulate n-gram matching and scoring
        // In a real implementation, this would query the trie/hash table
        let ngram_length = if i == 1 { 1 } else { std::cmp::min(i, 3) };
        let score = if is_oov { -100.0 } else { -1.5 * (i as f32) }; // Dummy score

        total_score += score;

        let context_start = if i >= ngram_length { i - ngram_length + 1 } else { 0 };
        let context: Vec<&str> = words[context_start..=i].to_vec();

        println!("  {:8.4} [{}] {} {}", score, ngram_length, context.join(" "), if is_oov {
            "[OOV]"
        } else {
            ""
        });
    }

    println!();
    println!("Sentence score: {:.4}", total_score);
    println!("OOV words: {}", oov_count);
    println!("Total tokens: {}", words.len());

    // Perplexity calculation
    let perplexity = (10.0f32).powf(-total_score / (words.len() as f32));
    println!("Perplexity: {:.2}", perplexity);
}
