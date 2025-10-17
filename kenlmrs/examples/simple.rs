use kenlmrs::types::{ Config, State };
use kenlmrs::vocabulary::Vocabulary;

fn main() {
    println!("KenLM-RS Simple Test");

    // Create a default configuration
    let config = Config::default();

    println!("Configuration created successfully");
    println!("Load method: {:?}", config.load_method);
    println!("Probing multiplier: {}", config.probing_multiplier);

    // Test vocabulary
    let mut vocab = kenlmrs::vocabulary::ProbingVocabulary::new();

    // Add some words
    let word1_idx = vocab.add_word("hello");
    let word2_idx = vocab.add_word("world");
    let word3_idx = vocab.add_word("test");

    println!("\nVocabulary test:");
    println!("'hello' -> {}", word1_idx);
    println!("'world' -> {}", word2_idx);
    println!("'test' -> {}", word3_idx);

    // Test lookups
    println!("Lookup 'hello': {}", vocab.index("hello"));
    println!("Lookup 'world': {}", vocab.index("world"));
    println!("Lookup 'unknown': {}", vocab.index("unknown"));

    // Test state
    let state = State::new();
    println!("\nEmpty state length: {}", state.length);

    println!("\nBasic functionality test passed!");
}
