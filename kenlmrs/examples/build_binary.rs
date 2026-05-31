/// Build binary example - similar to KenLM's build_binary_main.cc
/// This demonstrates building a binary language model from ARPA format
use kenlmrs::arpa_reader::read_arpa_counts;
use kenlmrs::types::{Config, LoadMethod, ModelType};
use kenlmrs::utils::pieces::file::FilePiece;
use std::fs::File;
use std::io::Write;
use tempfile::tempdir;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("KenLM-RS Build Binary Example");
    println!("==============================\n");

    // Parse command line arguments (simplified version)
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage(&args[0]);
        return Ok(());
    }

    // Configuration
    let mut config = Config::default();
    config.load_method = LoadMethod::ReadMethod;

    println!("Build configuration:");
    println!("  Probing multiplier: {}", config.probing_multiplier);
    println!(
        "  Unknown missing log prob: {}",
        config.unknown_missing_logprob
    );
    println!("  Building memory: {} bytes", config.building_memory);
    println!();

    // Create a sample ARPA file for testing
    let dir = tempdir()?;
    let arpa_path = dir.path().join("sample.arpa");
    create_sample_arpa(&arpa_path)?;

    println!("Created sample ARPA file: {:?}", arpa_path);
    println!();

    // Read ARPA file and get counts
    let mut fp = FilePiece::open(&arpa_path)?;
    let counts = read_arpa_counts(&mut fp)?;

    println!("ARPA file statistics:");
    println!("  Model order: {}", counts.len());
    for (order, count) in counts.iter().enumerate() {
        println!("  {}-gram count: {}", order + 1, count);
    }
    println!();

    // Calculate memory requirements (simplified)
    let model_type = ModelType::Probing;
    estimate_memory_usage(&counts, model_type, &config);

    println!("\nBuild binary example completed!");
    println!("\nNote: Full binary building requires implementing:");
    println!("  - Vocabulary building from ARPA");
    println!("  - Trie/Hash table construction");
    println!("  - Quantization (optional)");
    println!("  - Binary serialization");

    Ok(())
}

fn print_usage(program_name: &str) {
    println!("Usage: {} [options] input.arpa [output.bin]", program_name);
    println!();
    println!("Options:");
    println!("  -p <float>  Probing multiplier (default: 1.5)");
    println!("  -u <float>  Unknown word log probability (default: -100)");
    println!("  -t <type>   Model type: probing, trie (default: probing)");
    println!("  -q <bits>   Quantization bits (trie only)");
    println!("  -h          Show this help message");
    println!();
    println!("Model types:");
    println!("  probing     - Probing hash table (fastest, uses most memory)");
    println!("  trie        - Trie with bit packing (slower, uses less memory)");
}

fn create_sample_arpa(path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(path)?;

    // Write a more comprehensive ARPA file
    writeln!(file, "\\data\\")?;
    writeln!(file, "ngram 1=10")?;
    writeln!(file, "ngram 2=8")?;
    writeln!(file, "ngram 3=3")?;
    writeln!(file, "")?;

    writeln!(file, "\\1-grams:")?;
    writeln!(file, "-99\t<s>\t-0.4149733")?;
    writeln!(file, "-1.029493\t</s>")?;
    writeln!(file, "-1.995635\t<unk>\t-20")?;
    writeln!(file, "-1.285941\ta\t-0.69897")?;
    writeln!(file, "-1.687872\tlanguage\t-0.30103")?;
    writeln!(file, "-1.687872\tmodeling\t-0.30103")?;
    writeln!(file, "-1.687872\tis\t-0.30103")?;
    writeln!(file, "-1.687872\tfun\t-0.30103")?;
    writeln!(file, "-1.383514\t.\t-0.845098")?;
    writeln!(file, "-1.139057\t,\t-0.30103")?;
    writeln!(file, "")?;

    writeln!(file, "\\2-grams:")?;
    writeln!(file, "-0.4846522\t<s> language\t-0.4771214")?;
    writeln!(file, "-0.09132547\ta language\t-0.69897")?;
    writeln!(file, "-0.2922095\tlanguage modeling")?;
    writeln!(file, "-0.2922095\tmodeling is")?;
    writeln!(file, "-0.2922095\tis fun")?;
    writeln!(file, "-0.5136299\tfun .\t-0.4771212")?;
    writeln!(file, "-0.6925742\t. </s>")?;
    writeln!(file, "-0.7522095\t, is")?;
    writeln!(file, "")?;

    writeln!(file, "\\3-grams:")?;
    writeln!(file, "-0.0283603\tlanguage modeling is\t-0.4771212")?;
    writeln!(file, "-0.0283603\tmodeling is fun\t-0.4771212")?;
    writeln!(file, "-0.01916512\tis fun .")?;
    writeln!(file, "")?;

    writeln!(file, "\\end\\")?;
    file.flush()?;

    Ok(())
}

fn estimate_memory_usage(counts: &[u64], model_type: ModelType, config: &Config) {
    println!("Memory estimation for {:?} model:", model_type);

    // Rough estimates based on KenLM's approach
    let vocab_size = counts[0];
    let total_ngrams: u64 = counts.iter().sum();

    // Vocabulary memory: ~8 bytes per word index + string overhead
    let vocab_memory = vocab_size * 24; // Approximate

    // Model-specific memory
    let model_memory = match model_type {
        ModelType::Probing | ModelType::RestProbing => {
            // Probing: hash table with probing_multiplier overhead
            // Each entry: ~12 bytes (key + prob + backoff)
            ((total_ngrams as f32) * config.probing_multiplier * 12.0) as u64
        }
        ModelType::Trie | ModelType::QuantTrie => {
            // Trie: more compact but harder to estimate
            // Rough estimate: 8 bytes per n-gram
            total_ngrams * 8
        }
        _ => {
            total_ngrams * 10 // Generic estimate
        }
    };

    let total_memory = vocab_memory + model_memory;

    println!("  Vocabulary: {} MB", vocab_memory / (1024 * 1024));
    println!("  Model data: {} MB", model_memory / (1024 * 1024));
    println!("  Total estimated: {} MB", total_memory / (1024 * 1024));
    println!();
    println!("Breakdown:");
    for (order, count) in counts.iter().enumerate() {
        let bytes_per_entry = match model_type {
            ModelType::Probing | ModelType::RestProbing => 12,
            _ => 8,
        };
        let order_memory = count * bytes_per_entry;
        println!(
            "  {}-grams: {} entries, ~{} KB",
            order + 1,
            count,
            order_memory / 1024
        );
    }
}
