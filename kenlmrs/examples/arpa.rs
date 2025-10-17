/// Test reading ARPA files
use kenlmrs::arpa::{read_arpa_counts, PositiveProbWarn};
use kenlmrs::constant::WarningAction;
use kenlmrs::utils::pieces::file::FilePiece;
use std::fs::File;
use std::io::Write;
use tempfile::tempdir;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("KenLM-RS ARPA Reader Test\n");

    // Create a simple test ARPA file
    let dir = tempdir()?;
    let file_path = dir.path().join("test.arpa");
    let mut file = File::create(&file_path)?;

    // Write a minimal ARPA format file
    writeln!(file, "\\data\\")?;
    writeln!(file, "ngram 1=3")?;
    writeln!(file, "ngram 2=2")?;
    writeln!(file, "")?;
    writeln!(file, "\\1-grams:")?;
    writeln!(file, "-1.0\t<unk>\t0.0")?;
    writeln!(file, "-2.5\thello\t-0.3")?;
    writeln!(file, "-3.2\tworld\t-0.1")?;
    writeln!(file, "")?;
    writeln!(file, "\\2-grams:")?;
    writeln!(file, "-0.5\thello world")?;
    writeln!(file, "-1.2\tworld hello")?;
    writeln!(file, "")?;
    writeln!(file, "\\end\\")?;
    file.flush()?;
    drop(file);

    println!("Created test ARPA file: {:?}", file_path);

    // Read the ARPA counts
    let mut fp = FilePiece::open(&file_path)?;
    let counts = read_arpa_counts(&mut fp)?;

    println!("\nARPA counts:");
    for (order, count) in counts.iter().enumerate() {
        println!("  {}-grams: {}", order + 1, count);
    }

    assert_eq!(counts.len(), 2, "Expected 2 orders (unigrams and bigrams)");
    assert_eq!(counts[0], 3, "Expected 3 unigrams");
    assert_eq!(counts[1], 2, "Expected 2 bigrams");

    println!("\n✅ ARPA counts reading test passed!");

    // Test positive probability warning
    let warn = PositiveProbWarn::new(WarningAction::Silent);
    warn.warn(0.5); // Should not panic with Silent

    println!("✅ Positive probability warning test passed!");

    println!("\n🎉 All ARPA reader tests passed!");
    Ok(())
}
