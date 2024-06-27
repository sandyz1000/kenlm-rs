use std::cmp::Ordering;
use crate::stream;
use super::ordering::SuffixOrder;


const K_METADATA_HEADER: &str = "KenLM intermediate binary file";

pub struct ModelBuffer {
    file_base: String,
    keep_buffer: bool,
    output_q: bool,
    counts: Vec<u64>,
    vocab_file: Option<File>,
    files: Vec<File>,
}

impl ModelBuffer {
    pub fn new(file_base: &str, keep_buffer: bool, output_q: bool) -> Self {
        let file_base = file_base.to_string();
        let vocab_file = if keep_buffer {
            Some(util::create_or_throw(&format!("{}.vocab", file_base)))
        } else {
            Some(util::make_temp(&file_base))
        };

        ModelBuffer {
            file_base,
            keep_buffer,
            output_q,
            counts: Vec::new(),
            vocab_file,
            files: Vec::new(),
        }
    }

    pub fn from_file_base(file_base: &str) -> Self {
        let file_base = file_base.to_string();
        let full_name = format!("{}.kenlm_intermediate", file_base);
        let mut in_file = util::FilePiece::open(&full_name);

        let token = in_file.read_line();
        if token != K_METADATA_HEADER {
            panic!("File {} begins with \"{}\" not {}", full_name, token, K_METADATA_HEADER);
        }

        let token = in_file.read_delimited();
        if token != "Counts" {
            panic!("Expected Counts, got \"{}\" in {}", token, full_name);
        }

        let mut counts = Vec::new();
        while let Some(count) = in_file.read_ulong() {
            counts.push(count);
        }

        let token = in_file.read_delimited();
        if token != "Payload" {
            panic!("Expected Payload, got \"{}\" in {}", token, full_name);
        }

        let token = in_file.read_delimited();
        let output_q = match token.as_str() {
            "q" => true,
            "pb" => false,
            _ => panic!("Unknown payload {}", token),
        };

        let vocab_file = Some(util::open_read_or_throw(&format!("{}.vocab", file_base)));

        let mut files = Vec::with_capacity(counts.len());
        for i in 0..counts.len() {
            files.push(util::open_read_or_throw(&format!("{}.{}", file_base, i + 1)));
        }

        ModelBuffer {
            file_base,
            keep_buffer: false,
            output_q,
            counts,
            vocab_file,
            files,
        }
    }

    pub fn sink(&mut self, chains: &mut stream::chain::Chain, counts: Vec<u64>) {
        self.counts = counts;
        self.files = Vec::with_capacity(chains.len());
        
        for (i, chain) in chains.iter_mut().enumerate() {
            let file = if self.keep_buffer {
                util::create_or_throw(&format!("{}.{}", self.file_base, i + 1))
            } else {
                util::make_temp(&self.file_base)
            };
            files.push(file);
            chain.write_to_file(files.last().unwrap());
        }

        if self.keep_buffer {
            let metadata = create_or_throw(&format!("{}.kenlm_intermediate", self.file_base));
            let mut meta = FileStream::new(metadata);
            meta.write_all(K_METADATA_HEADER.as_bytes());
            meta.write_all(b"\nCounts");
            for &count in &self.counts {
                meta.write_all(format!(" {}", count).as_bytes());
            }
            meta.write_all(b"\nPayload ");
            meta.write_all(if self.output_q { b"q" } else { b"pb" });
            meta.write_all(b"\n");
        }
    }

    pub fn source(&mut self, chains: &mut util::stream::Chains) {
        assert!(chains.len() <= self.files.len());
        for (i, chain) in chains.iter_mut().enumerate() {
            chain.set_progress_target(util::file_size(&self.files[i]));
            chain.read_from_file(&self.files[i]);
        }
    }

    pub fn source_single(&mut self, order_minus_1: usize, chain: &mut util::stream::Chain) {
        chain.read_from_file(&self.files[order_minus_1]);
    }

    pub fn slow_query(&self, context: &ngram::State, word: WordIndex, out: &mut ngram::State) -> f32 {
        let mut value: ProbBackoff = Default::default();
        let offset = word as usize * (std::mem::size_of::<WordIndex>() + std::mem::size_of::<ProbBackoff>()) + std::mem::size_of::<WordIndex>();
        util::read_from_file_at_offset(&self.files[0], &mut value, offset);

        out.backoff[0] = value.backoff;
        out.words[0] = word;
        out.length = 1;

        let mut buffer = vec![0; context.length + 1];
        let mut query = vec![0; context.length + 1];
        query[..context.length].copy_from_slice(&context.words);
        query[context.length] = word;
        query.reverse();

        for order in 2..=query.len().min(context.length + 1) {
            let less = SuffixOrder::new(order);
            let key = &query[query.len() - order..];
            let file = &self.files[order - 1];
            let length = order * mem::size_of::<WordIndex>() + mem::size_of::<ProbBackoff>();
            let mut begin = 0;
            let mut end = util::file_size(file) / length as u64;

            loop {
                if end <= begin {
                    return context.backoff[out.length - 1..context.length].iter().sum::<f32>() + value.prob;
                }

                let test = begin + (end - begin) / 2;
                util::read_from_file_at_offset(file, &mut buffer, test as usize * length);

                match less.compare(&buffer[..order], key) {
                    Ordering::Less => begin = test + 1,
                    Ordering::Greater => end = test,
                    Ordering::Equal => {
                        util::read_from_file_at_offset(file, &mut value, test as usize * length + order * mem::size_of::<WordIndex>());
                        if order != self.order() {
                            out.length = order;
                            out.backoff[order - 1] = value.backoff;
                            out.words[order - 1] = key[0];
                        }
                        break;
                    }
                }
            }
        }

        value.prob
    }
}
