//


pub type WordIndex = usize;

pub trait Vocabulary {
    fn begin_sentence(&self) -> WordIndex;
    fn end_sentence(&self) -> WordIndex;
    fn not_found(&self) -> WordIndex;

    fn index(&self, str: &str) -> WordIndex;

    fn index_from_string(&self, str: &String) -> WordIndex {
        self.index(str.as_str())
    }

    fn index_from_str(&self, str: &str) -> WordIndex {
        self.index(str)
    }

    fn set_special(&mut self, begin_sentence: WordIndex, end_sentence: WordIndex, not_found: WordIndex);
}

pub struct BaseVocabulary {
    begin_sentence: WordIndex,
    end_sentence: WordIndex,
    not_found: WordIndex,
}

impl Vocabulary for BaseVocabulary {
    fn begin_sentence(&self) -> WordIndex {
        self.begin_sentence
    }

    fn end_sentence(&self) -> WordIndex {
        self.end_sentence
    }

    fn not_found(&self) -> WordIndex {
        self.not_found
    }

    fn index(&self, _str: &str) -> WordIndex {
        unimplemented!("This method should be implemented by derived classes.")
    }

    fn set_special(&mut self, begin_sentence: WordIndex, end_sentence: WordIndex, not_found: WordIndex) {
        self.begin_sentence = begin_sentence;
        self.end_sentence = end_sentence;
        self.not_found = not_found;
    }
}

impl BaseVocabulary {
    pub fn new() -> Self {
        Self {
            begin_sentence: 0,
            end_sentence: 0,
            not_found: 0,
        }
    }

    pub fn with_special(begin_sentence: WordIndex, end_sentence: WordIndex, not_found: WordIndex) -> Self {
        let mut vocab = Self::new();
        vocab.set_special(begin_sentence, end_sentence, not_found);
        vocab
    }
}


pub struct ChildVocabulary {
    base: BaseVocabulary,
    vocab: nplm::Vocabulary,
    null_word: WordIndex,
}

impl ChildVocabulary {
    pub fn new(vocab: nplm::Vocabulary) -> Self {
        // let base = BaseVocabulary::new(
        //     vocab.lookup_word("<s>"),
        //     vocab.lookup_word("</s>"),
        //     vocab.lookup_word("<unk>"),
        // );
        // let null_word = vocab.lookup_word("<null>");
        // Vocabulary { base, vocab, null_word }
        todo!()
    }

    pub fn index(&self, str: &str) -> WordIndex {
        self.vocab.lookup_word(str)
    }
}

pub struct Backend {
    lm: nplm::NeuralLM,
    ngram: Eigen::Matrix<i32, Eigen::Dynamic, 1>,
}

impl Backend {
    pub fn new(from: &nplm::NeuralLM, cache_size: usize) -> Self {
        let mut lm = from.clone();
        lm.set_cache(cache_size);
        let ngram = Eigen::Matrix::new(from.get_order(), 1);
        Backend { lm, ngram }
    }

    pub fn lm(&self) -> &nplm::NeuralLM {
        &self.lm
    }

    pub fn lm_mut(&mut self) -> &mut nplm::NeuralLM {
        &mut self.lm
    }

    pub fn staging_ngram(&mut self) -> &mut Eigen::Matrix<i32, Eigen::Dynamic, 1> {
        &mut self.ngram
    }

    pub fn lookup_from_staging(&mut self) -> f64 {
        self.lm.lookup_ngram(&self.ngram)
    }

    pub fn order(&self) -> i32 {
        self.lm.get_order()
    }
}

pub struct Model {
    base_instance: Arc<nplm::NeuralLM>,
    vocab: Vocabulary,
    cache_size: usize,
    backend: Option<Backend>,
    null_word: WordIndex,
}

impl Model {
    pub fn new(file: &str, cache: usize) -> Self {
        let base_instance = Arc::new(load_nplm(file));
        let vocab = Vocabulary::new(base_instance.get_vocabulary());
        let null_word = base_instance.lookup_word("<null>");
        let mut model = Model {
            base_instance,
            vocab,
            cache_size: cache,
            backend: None,
            null_word,
        };
        model.init();
        model
    }

    fn init(&mut self) {
        self.base_instance.set_log_base(10.0);
        let begin_sentence = State::new(
            &vec![self.base_instance.lookup_word("<s>"); NPLM_MAX_ORDER - 1],
        );
        let null_context = State::new(
            &vec![self.null_word; NPLM_MAX_ORDER - 1],
        );
        self.init_context(begin_sentence, null_context);
    }

    fn init_context(&self, begin_sentence: State, null_context: State) {
        // Initialize context states
    }

    pub fn recognize(name: &str) -> bool {
        match util::open_read_or_throw(name) {
            Ok(mut file) => {
                let mut magic_check = [0u8; 16];
                if let Ok(_) = file.read_exact(&mut magic_check) {
                    let nnlm_magic = b"\\config\nversion ";
                    magic_check == nnlm_magic[..]
                } else {
                    false
                }
            }
            Err(_) => false,
        }
    }

    pub fn full_score(
        &mut self,
        from: &State,
        new_word: WordIndex,
        out_state: &mut State,
    ) -> FullScoreReturn {
        let backend = self.get_backend();
        for i in 0..backend.order() - 1 {
            backend.staging_ngram()[i as usize] = from.words[i as usize];
        }
        backend.staging_ngram()[backend.order() as usize - 1] = new_word;
        let prob = backend.lookup_from_staging();
        let ngram_length = backend.order();
        out_state.words.copy_from_slice(&from.words[1..backend.order() as usize - 1]);
        out_state.words[backend.order() as usize - 2] = new_word;
        out_state.words[backend.order() as usize - 1..].fill(0);
        FullScoreReturn { prob, ngram_length }
    }

    pub fn full_score_forgot_state(
        &mut self,
        context_rbegin: &[WordIndex],
        context_rend: &[WordIndex],
        new_word: WordIndex,
        out_state: &mut State,
    ) -> FullScoreReturn {
        let state_length = std::cmp::min(self.order() as usize - 1, context_rend.len() - context_rbegin.len());
        let mut state = State::new(&vec![self.null_word; self.order() as usize - 1]);
        state.words[self.order() as usize - 1 - state_length..].copy_from_slice(&context_rbegin[..state_length]);
        self.full_score(&state, new_word, out_state)
    }

    fn get_backend(&mut self) -> &mut Backend {
        if self.backend.is_none() {
            self.backend = Some(Backend::new(&self.base_instance, self.cache_size));
        }
        self.backend.as_mut().unwrap()
    }
}
