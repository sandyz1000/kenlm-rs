


pub trait Search {

    fn fast_make_node(&self, begin: WordIndex, end: WordIndex, node: &Node) -> bool;
    fn lookup_longest(&self, word: WordIndex, node: &Node) -> LongestPointer;
    fn order(&self) -> &str;

    fn update_config_from_binary(
        &self,
        file: &BinaryFormat,
        counts: &Vec<u64>,
        offset: u64,
        config: &Config,
    );

    fn size(&self, counts: &Vec<u64>, config: &Config) -> u64;

    fn setup_memory(&self, start: u8, counts: &Vec<u64>, config: &Config) -> &u8;

    fn initialize_from_arpa(
        &self,
        file: &str,
        f: &FilePiece,
        counts: &Vec<u64>,
        config: &Config,
        vocab: &SortedVocabulary,
        backing: &BinaryFormat,
    );

    fn lookup_unigram(
        &self,
        word: WordIndex,
        node: &Node,
        independent_left: &bool,
        extend_left: &u64,
    ) -> UnigramPointer;

    fn unknown_unigram(&self) -> ProbBackoff;

    fn unpack(&self, extend_pointer: u64, extend_length: &str, node: &Node) -> MiddlePointer;

    fn lookup_middle(
        &self,
        order_minus_2: &str,
        word: WordIndex,
        node: &Node,
        independent_left: &bool,
        extend_left: &u64,
    ) -> MiddlePointer;
}
