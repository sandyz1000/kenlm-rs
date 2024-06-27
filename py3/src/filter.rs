
// use crate::kenlmrs::*;

use std::collections::HashMap;


// fn RunThreadedFilter() -> Option() {
//     if config.threads == 1  {
// 		RunFilter(in_lm, filter_, output)
// 	}
//     else {
// 		Threaded: Controller<Filter, OutputBuffer, Output>

// 		let threading = Threaded(config.batch_size, config.threads * 2, config.threads, filter_, output)

// 		RunFilter(in_lm, threading, output)
//     }
// }

// fn RunContextFilter() -> Option() {

// }

// fn DispatchBinaryFilter() -> Option() {

// }


// pub fn DispatchFilterModes(
// 	config: &FilterConfig, in_vocab: IStream, in_lm: &FilePiece,
// 	out_name: &str, format_type: Format
// ) -> Result<()> {
// 	// ctypedef unordered_map[string, vector[U_int]] Words
    
//     // substrings: Substrings
//     // unordered_map[string, vector[uint64_t]] 
//     let words: HashMap<String, Vec<u64>> = HashMap::new();

// 	if config.mode == ct.FilterMode.MODE_MULTIPLE {
// 		out = (out_name, ReadMultiple(in_vocab, substrings));

// 		if config.phrase {
// 			RunContextFilter(
// 				config, in_lm, PhraseFilter(substrings), out,
// 			    dtype=[format_type, MultipleOutputBuffer]
// 			);
//         }
// 		else {
// 			RunContextFilter(
// 				config, in_lm, VocabFilter(words), out,
// 			    dtype=[Format, MultipleOutputBuffer]
// 			);
//         }
//     }
// 	out(out_name)

// 	if config.mode == FilterMode.MODE_COPY {
// 		tp.Format.Copy(in_lm, out);
// 		return None;
//     }

// 	if config.mode == FilterMode.MODE_SINGLE {
// 		// Words words
// 		ReadSingle(in_vocab, words);
// 		DispatchBinaryFilter[Format, Single](config, in_lm, Single(words), out);
// 		return Ok();
//     }
	
//     if config.mode == FilterMode.MODE_UNION {
// 		if config.phrase {
// 			// substrings: Substrings
// 			ReadMultiple(in_vocab, substrings);
// 			DispatchBinaryFilter<Format, Union>(config, in_lm, Union(substrings), out);
//         }
// 		else {
// 			// Words words
// 			ReadMultiple(in_vocab, words);
// 			DispatchBinaryFilter<Format, Union>(config, in_lm, Union(words), out);
//         }
//     }
// }