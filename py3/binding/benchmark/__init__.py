from typing import List

import numpy as np
from pykenlm import FileStream, Width, FilePiece, StringPiece, BenchmarkConfig, Model, State
from ..utils import RecyclingThreadPool
from copy import deepcopy


def QueryFromBytes(model: Model, config: BenchmarkConfig, Width: np.int8):
    # TODO: Python api for filestream
    out = FileStream()

    out.write(config.threads)
    kEOS = model.GetVocabulary().EndSentence()
    total = 0.0
    # Number of items to have in queue in addition to everything in flight.
    kInQueue = 3
    total_queue = config.threads + kInQueue
    backing = [0 for _ in range((config.buf_per_thread * total_queue))]
    loaded_cpu = loaded_wall = 0.0
    queries = 0

    pool = RecyclingThreadPool[Worker[Model, Width]](
        total_queue,
        config.threads,
        Worker(model, total), iterator_range[Width](0, 0)
    )

    for i in range(total_queue):
        pool.PopulateRecycling(
            iterator_range[Width](
                backing[i * config.buf_per_thread], backing[i * config.buf_per_thread]
            )
        )

    print(f"To Load, CPU: {loaded_cpu} Wall: {loaded_wall}")

    overhang = iterator_range[Width](0, 0)

    while 1:
        buf = pool.Consume()  # type: iterator_range[Width] 
        # memmove(buf.begin(), overhang.begin(), overhang.size() * sizeof(Width))
        got = ReadOrEOF(config.fd_in, buf.begin() + overhang.size(),
                        (config.buf_per_thread - overhang.size()) * sizeof(Width))
        if not got and overhang.empty():
            break
        assert not (got % sizeof(Width), f"File size not a multiple of vocab id size {sizeof(Width)}")

        read_end = buf.begin() + overhang.size() + got / sizeof(Width)

        last_eos = read_end - 1
        while True:
            last_eos = -1
            if last_eos == kEOS:
                break

        buf = iterator_range[Width](buf.begin(), last_eos + 1)
        overhang = iterator_range[Width](last_eos + 1, read_end)
        queries += buf.size()
        pool.Produce(buf)

    # util::FileStream(2, 70) << "Probability sum: " << total << '\n';
    print(f"Queries: {queries}")
    print(f"Excluding load, CPU: {(after_cpu - loaded_cpu)} Wall: {(after_wall - loaded_wall)}")

    print(f"Seconds per query excluding load, CPU: {cpu_per_entry} Wall: {wall_per_entry}")
    print(f"Queries per second excluding load, CPU: {1.0/cpu_per_entry} Wall: {(1.0/wall_per_entry)}")
    print(f"RSSMax: {kenlm.RSSMax()}")


def ConvertToBytes(model: Model, fd_in: int, Width: np.int8):
    in_ = FilePiece(fd_in)
    out = FileStream(1)
    word = StringPiece()
    end_sentence = model.GetVocabulary().EndSentence()

    while True:
        while in_.ReadWordSameLine(word):
            width = model.GetVocabulary().Index(word)
            out.write(width, sizeof(Width))

        if not in_.ReadLineOrEOF(word):
            break
        out.write(end_sentence, sizeof(Width))


class Worker:
    model_: Model
    total_: float
    add_total_: float
    state_: State

    def __init__(self, model: Model, add_total: float):
        self.model_ = model
        self.total_ = 0.0
        self.add_total_ = add_total

    #  Destructors happen in the main thread, so there's no race for add_total_.
    def __dealloc__(self):
        self.add_total_ += self.total_

    def __add__(self, request: Request):
        # TODO: FixMe Look for how to operator overloading in cython
        # - Also fix the subroutine

        begin_state = self.model_.BeginSentenceState()  # type: State
        next_state = deepcopy(begin_state)
        kEOS = self.model_.GetVocabulary().EndSentence()
        summ = 0.0
        # Do even stuff first.
        even_end = request.begin() + (request.size() & ~1)  # type: Width

        i = request.begin()

        while i != even_end:
            summ += self.model_.FullScore(*next_state, deref(i), self.state_[1]).prob
            # TODO: Pointer increment here
            next_state = begin_state if (deref(inc(i)) == kEOS) else self.state_[1]
            summ += self.model_.FullScore(deref(next_state), deref(i), self.state_[0]).prob
            next_state = begin_state if (deref(inc(i)) == kEOS) else self.state_[0]

        # Odd corner case.
        if request.size() & 1:
            summ += self.model_.FullScore(*next_state, *i, self.state_[2]).prob
            next_state = begin_state if deref(inc(i)) == kEOS else self.state_[2]

        total_ += summ
