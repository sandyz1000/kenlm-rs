from typing import List, Iterator
import numpy as np
import io
from pykenlm import ChainConfig, Chain, FilePiece as KLMFilePiece


def GuessPhysicalMemory() -> int:
    """_summary_

    :return _type_: _description_
    """


def ParseSize(arg: str) -> int:
    """_summary_

    :param _type_ arg: _description_
    :return _type_: _description_
    """
    

def ParseBitCount(inp_: str) -> int:
    """_summary_

    :param _type_ inp_: _description_
    :return _type_: _description_
    """


class StringPiece:
    sp: 'StringPiece'

    def __init__(self, string_: str):
        self.sp = StringPiece(string_)

    def __dealloc__(self):
        del self.sp

    def __iter__(self):
        """
        An string iterator to iterate over string
        Returns
        -------

        """
        yield


class FilePiece:
    file_piece: KLMFilePiece
    min_buffer: int

    def __init__(self, fp: io.FileIO, show_progress: bool = False, min_buffer: int = 1048576):
        self.fp = fp
        self.show_progress = show_progress
        self.min_buffer = min_buffer

    def begin(self) -> Iterator:
        pass

    def end(self) -> Iterator:
        pass

    def peek(self) -> str:
        pass

    def ReadDelimited(self, delim: List[bool] = np.zeros((256,), dtype=bool)) -> StringPiece:
        pass

    def ReadWordSameLine(self, to: StringPiece, delim: List[bool] = np.zeros((256,), dtype=bool)) -> bool:
        pass

    def ReadLine(self, delim: str = '\n', strip_cr: bool = True) -> StringPiece:
        pass

    def ReadLineOrEOF(self, to: StringPiece, delim: str = '\n', strip_cr: bool = True) -> bool:
        pass

    def ReadFloat(self) -> np.float:
        pass

    def ReadDouble(self) -> np.double:
        pass

    def ReadLong(self) -> np.long:
        pass

    def ReadULong(self) -> np.uint:
        pass

    def SkipSpaces(self, delim: List[bool] = np.zeros((256,), dtype=bool)) -> None:
        # Skip spaces defined by isspace.
        pass

    def Offset(self) -> np.intc:
        pass

    def FileName(self) -> str:
        pass

    def UpdateProgress(self) -> None:
        # Force a progress update.
        pass


class RecyclingThreadPool:
    queue: int
    workers: int
    handler_construct: Construct
    poison: Request

    def __init__(
        self,
        queue: int, 
        workers: int, 
        handler_construct: Construct,
        poison: Request, 
        HandlerT: HandlerType  = None
    ):
        self._pool = RecyclingThreadPool[HandlerT](queue, workers, handler_construct, poison)
    
    def PopulateRecycling(self, request: Request):
        self._pool.PopulateRecycling(request)

    def Consume(self) -> Request:
        self._pool.Consume()

    def Produce(self, request: Request):
        self._pool.Produce(request)
