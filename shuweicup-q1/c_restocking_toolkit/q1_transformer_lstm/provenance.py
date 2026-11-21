"""Record project file opens phase by phase, including whether a checkpoint was opened for reading."""
import os
import sys
from contextlib import contextmanager
from pathlib import Path

from study_paths import HERE, OUT, RELEASE, ROOT


class ReadAudit:
    def __init__(self):
        self.phase = None
        self.events = {}
        self.comparators = set()
        self.workspace = next((p.parent for p in ROOT.parents if p.name == "ShuWeiCup"), ROOT)
        sys.addaudithook(self._on_open)

    def _on_open(self, event, args):
        if event != "open" or self.phase is None or not isinstance(args[0], (str, bytes)):
            return
        path = Path(os.path.abspath(os.fsdecode(args[0])))
        weights = path.suffix.lower() in {".pt", ".pth", ".ckpt", ".safetensors"}
        if not weights and not path.is_relative_to(self.workspace):
            return
        path = path.resolve()
        flags = args[2] or 0
        write = (flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)) != 0
        read = (flags & os.O_WRONLY) == 0
        owned = path.is_relative_to(HERE) or path.is_relative_to(OUT)
        release = path.is_relative_to(RELEASE)
        comparator = path in self.comparators
        if not owned and not ((release or comparator) and not write):
            raise RuntimeError(f"{self.phase} tried to open an undeclared project input: {path}")
        if weights and read and (self.phase in {"decode", "random_fit"} or not path.is_relative_to(OUT)):
            raise RuntimeError(f"{self.phase} tried to read a checkpoint outside its independent fit: {path}")
        seen = self.events[self.phase]
        if weights and read:
            seen["new_checkpoint_reads"].add(path.name)
        if read and release:
            seen["official_inputs"].add(path.relative_to(RELEASE).as_posix())
        if read and comparator:
            seen["archived_result_reads"].add(path.name)

    @contextmanager
    def during(self, phase, comparators=()):
        self.events[phase] = dict(new_checkpoint_reads=set(), official_inputs=set(), archived_result_reads=set())
        self.comparators = {p.resolve() for p in comparators}
        self.phase = phase
        try:
            yield
        finally:
            self.phase = None
            self.comparators.clear()

    def result(self):
        return {phase:{kind:sorted(files) for kind,files in row.items()} for phase,row in self.events.items()}
