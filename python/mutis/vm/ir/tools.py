from typing import List
from mutis.vm.ir.program import VirtualMachineProgram
from .functor import VirtualMachineVisitor
from .inst import Instruction


class InstructionCollector(VirtualMachineVisitor):
    def __init__(self):
        super().__init__()
        self.instructions: List[Instruction] = []

    def visit_Instruction(self, inst: Instruction):
        self.instructions.append(inst)


def collect_instructions(prog: VirtualMachineProgram) -> List[Instruction]:
    collector = InstructionCollector()
    collector(prog)
    return collector.instructions
