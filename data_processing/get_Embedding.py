"""利用tokenizer得到opcode sequence的embedding"""
"""利用tokenizer得到opcode sequence的embedding"""
from tools import opcode2embedding


file = ["reentrancy", "timestamp", "delegatecall", "integeroverflow",
            "SBaccess_control", "SBarithmetic", "SBdenial_of_service",
            "SBfront_running", "SBshort_address",
            "SBunchecked_low_level_calls", "normal", "normal_all"]
for name in file:
    opcode2embedding(name)
