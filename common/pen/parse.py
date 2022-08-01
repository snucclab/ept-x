from common.const.operand import *
from common.const.operator import *
from common.const.pad import PAD_ID

from .pattern import NUMBER_OR_FRACTION_PATTERN

RELATIONS = {
    '=': sympy.Eq,
    '>': sympy.Gt,
    '<': sympy.Lt,
    '>=': sympy.Ge,
    '<=': sympy.Le
}
OPERATORS = {
    '+': sympy.Add,
    '-': lambda a, b: a - b,
    '*': sympy.Mul,
    '/': lambda a, b: a / b,
    '^': sympy.Pow
}
OPERATOR_PRECEDENCE = {
    '+': 1,
    '-': 1,
    '*': 2,
    '/': 2,
    '^': 3
}
RELATION_CLASSES = tuple(RELATIONS.values())


def get_operator_precedence(op: str) -> int:
    if op in '()':
        # Next highest precedence for the parentheses
        return 9999

    if op in RELATIONS:
        # Set the lowest precedence for the relations
        return 0

    # Otherwise return the predefined values, starting from 1.
    return OPERATOR_PRECEDENCE[op]


def parse_infix(equation: str, variables: list, offset=0):
    # Storage for expressions
    expressions = []
    # Stack for operators
    operator_stack = []
    # Storage for operands
    operands_stack = []

    def add_expression():
        nonlocal operands_stack, offset

        # Pop the operator
        operator = OPR_TOKENS.index(operator_stack.pop())

        # Pop the last two operands (all operators are binary.)
        operands = operands_stack[-2:]
        operands_stack = operands_stack[:-2]

        # Sort argument for commutable operator (order: constant - number - variable - previous_result)
        if OPR_VALUES[operator][KEY_COMMUTATIVE]:
            operands = sorted(operands)

        # Add current operator
        # Register the expression
        if len(operands) < OPR_MAX_ARITY:
            operands = operands + [PAD_ID] * (OPR_MAX_ARITY - len(operands))

        expression = (operator, *operands)
        expressions.append(expression)

        # Register the new expression as an operand.
        operands_stack.append(RES_BEGIN + offset)
        offset += 1

    # Trim the whitespaces
    equation = equation.split()
    for token in equation:
        token = token.strip()

        if token.startswith(NUM_PREFIX):
            # We found a number.
            index = int(token[PREFIX_LEN:])
            operands_stack.append(NUM_BEGIN + index)
            continue

        if token in variables:
            # We found a variable
            operands_stack.append(RES_BEGIN + variables.index(token))
            continue

        num_match = NUMBER_OR_FRACTION_PATTERN.fullmatch(token)
        if num_match:
            # We found a constant.
            operands_stack.append(CON_TOKENS.index(token))
            continue

        # Otherwise, it is an operator: One of +, *, ^, =, (, ). Others are not permitted.
        if token == ')':
            # Pop until find the opening paren '('
            while operator_stack:
                if operator_stack[-1] == '(':
                    # Discard the matching '('
                    operator_stack.pop()
                    break
                else:
                    add_expression()
        else:
            precedence = get_operator_precedence(token)

            # '(' has the highest precedence when in the input string.
            while operator_stack:
                # Pop until the top < current_precedence.
                # '(' has the lowest precedence in the stack.
                top = operator_stack[-1]
                top_prec = get_operator_precedence(top) if top != '(' else -9999
                if top_prec < precedence:
                    break
                else:
                    add_expression()

            operator_stack.append(token)

    while operator_stack:
        add_expression()

    # Return current sequence of generating the equation
    return expressions
