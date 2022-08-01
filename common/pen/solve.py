from logging import Logger
from multiprocessing import Queue, Process
from queue import Empty
from typing import List, Dict, Union, Set, Tuple, Optional

from sympy import Number, Symbol, Eq, Expr, solve

"""
*************************
*** Utility functions ***
*************************
"""


def _wrap_key_as_symbol(x: Dict[Union[str, Symbol], Number]) -> Dict[Symbol, Number]:
    """
    (Internal purpose) Wraps key in a dictionary as real-valued symbol.
    Instead of using this function, sympy cannot solve an equation
    as it thinks Symbol('x') != Symbol('x', real=True, positive=True).

    :param x: a dictionary whose keys will be wrapped.
    :return: A wrapped dictionary.
    """
    return {(Symbol(k, real=True) if type(k) is str else k): v
            for k, v in x.items()}


"""
***************************
*** Evaluation function ***
***************************

(1) Functions for evaluating Sympy-like equation.
(2) Functions for solving EquationSystem
(3) Class for automatic answer checking
"""


def _substitute_numbers(sympy_eqn: Expr, numbers: Dict[str, Number] = None) -> Expr:
    """
    (Internal purpose) Substitute numbers with given dictionary.

    :param sympy.Expr sympy_eqn: Equation to be evaluated.
    :param Dict[str, Number] numbers: Dictionary mapping from symbol name to decimal value.
    :rtype: sympy.Expr
    :return: Equation without given number symbols.
    """
    # Replace numbers(N0, N1, ..., T0, T1, ...)
    if numbers:
        sympy_eqn = sympy_eqn.subs(_wrap_key_as_symbol(numbers))

    return sympy_eqn


def _solve_equation_system(recv: Queue, send: Queue):
    """
    Evaluate sympy-like equations in the equation system and solve it in Real-number domain.

    :param Queue send: Queue for sending the result.
        This function will generate Dict[sympy.Expr, sympy.Expr] for the result of solving given equation.
    :param Queue recv: Queue for receiving the equations to be computed
    """
    while True:
        try:
            # Receive an object
            received_object = recv.get(block=True, timeout=600)
            # Wait 600 seconds for messages
        except Empty:
            continue
        except Exception as e:
            send.put(([], e))
            continue

        if not received_object:
            # Break the loop if received_object is false.
            break

        try:
            # Read received object
            system, numbers = received_object

            # Split system into equations and conditions
            system_of_equations = [_substitute_numbers(exp, numbers)
                                   for exp in system if isinstance(exp, Eq)]
            system_of_condition = [_substitute_numbers(exp, numbers)
                                   for exp in system if not isinstance(exp, Eq)]

            # Solve the equation
            candidates = solve(system_of_equations, dict=True)

            # Collect answers that satisfy given conditions.
            answers = []
            for candidate in candidates:
                if any(not v.is_real for v in candidate.values()):
                    # Ignore the solution when it is a solution with real numbers.
                    continue

                if all(cond.subs(candidate) for cond in system_of_condition):
                    # Ignore the solution if it fails to meet one of the condition.
                    answers.append(candidate)

            send.put((answers, None))
        except Exception as e:
            send.put(([], e))

    send.close()
    recv.close()


class Solver(object):
    """
    Class for answer checking purposes.
    """

    def __init__(self, error_limit: float = 1E-3, time_limit: float = 5, logger: Logger = None):
        """
        Class for evaluating answers

        :param float error_limit:
            the maximum amount of acceptable error between the result and the answer (default 1E-1)
        :param float time_limit:
            maximum amount of allowed time for computation in seconds (default 5)
        """

        self.error_limit = error_limit
        self.time_limit = time_limit

        self.solver_process = None
        self.to_solver = None
        self.from_solver = None
        self.logger = logger
        self._start_process()

    def _info(self, *args, **kwargs):
        if self.logger:
            self.logger.info(*args, **kwargs)

    def _err(self, *args, **kwargs):
        if self.logger:
            self.logger.error(*args, **kwargs)

    def _start_process(self):
        """
        Begin child process for running sympy
        """
        try:
            recv = Queue(maxsize=4)
            send = Queue(maxsize=4)
            self.solver_process = Process(target=_solve_equation_system, name='SympySolver', args=(send, recv))
            self.to_solver = send
            self.from_solver = recv
            self.solver_process.start()
        except Exception as e:
            self._err('Failed to start solver process', exc_info=e)
            pass

    def close(self):
        """
        Terminate child process for sympy
        """
        try:
            child_pid = self.solver_process.pid
            self._info('Sending terminate signal to solver (PID: %s)....', child_pid)
            self.to_solver.put(False)
            self._info('Closing solver queues (PID: %s)....', child_pid)
            self.to_solver.close()
            self.from_solver.close()

            if self.solver_process.is_alive():
                self._info('Kill the solver process (PID: %s)....', child_pid)
                self.solver_process.kill()
        except Exception as e:
            self._err('Failed to kill solver process', exc_info=e)
            pass

    def _restart_process(self):
        """
        Restart child process for sympy
        """
        self.close()
        self._start_process()

    def check_answer(self, expected: List[Dict[str, Number]], result: List[Dict[str, Optional[Number]]]) -> bool:
        """
        Verify whether the answer is equivalent to the obtained result.

        :param List[Dict[str,Number]] expected:
            List of the expected answer dictionaries given in this problem
        :param Set[Dict[str,Optional[Number]]] result:
            Set of pairs obtained by evaluating the formula for this problem
        :rtype: bool
        :return: True if both are the same.
        """

        # Now, the result should not any free variables. If so, return false.
        if any(x is None for candidate in result for x in candidate.values()):
            return False

        # Extract the keys
        varnames_sorted = sorted(expected[0].keys())

        try:
            # Convert dictionaries into tuples, whose ordering follows varnames_sorted.
            expected = [tuple([answer[v] for v in varnames_sorted])
                        for answer in expected]
            result = [tuple([candidate[v] for v in varnames_sorted])
                      for candidate in result]

            # Check whether expected tuples == result tuples, considering underflow errors.
            while True:
                # Remove matched pair
                answer = expected[0]
                matched = -1
                for p in range(len(result)):
                    if all(abs(a - c) <= self.error_limit for a, c in zip(answer, result[p])):
                        matched = p
                        break

                if matched == -1:
                    # There is an expected pair that is not in the results.
                    return False
                else:
                    expected = expected[1:]
                    result = result[:matched] + result[matched+1:]

                if len(expected) == 0:
                    # All the pairs are matched
                    return True
                if len(result) == 0:
                    # There is an expected pair that is not in the results.
                    return False
        except KeyError:
            # Case when some questioned variables do not appear in the result
            return False

    def solve(self, system: List[Expr],
              numbers: Dict[str, Number]) -> Tuple[List[Dict[str, Optional[Number]]], Optional[Exception]]:
        """
        Evaluate current equation system

        :param List[sympy.Expr] system:
            System of equations including equation and comparison conditions.
        :param Dict[str,Number] numbers:
            Dictionary of mapping from number symbol to decimal value.
        :rtype: Tuple[List[Dict[str, Optional[Number]]], Optional[Exception]]
        :return:
            List of dictionaries, and exception if occurred
            Each dictionary specifies possible value combinations for given unknown variables.
        """
        solution = []
        try:
            self.to_solver.put((system, numbers))
            solution, exception = self.from_solver.get(timeout=self.time_limit)
        except Exception as e:
            exception = e
            self._info('Attempt to replace solver process since pipe is broken...')
            self._restart_process()
            pass

        if exception:
            return [], exception

        # Convert value into Decimal values.
        solution = [{key.name: value if isinstance(value, Number) else (value.evalf(10) if value.is_constant() else None)
                     for key, value in candidate.items()}
                    for candidate in solution]
        # Filter empty dictionaries
        solution = [candidate
                    for candidate in solution if candidate]

        return solution, None


__all__ = ['Solver']
